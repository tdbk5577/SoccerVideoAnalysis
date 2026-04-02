#!/usr/bin/env python3
"""
Ball tracking v3 - physics-aware state machine with adaptive Kalman.

Key improvements over v2:
1. Freeze at seed for first FREEZE_FRAMES frames (no detections, hold position)
2. 3-frame accumulated motion (compare curr vs frame_{i-3}) for better slow-ball detection
3. Adaptive Kalman measurement noise: high when slow (ignore detections), low when fast (trust them)
4. On loss: HOLD last confirmed position, don't extrapolate off-field
5. Unbiased recovery: find best candidate anywhere on field, ignore wrong trajectory
6. Kick detection: look for large motion near last position to re-enter active mode

Usage:
  .venv/bin/python3 analyze_tracking_v3.py --video Test.mp4 --labels ball_labels.json
  .venv/bin/python3 analyze_tracking_v3.py --video Test.mp4 --labels ball_labels.json --frames-dir /tmp/soccer_v2_frames
"""

import argparse
import json
import os
import subprocess
import time
from itertools import product as iproduct
import random

import cv2
import numpy as np

FPS = 10.0
W, H = 1920, 1080

FREEZE_FRAMES = 25          # hold seed without any detection
FIELD_Y_MIN   = int(0.20 * H)
FIELD_Y_MAX   = int(0.82 * H)
BALL_AREA_MIN = 5
BALL_AREA_MAX = 800
K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# ─────────────────────────────────────────────
# Kalman
# ─────────────────────────────────────────────

def make_kf(px, py, proc_noise=2.0, meas_noise=15.0):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix  = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kf.transitionMatrix   = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    kf.processNoiseCov    = np.eye(4, dtype=np.float32) * proc_noise
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * meas_noise
    kf.errorCovPost       = np.eye(4, dtype=np.float32) * 100
    kf.statePost          = np.array([[px],[py],[0.0],[0.0]], np.float32)
    return kf

def kf_pred(kf):
    p = kf.predict()
    return int(np.clip(p[0,0],0,W-1)), int(np.clip(p[1,0],0,H-1))

def kf_update(kf, px, py, meas_noise_override=None):
    if meas_noise_override is not None:
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * meas_noise_override
    kf.correct(np.array([[float(px)],[float(py)]], np.float32))

def kf_speed(kf):
    vx, vy = float(kf.statePost[2,0]), float(kf.statePost[3,0])
    return (vx**2+vy**2)**0.5


# ─────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────

def accumulated_motion(frames, idx, step, k=3):
    """
    Compare frame idx with frame idx - k*step (k frames back in direction of travel).
    Larger k → catches slower ball movement.
    Returns motion mask (full frame).
    """
    ref_idx = idx - k * step
    if not (0 <= ref_idx < len(frames)):
        return None
    curr = cv2.imread(frames[idx])
    ref  = cv2.imread(frames[ref_idx])
    if curr is None or ref is None:
        return None
    gc = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(ref,  cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gc, gr)
    _, mot = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
    mot = cv2.morphologyEx(mot, cv2.MORPH_OPEN,  K3)
    mot = cv2.dilate(mot, K5, iterations=1)
    return mot, curr, gc


def find_candidates(mot_mask, curr_gray, curr_bgr, pred_px, pred_py, radius,
                    bright_thresh=75):
    """
    Find ball candidates within radius of (pred_px, pred_py).
    Uses motion mask + non-green + brightness.
    Returns sorted list of (px, py, score).
    """
    x1 = max(0, pred_px-radius); x2 = min(W, pred_px+radius)
    y1 = max(0, pred_py-radius); y2 = min(H, pred_py+radius)

    roi_mot  = mot_mask[y1:y2, x1:x2] if mot_mask is not None else None
    roi_gray = curr_gray[y1:y2, x1:x2]
    roi_bgr  = curr_bgr[y1:y2, x1:x2]

    if roi_gray.size == 0:
        return []

    hsv   = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([30,25,25]), np.array([90,255,255]))
    ng    = cv2.bitwise_not(green)
    br    = (roi_gray > bright_thresh).astype(np.uint8) * 255

    if roi_mot is not None:
        cand = cv2.bitwise_and(cv2.bitwise_and(roi_mot, ng), br)
    else:
        cand = cv2.bitwise_and(ng, br)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, K3)

    cnts, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in cnts:
        area = cv2.contourArea(c)
        if not (BALL_AREA_MIN <= area <= BALL_AREA_MAX):
            continue
        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
        fx = x1+cx; fy = y1+cy
        if not (FIELD_Y_MIN <= fy <= FIELD_Y_MAX):
            continue
        dist = ((fx-pred_px)**2+(fy-pred_py)**2)**0.5
        if dist > radius:
            continue
        p = cv2.arcLength(c, True)
        circ = 4*np.pi*area/(p**2) if p > 0 else 0
        patch = roi_gray[max(0,cy-5):cy+5, max(0,cx-5):cx+5]
        bv = float(np.mean(patch)) / 220.0 if patch.size > 0 else 0
        size_sc = np.exp(-((area-80)**2)/(2*150**2)) + 0.2
        sc = (circ+0.15) * bv * size_sc * (1.0 - dist/(radius+1))
        results.append((fx, fy, sc))

    # Brightness-only fallback (catches slow/stationary ball)
    if not results or (results and max(r[2] for r in results) < 0.04):
        ng_br = cv2.bitwise_and(roi_gray, roi_gray, mask=ng)
        blr   = cv2.GaussianBlur(ng_br.astype(np.float32), (7,7), 2)
        _, mv, _, ml = cv2.minMaxLoc(blr)
        if mv > 95:
            fx = x1+ml[0]; fy = y1+ml[1]
            if FIELD_Y_MIN <= fy <= FIELD_Y_MAX:
                dist = ((fx-pred_px)**2+(fy-pred_py)**2)**0.5
                if dist <= radius:
                    sc = (mv/255.0)*(1-dist/(radius+1))*0.30
                    results.append((fx, fy, sc))

    return sorted(results, key=lambda r: r[2], reverse=True)


def find_candidates_full_frame(curr_bgr, curr_gray, mot_mask, bright_thresh=70):
    """
    Unbiased full-frame scan for recovery.
    Returns sorted list of (px, py, score).
    """
    hsv   = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([30,25,25]), np.array([90,255,255]))
    ng    = cv2.bitwise_not(green)
    br    = (curr_gray > bright_thresh).astype(np.uint8)*255

    if mot_mask is not None:
        cand = cv2.bitwise_and(cv2.bitwise_and(mot_mask, ng), br)
    else:
        cand = cv2.bitwise_and(ng, br)

    # Field-only mask
    field = np.zeros_like(cand)
    field[FIELD_Y_MIN:FIELD_Y_MAX, :] = 255
    cand = cv2.bitwise_and(cand, field)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, K3)

    cnts, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in cnts:
        area = cv2.contourArea(c)
        if not (BALL_AREA_MIN <= area <= BALL_AREA_MAX):
            continue
        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
        if not (FIELD_Y_MIN <= cy <= FIELD_Y_MAX):
            continue
        p = cv2.arcLength(c, True)
        circ = 4*np.pi*area/(p**2) if p > 0 else 0
        patch = curr_gray[max(0,cy-5):cy+5, max(0,cx-5):cx+5]
        bv = float(np.mean(patch))/220.0 if patch.size > 0 else 0
        size_sc = np.exp(-((area-80)**2)/(2*150**2)) + 0.2
        sc = (circ+0.15) * bv * size_sc
        results.append((cx, cy, sc))

    return sorted(results, key=lambda r: r[2], reverse=True)[:20]


def detect_kick_near(frames, idx, step, last_px, last_py,
                     search_r=200, mot_k=3, min_disp=20):
    """
    Look for ball that has moved significantly from last_pos.
    Compare curr vs k frames back. Look for large motion blob near last_pos.
    Returns (px, py) or None.
    """
    result = accumulated_motion(frames, idx, step, k=mot_k)
    if result is None:
        return None
    mot, curr, gc = result

    x1 = max(0, last_px-search_r); x2 = min(W, last_px+search_r)
    y1 = max(0, last_py-search_r); y2 = min(H, last_py+search_r)
    roi_mot = mot[y1:y2, x1:x2]
    roi_gc  = gc[y1:y2, x1:x2]
    roi_bgr = curr[y1:y2, x1:x2]

    if roi_mot.size == 0:
        return None

    cnts, _ = cv2.findContours(roi_mot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None; best_sc = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if not (BALL_AREA_MIN <= area <= BALL_AREA_MAX):
            continue
        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
        fx = x1+cx; fy = y1+cy
        # Must have moved from last position
        disp = ((fx-last_px)**2+(fy-last_py)**2)**0.5
        if disp < min_disp:
            continue
        p = cv2.arcLength(c, True)
        circ = 4*np.pi*area/(p**2) if p > 0 else 0
        patch = roi_gc[max(0,cy-5):cy+5, max(0,cx-5):cx+5]
        bv = float(np.mean(patch))/220.0 if patch.size > 0 else 0
        sc = (circ+0.15) * bv * min(disp/50.0, 1.0)
        if sc > best_sc:
            best_sc = sc; best = (fx, fy)
    return best if best_sc > 0.05 else None


# ─────────────────────────────────────────────
# Main tracker
# ─────────────────────────────────────────────

FREEZE   = "FREEZE"
ACTIVE   = "ACTIVE"
HOLD     = "HOLD"
RECOVER  = "RECOVER"


def track_v3(frames, seed_frame, seed_pos,
             proc_noise=2.0,
             meas_noise_fast=10.0,
             meas_noise_slow=200.0,
             active_thresh=0.07,
             kick_speed=20.0,
             max_hold=20,
             max_recover=40,
             radius_base=60,
             mot_k=3):
    """
    State machine tracker v3.

    FREEZE  → hold seed, watch for kick
    ACTIVE  → Kalman + motion detection (adaptive radius)
    HOLD    → lost, hold last confirmed, full-frame scan each frame
    RECOVER → extended hold + scan, eventually give up
    """
    n = len(frames)
    positions = {seed_frame: seed_pos}

    def run_dir(start_idx, start_pos, step):
        kf            = make_kf(*start_pos, proc_noise=proc_noise,
                                meas_noise=meas_noise_fast)
        state         = FREEZE
        hold_count    = 0
        recover_count = 0
        last_confirmed = start_pos
        last_confirmed_idx = start_idx

        idx = start_idx + step
        while 0 <= idx < n:
            frames_from_start = abs(idx - start_idx)

            # --- Load frame data ---
            result = accumulated_motion(frames, idx, step, k=mot_k)
            if result is None:
                idx += step
                continue
            mot, curr_bgr, curr_gray = result

            # Kalman predict
            pred_px, pred_py = kf_pred(kf)
            speed = kf_speed(kf)

            # ── FREEZE ──────────────────────────────────────
            if state == FREEZE:
                # Hold seed position
                positions[idx] = start_pos
                # Update Kalman with seed (very high noise → essentially ignore)
                kf_update(kf, start_pos[0], start_pos[1],
                          meas_noise_override=meas_noise_slow)

                # Watch for the ball moving away from start_pos
                kick = detect_kick_near(frames, idx, step,
                                        start_pos[0], start_pos[1],
                                        search_r=150, mot_k=mot_k, min_disp=20)
                if kick is not None and frames_from_start >= FREEZE_FRAMES:
                    # Ball has been kicked — switch to active
                    kf = make_kf(*kick, proc_noise=proc_noise,
                                 meas_noise=meas_noise_fast)
                    positions[idx] = kick
                    last_confirmed = kick
                    last_confirmed_idx = idx
                    state = ACTIVE

            # ── ACTIVE ──────────────────────────────────────
            elif state == ACTIVE:
                # Adaptive radius
                radius = max(radius_base, min(220, int(speed * 2.5 + radius_base)))
                # Adaptive measurement noise
                mn = meas_noise_fast if speed > 3.0 else meas_noise_slow

                cands = find_candidates(mot, curr_gray, curr_bgr,
                                        pred_px, pred_py, radius)

                if cands and cands[0][2] >= active_thresh:
                    fx, fy, sc = cands[0]
                    positions[idx] = (fx, fy)
                    kf_update(kf, fx, fy, meas_noise_override=mn)
                    last_confirmed = (fx, fy)
                    last_confirmed_idx = idx
                    hold_count = 0
                else:
                    # Not found — hold last confirmed, DON'T extrapolate wildly
                    hold_count += 1
                    if hold_count <= 3:
                        # For very short losses, trust velocity briefly
                        positions[idx] = (pred_px, pred_py)
                    else:
                        # Hold last confirmed (don't let Kalman send us off-field)
                        positions[idx] = last_confirmed
                        kf_update(kf, last_confirmed[0], last_confirmed[1],
                                  meas_noise_override=meas_noise_slow)

                    if hold_count > max_hold:
                        state = HOLD
                        hold_count = 0

            # ── HOLD ────────────────────────────────────────
            elif state == HOLD:
                # Hold last confirmed and scan full frame for recovery
                positions[idx] = last_confirmed
                hold_count += 1

                # Full-frame unbiased scan
                all_cands = find_candidates_full_frame(curr_bgr, curr_gray, mot)
                if all_cands:
                    # Pick candidate closest to last_confirmed
                    # (unbiased: don't use bad Kalman prediction)
                    lx, ly = last_confirmed
                    best = min(all_cands,
                               key=lambda c: (c[0]-lx)**2+(c[1]-ly)**2)
                    dist = ((best[0]-lx)**2+(best[1]-ly)**2)**0.5
                    # Accept if within 300px of last confirmed AND high score
                    if dist < 300 and best[2] > 0.06:
                        pos = (best[0], best[1])
                        kf = make_kf(*pos, proc_noise=proc_noise,
                                     meas_noise=meas_noise_fast)
                        positions[idx] = pos
                        last_confirmed = pos
                        last_confirmed_idx = idx
                        state = ACTIVE
                        hold_count = 0

                if hold_count > max_hold:
                    state = RECOVER
                    recover_count = 0

            # ── RECOVER ─────────────────────────────────────
            elif state == RECOVER:
                # Full-frame scan, expanding search radius from last confirmed
                positions[idx] = last_confirmed  # default
                recover_count += 1

                # Try accumulating more motion (compare vs further back)
                long_result = accumulated_motion(frames, idx, step, k=min(mot_k*2, 6))
                use_mot = long_result[0] if long_result else mot
                use_curr = long_result[1] if long_result else curr_bgr
                use_gray = long_result[2] if long_result else curr_gray

                all_cands = find_candidates_full_frame(use_curr, use_gray, use_mot,
                                                       bright_thresh=65)
                if all_cands:
                    lx, ly = last_confirmed
                    # Gradually expand acceptance radius
                    accept_r = min(300 + recover_count * 10, 700)
                    good = [(c, ((c[0]-lx)**2+(c[1]-ly)**2)**0.5)
                            for c in all_cands if ((c[0]-lx)**2+(c[1]-ly)**2)**0.5 < accept_r]
                    if good:
                        best_c, dist = min(good, key=lambda x: x[1])
                        if best_c[2] > 0.05:
                            pos = (best_c[0], best_c[1])
                            kf = make_kf(*pos, proc_noise=proc_noise,
                                         meas_noise=meas_noise_fast)
                            positions[idx] = pos
                            last_confirmed = pos
                            last_confirmed_idx = idx
                            state = ACTIVE
                            recover_count = 0

                if recover_count > max_recover:
                    # Give up — leave as None
                    positions.pop(idx, None)

            idx += step

    run_dir(seed_frame, seed_pos, +1)
    run_dir(seed_frame, seed_pos, -1)
    return positions


# ─────────────────────────────────────────────
# Seed detection
# ─────────────────────────────────────────────

def find_seed(frames, avg_n=25):
    x1r, x2r = int(0.35*W), int(0.65*W)
    y1r, y2r = int(0.33*H), int(0.67*H)
    stack = []
    for i in range(min(avg_n, len(frames))):
        f = cv2.imread(frames[i])
        if f is not None:
            stack.append(f[y1r:y2r, x1r:x2r].astype(np.float32))
    if stack:
        avg  = np.mean(stack, axis=0).astype(np.uint8)
        gray = cv2.cvtColor(avg, cv2.COLOR_BGR2GRAY)
        blr  = cv2.GaussianBlur(gray, (5,5), 1)
        circles = cv2.HoughCircles(blr, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                                   param1=30, param2=8, minRadius=3, maxRadius=13)
        if circles is not None:
            circles = np.round(circles[0]).astype(int)
            roi_cx = (x2r-x1r)/2; roi_cy = (y2r-y1r)/2
            best = None; best_sc = -1
            for cx, cy, r in circles:
                patch = gray[max(0,cy-r):cy+r, max(0,cx-r):cx+r]
                bv = float(np.mean(patch)) if patch.size > 0 else 0
                dist = ((cx-roi_cx)**2+(cy-roi_cy)**2)**0.5
                sc = bv/(dist+10)
                if sc > best_sc:
                    best_sc = sc; best = (cx+x1r, cy+y1r)
            if best:
                print(f"  Seed (temporal avg): px={best}")
                return 0, best[0], best[1]
    print(f"  Seed fallback: center")
    return 0, W//2, H//2


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(positions, labels):
    errors = []
    for fidx, gt in sorted(labels.items()):
        pred = positions.get(fidx)
        if pred is None:
            errors.append(None)
        else:
            errors.append(((pred[0]-gt[0])**2+(pred[1]-gt[1])**2)**0.5)
    tracked = [e for e in errors if e is not None]
    if not tracked:
        return dict(tracked=0, total=len(errors), within30=0, within50=0,
                    median=9999, mean=9999, score=0)
    return dict(
        tracked  = len(tracked),
        total    = len(errors),
        within30 = sum(1 for e in tracked if e<=30),
        within50 = sum(1 for e in tracked if e<=50),
        median   = float(np.median(tracked)),
        mean     = float(np.mean(tracked)),
        score    = (len(tracked)/len(errors)) *
                   (sum(1 for e in tracked if e<=50)/len(tracked))
    )

def pev(name, ev):
    print(f"  {name:<50}  {ev['tracked']:>3}/{ev['total']:<3}  "
          f"W30={ev['within30']:>3}  W50={ev['within50']:>3}  "
          f"Med={ev['median']:>6.1f}  Score={ev['score']:.4f}")


# ─────────────────────────────────────────────
# Grid search
# ─────────────────────────────────────────────

def grid_search(frames, seed_frame, seed_pos, labels):
    random.seed(99)
    grid = {
        "proc_noise":        [1.0, 3.0, 5.0],
        "meas_noise_fast":   [8.0, 15.0],
        "meas_noise_slow":   [150.0, 300.0],
        "active_thresh":     [0.05, 0.08],
        "kick_speed":        [15.0, 25.0],
        "max_hold":          [15, 25],
        "max_recover":       [30, 50],
        "radius_base":       [55, 80, 110],
        "mot_k":             [2, 3],
    }
    combos = list(iproduct(
        grid["proc_noise"], grid["meas_noise_fast"], grid["meas_noise_slow"],
        grid["active_thresh"], grid["kick_speed"], grid["max_hold"],
        grid["max_recover"], grid["radius_base"], grid["mot_k"],
    ))
    keys = ["proc_noise","meas_noise_fast","meas_noise_slow","active_thresh",
            "kick_speed","max_hold","max_recover","radius_base","mot_k"]

    # Always include a few hand-tuned combos
    hand_tuned = [
        dict(proc_noise=2.0, meas_noise_fast=10.0, meas_noise_slow=200.0,
             active_thresh=0.06, kick_speed=20.0, max_hold=20, max_recover=40,
             radius_base=65, mot_k=3),
        dict(proc_noise=1.0, meas_noise_fast=8.0,  meas_noise_slow=300.0,
             active_thresh=0.05, kick_speed=15.0, max_hold=25, max_recover=50,
             radius_base=80, mot_k=3),
        dict(proc_noise=3.0, meas_noise_fast=15.0, meas_noise_slow=150.0,
             active_thresh=0.07, kick_speed=25.0, max_hold=15, max_recover=30,
             radius_base=55, mot_k=2),
        dict(proc_noise=2.0, meas_noise_fast=10.0, meas_noise_slow=300.0,
             active_thresh=0.05, kick_speed=20.0, max_hold=25, max_recover=50,
             radius_base=110, mot_k=3),
    ]

    sample_combos = [dict(zip(keys,c)) for c in random.sample(combos, min(36, len(combos)))]
    all_combos = hand_tuned + sample_combos

    print(f"  Running {len(all_combos)} combinations...")
    results = []
    for i, params in enumerate(all_combos):
        pos = track_v3(frames, seed_frame, seed_pos, **params)
        ev  = evaluate(pos, labels)
        results.append((params, pos, ev))
        if (i+1) % 10 == 0:
            print(f"    {i+1}/{len(all_combos)} done — best so far: "
                  f"{max(results, key=lambda x: x[2]['score'])[2]['score']:.4f}")

    results.sort(key=lambda x: x[2]["score"], reverse=True)
    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",       required=True)
    parser.add_argument("--labels",      required=True)
    parser.add_argument("--frames-dir",  default=None)
    parser.add_argument("--render",      action="store_true")
    parser.add_argument("--out",         default="tracking_analysis_v3.txt")
    args = parser.parse_args()

    with open(args.labels) as f:
        data = json.load(f)
    labels = {e["frame"]: (e["px"], e["py"]) for e in data["labels"]}
    print(f"Loaded {len(labels)} labels")

    # Frames
    if args.frames_dir and os.path.isdir(args.frames_dir):
        frames = sorted([os.path.join(args.frames_dir, f)
                         for f in os.listdir(args.frames_dir)
                         if f.startswith("f_") and f.endswith(".jpg")])
        print(f"Using {len(frames)} pre-extracted frames")
    else:
        fdir = "/tmp/soccer_v2_frames"
        existing = [f for f in os.listdir(fdir) if f.startswith("f_")] \
                   if os.path.isdir(fdir) else []
        if existing:
            frames = sorted([os.path.join(fdir, f) for f in existing])
            print(f"Using {len(frames)} cached frames from {fdir}")
        else:
            os.makedirs(fdir, exist_ok=True)
            print("Extracting frames...")
            pat = os.path.join(fdir, "f_%04d.jpg")
            subprocess.run(["ffmpeg","-i",args.video,"-vf",f"fps={FPS}",
                            "-q:v","2","-loglevel","error",pat], check=True)
            frames = sorted([os.path.join(fdir,f) for f in os.listdir(fdir)
                             if f.startswith("f_")])
            print(f"Extracted {len(frames)} frames")

    print("\nFinding seed...")
    seed_idx, seed_px, seed_py = find_seed(frames)
    print(f"Seed: frame {seed_idx} ({seed_idx/FPS:.1f}s) px=({seed_px},{seed_py})")

    # Baseline
    print("\n--- Baseline v3 ---")
    t0 = time.time()
    base_pos = track_v3(frames, seed_idx, (seed_px, seed_py))
    base_ev  = evaluate(base_pos, labels)
    print(f"Time: {time.time()-t0:.1f}s")
    pev("baseline_v3", base_ev)

    # Grid search
    print("\n--- Grid search v3 ---")
    t0 = time.time()
    grid = grid_search(frames, seed_idx, (seed_px, seed_py), labels)
    print(f"Grid search time: {time.time()-t0:.1f}s")

    print(f"\n{'='*80}")
    print("  TOP 10:")
    print(f"{'='*80}")
    for params, pos, ev in grid[:10]:
        name = (f"pn={params['proc_noise']} mnf={params['meas_noise_fast']} "
                f"mns={params['meas_noise_slow']} at={params['active_thresh']} "
                f"mh={params['max_hold']} mr={params['max_recover']} "
                f"rb={params['radius_base']} k={params['mot_k']}")
        pev(name, ev)

    winner_params, winner_pos, winner_ev = grid[0]
    print(f"\n{'='*80}")
    print(f"  WINNER: {winner_params}")
    print(f"  Score={winner_ev['score']:.4f}  "
          f"Tracked={winner_ev['tracked']}/{winner_ev['total']}  "
          f"W30={winner_ev['within30']}  W50={winner_ev['within50']}  "
          f"Median={winner_ev['median']:.1f}px")
    print(f"{'='*80}\n")

    print("Per-frame detail (winner):")
    for fidx, gt in sorted(labels.items()):
        pred = winner_pos.get(fidx)
        if pred is None:
            print(f"  frame {fidx:4d}: NOT TRACKED  gt=({gt[0]},{gt[1]})")
        else:
            err  = ((pred[0]-gt[0])**2+(pred[1]-gt[1])**2)**0.5
            flag = " <-- MISS" if err > 50 else ""
            print(f"  frame {fidx:4d}: err={err:6.1f}px  "
                  f"pred=({pred[0]:4d},{pred[1]:4d})  "
                  f"gt=({gt[0]:4d},{gt[1]:4d}){flag}")

    with open(args.out, "w") as f:
        f.write(f"Winner params: {winner_params}\n")
        f.write(f"Score: {winner_ev['score']:.4f}\n")
        f.write(f"Tracked: {winner_ev['tracked']}/{winner_ev['total']}\n")
        f.write(f"Within30: {winner_ev['within30']}\n")
        f.write(f"Within50: {winner_ev['within50']}\n")
        f.write(f"Median: {winner_ev['median']:.1f}px\n\n")
        f.write("All grid results:\n")
        for p, _, ev in grid:
            f.write(f"  {p}: score={ev['score']:.4f} tracked={ev['tracked']}/{ev['total']} "
                    f"w30={ev['within30']} w50={ev['within50']} median={ev['median']:.1f}\n")
    print(f"\nSaved to {args.out}")

    if args.render:
        out_path = "tracking_v3_winner.mp4"
        print(f"\nRendering {out_path}...")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        ref    = cv2.imread(frames[0])
        writer = cv2.VideoWriter(out_path, fourcc, FPS, (ref.shape[1], ref.shape[0]))
        for i, fp in enumerate(frames):
            img = cv2.imread(fp)
            if img is None: continue
            cv2.putText(img, f"{i/FPS:.1f}s", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            pred = winner_pos.get(i)
            if pred:
                cv2.circle(img, pred, 14, (0,255,0), 3)
                cv2.circle(img, pred,  3, (0,255,0), -1)
            gt = labels.get(i)
            if gt:
                cv2.circle(img, gt, 8, (0,0,255), 2)
            writer.write(img)
        writer.release()
        print(f"Saved {out_path}  (green=pred, red=gt)")


if __name__ == "__main__":
    main()
