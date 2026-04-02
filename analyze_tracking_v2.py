#!/usr/bin/env python3
"""
Ball tracking v2 - completely redesigned architecture.

Problems with v1:
- Seeds once, never recovers when tracker drifts
- No stationary ball handling (ball sits still for 17+ seconds)
- Velocity prediction amplifies errors instead of correcting
- Fixed search radius doesn't handle 0-130px/frame speed range

v2 architecture:
- Kalman filter (x, y, vx, vy) for proper state estimation
- State machine: ACTIVE / STATIONARY / RECOVERING
- Full-frame scan for re-initialization when lost
- Per-frame candidate detection (not just template search)
- Physics-aware scoring (ball decelerates, doesn't teleport)

Usage:
  .venv/bin/python3 analyze_tracking_v2.py --video Test.mp4 --labels ball_labels.json
  .venv/bin/python3 analyze_tracking_v2.py --video Test.mp4 --labels ball_labels.json --frames-dir /tmp/soccer_frames_xxx
"""

import argparse
import json
import os
import subprocess
import time

import cv2
import numpy as np

FPS  = 10.0
W, H = 1920, 1080

# Field region (exclude crowd/sky/ads)
FIELD_Y_MIN = 0.20
FIELD_Y_MAX = 0.82

# Ball size bounds (pixels²)
BALL_AREA_MIN = 5
BALL_AREA_MAX = 700

# Stationary: ball considered held when speed < this px/frame
STATIONARY_SPEED = 4.0

K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# ============================================================
# KALMAN FILTER
# ============================================================

def make_kalman(px, py, vx=0, vy=0, proc_noise=2.0, meas_noise=12.0):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix  = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kf.transitionMatrix   = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    kf.processNoiseCov    = np.eye(4, dtype=np.float32) * proc_noise
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * meas_noise
    kf.errorCovPost       = np.eye(4, dtype=np.float32) * 100
    kf.statePost          = np.array([[px],[py],[vx],[vy]], np.float32)
    return kf


def kf_predict(kf):
    pred = kf.predict()
    return (int(np.clip(pred[0,0], 0, W-1)),
            int(np.clip(pred[1,0], 0, H-1)))


def kf_update(kf, px, py):
    kf.correct(np.array([[float(px)],[float(py)]], np.float32))


def kf_velocity(kf):
    return float(kf.statePost[2,0]), float(kf.statePost[3,0])


def kf_speed(kf):
    vx, vy = kf_velocity(kf)
    return (vx**2 + vy**2)**0.5


# ============================================================
# DETECTION
# ============================================================

def field_mask(frame_h):
    """Row range for the field area."""
    return int(FIELD_Y_MIN * frame_h), int(FIELD_Y_MAX * frame_h)


def non_green(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([30,25,25]), np.array([90,255,255]))
    return cv2.bitwise_not(green)


def motion_diff(gray_curr, gray_prev, thresh=12):
    diff = cv2.absdiff(gray_curr, gray_prev)
    _, m = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, K3)
    m = cv2.dilate(m, K5, iterations=1)
    return m


def score_blob(cx, cy, area, circ, gray_roi, px_offset_x, px_offset_y):
    """Score a blob as a ball candidate [0-1]."""
    # Size score: peak at ~100px², tail off
    size_sc = np.exp(-((area - 80)**2) / (2 * 150**2)) + 0.2
    # Circularity
    circ_sc = circ + 0.15
    # Brightness at centroid
    bx1 = max(0, cx-5); bx2 = min(gray_roi.shape[1], cx+5)
    by1 = max(0, cy-5); by2 = min(gray_roi.shape[0], cy+5)
    patch = gray_roi[by1:by2, bx1:bx2]
    bright_sc = float(np.mean(patch)) / 220.0 if patch.size > 0 else 0
    return float(size_sc * circ_sc * bright_sc)


def detect_candidates_in_window(curr, prev, pred_px, pred_py, radius,
                                 mot_thresh=12, bright_thresh=80):
    """
    Find ball candidates within a search window.
    Returns sorted list of (full_px, full_py, score).
    """
    x1 = max(0, pred_px - radius); x2 = min(W, pred_px + radius)
    y1 = max(0, pred_py - radius); y2 = min(H, pred_py + radius)

    roi_c = curr[y1:y2, x1:x2]
    if roi_c.size == 0:
        return []

    gc = cv2.cvtColor(roi_c, cv2.COLOR_BGR2GRAY)

    # Non-green mask
    ng = non_green(roi_c)
    # Brightness
    br = (gc > bright_thresh).astype(np.uint8) * 255

    if prev is not None:
        roi_p = prev[y1:y2, x1:x2]
        gp = cv2.cvtColor(roi_p, cv2.COLOR_BGR2GRAY)
        mot = motion_diff(gc, gp, thresh=mot_thresh)
        cand = cv2.bitwise_and(cv2.bitwise_and(mot, ng), br)
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
        fx = x1 + cx; fy = y1 + cy
        dist = ((fx-pred_px)**2+(fy-pred_py)**2)**0.5
        if dist > radius:
            continue
        p = cv2.arcLength(c, True)
        circ = 4*np.pi*area/(p**2) if p > 0 else 0
        sc = score_blob(cx, cy, area, circ, gc, x1, y1)
        sc *= (1.0 - dist/(radius+1))  # proximity bonus
        results.append((fx, fy, sc))

    # Fallback: brightness-only (catches stationary ball)
    if not results or max(r[2] for r in results) < 0.06:
        ng_bright = cv2.bitwise_and(gc, gc, mask=ng)
        blurred = cv2.GaussianBlur(ng_bright.astype(np.float32), (7,7), 2)
        _, max_val, _, max_loc = cv2.minMaxLoc(blurred)
        if max_val > 100:
            fx = x1 + max_loc[0]; fy = y1 + max_loc[1]
            dist = ((fx-pred_px)**2+(fy-pred_py)**2)**0.5
            if dist <= radius:
                sc = (max_val/255.0) * (1-dist/(radius+1)) * 0.35
                results.append((fx, fy, sc))

    return sorted(results, key=lambda r: r[2], reverse=True)


def detect_candidates_full_frame(curr, prev, mot_thresh=12, bright_thresh=75):
    """
    Full-frame detection — used for re-initialization.
    Returns sorted list of (px, py, score).
    """
    fy_min, fy_max = field_mask(H)

    gc = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    ng = non_green(curr)
    br = (gc > bright_thresh).astype(np.uint8) * 255

    if prev is not None:
        gp = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        mot = motion_diff(gc, gp, thresh=mot_thresh)
        cand = cv2.bitwise_and(cv2.bitwise_and(mot, ng), br)
    else:
        cand = cv2.bitwise_and(ng, br)

    # Restrict to field
    mask_field = np.zeros_like(cand)
    mask_field[fy_min:fy_max, :] = 255
    cand = cv2.bitwise_and(cand, mask_field)
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
        if not (fy_min <= cy <= fy_max):
            continue
        p = cv2.arcLength(c, True)
        circ = 4*np.pi*area/(p**2) if p > 0 else 0
        sc = score_blob(cx, cy, area, circ, gc, 0, 0)
        results.append((cx, cy, sc))

    return sorted(results, key=lambda r: r[2], reverse=True)[:30]


# ============================================================
# STATE MACHINE TRACKER
# ============================================================

ACTIVE     = "ACTIVE"
STATIONARY = "STATIONARY"
RECOVERING = "RECOVERING"


def track_v2(frames, seed_frame, seed_pos,
             proc_noise=2.0, meas_noise=12.0,
             active_thresh=0.07, stat_speed=STATIONARY_SPEED,
             max_lost_active=8, max_lost_recovering=25,
             active_radius_base=55, verbose=False):
    """
    State-machine Kalman tracker with full-frame recovery.
    """
    n = len(frames)
    positions = {}
    positions[seed_frame] = seed_pos

    def run_direction(start_idx, start_pos, step):
        kf   = make_kalman(*start_pos, proc_noise=proc_noise, meas_noise=meas_noise)
        state     = ACTIVE
        lost      = 0
        last_good = start_pos
        consec_slow = 0

        idx = start_idx + step

        while 0 <= idx < n:
            curr = cv2.imread(frames[idx])
            prev_idx = idx - step
            prev = cv2.imread(frames[prev_idx]) if 0 <= prev_idx < n else None
            if curr is None:
                break

            pred_px, pred_py = kf_predict(kf)
            speed = kf_speed(kf)

            # --- STATIONARY detection ---
            if speed < stat_speed:
                consec_slow += 1
            else:
                consec_slow = 0

            if consec_slow >= 4 and state == ACTIVE:
                state = STATIONARY
                if verbose:
                    print(f"  → STATIONARY at frame {idx}")

            # ===================================
            if state == STATIONARY:
                # Ball is not moving — search very tight window using brightness only
                cands = detect_candidates_in_window(
                    curr, prev, pred_px, pred_py, radius=30,
                    mot_thresh=8, bright_thresh=85)

                if cands and cands[0][2] > 0.04:
                    fx, fy, sc = cands[0]
                    positions[idx] = (fx, fy)
                    kf_update(kf, fx, fy)
                    last_good = (fx, fy)
                    lost = 0
                    # Check if ball started moving
                    spd = kf_speed(kf)
                    if spd > stat_speed * 2:
                        state = ACTIVE
                        consec_slow = 0
                        if verbose:
                            print(f"  → ACTIVE (ball moved) at frame {idx}")
                else:
                    # Ball not detected at current position — hold Kalman state
                    positions[idx] = (pred_px, pred_py)
                    lost += 1
                    if lost > max_lost_recovering:
                        state = RECOVERING
                        if verbose:
                            print(f"  → RECOVERING from STATIONARY at frame {idx}")

            # ===================================
            elif state == ACTIVE:
                # Adaptive radius: wider when moving fast
                radius = max(active_radius_base,
                             min(220, int(speed * 2.5 + active_radius_base)))

                cands = detect_candidates_in_window(
                    curr, prev, pred_px, pred_py, radius=radius)

                if cands and cands[0][2] >= active_thresh:
                    fx, fy, sc = cands[0]
                    positions[idx] = (fx, fy)
                    kf_update(kf, fx, fy)
                    last_good = (fx, fy)
                    lost = 0
                else:
                    lost += 1
                    if lost <= max_lost_active:
                        positions[idx] = (pred_px, pred_py)
                    else:
                        state = RECOVERING
                        lost  = 0
                        if verbose:
                            print(f"  → RECOVERING at frame {idx}")

            # ===================================
            elif state == RECOVERING:
                # Full-frame scan
                vx, vy = kf_velocity(kf)
                exp_x = int(np.clip(pred_px, 0, W-1))
                exp_y = int(np.clip(pred_py, 0, H-1))

                all_cands = detect_candidates_full_frame(curr, prev)

                best_pos, best_sc = None, 0.0
                for fx, fy, sc in all_cands:
                    dist = ((fx-exp_x)**2+(fy-exp_y)**2)**0.5
                    # Prefer candidates near expected position
                    proximity_bonus = np.exp(-dist/300)
                    total_sc = sc * (proximity_bonus + 0.1)
                    if total_sc > best_sc:
                        best_sc = total_sc
                        best_pos = (fx, fy)

                if best_pos and best_sc > 0.04:
                    positions[idx] = best_pos
                    # Re-initialize Kalman at found position
                    kf = make_kalman(*best_pos, proc_noise=proc_noise,
                                     meas_noise=meas_noise)
                    last_good = best_pos
                    lost = 0
                    consec_slow = 0
                    state = ACTIVE
                    if verbose:
                        print(f"  → ACTIVE (recovered) at frame {idx} "
                              f"pos={best_pos} sc={best_sc:.3f}")
                else:
                    lost += 1
                    if lost <= max_lost_recovering:
                        positions[idx] = (pred_px, pred_py)
                    # else: leave this frame untracked (None)

            if len(positions) > 0 and idx in positions:
                pass  # already set
            idx += step

        return positions

    # Forward pass
    fwd = run_direction(seed_frame, seed_pos, +1)
    positions.update(fwd)

    # Backward pass
    bwd = run_direction(seed_frame, seed_pos, -1)
    for k, v in bwd.items():
        if k not in positions:
            positions[k] = v

    return positions


# ============================================================
# SEED DETECTION
# ============================================================

def find_seed(frames, avg_n=25):
    """
    Temporal average + Hough → motion fallback.
    """
    x1r, x2r = int(0.35*W), int(0.65*W)
    y1r, y2r = int(0.33*H), int(0.67*H)

    # Strategy 1: Temporal average
    stack = []
    for i in range(min(avg_n, len(frames))):
        f = cv2.imread(frames[i])
        if f is not None:
            stack.append(f[y1r:y2r, x1r:x2r].astype(np.float32))
    if stack:
        avg  = np.mean(stack, axis=0).astype(np.uint8)
        gray = cv2.cvtColor(avg, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5), 1)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,
                                   dp=1, minDist=10, param1=30, param2=8,
                                   minRadius=3, maxRadius=13)
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
                print(f"  Seed (temporal avg): px={best} sc={best_sc:.2f}")
                return 0, best[0], best[1]

    # Strategy 2: Motion-based
    x1r, x2r = int(0.22*W), int(0.78*W)
    y1r, y2r = int(0.20*H), int(0.80*H)
    for i in range(min(80, len(frames)-1)):
        fa = cv2.imread(frames[i]); fb = cv2.imread(frames[i+1])
        if fa is None or fb is None:
            continue
        ga = cv2.cvtColor(fa[y1r:y2r, x1r:x2r], cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(fb[y1r:y2r, x1r:x2r], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(ga, gb)
        _, mot = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        mot = cv2.morphologyEx(mot, cv2.MORPH_OPEN, K3)
        cnts, _ = cv2.findContours(mot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = []
        for c in cnts:
            a = cv2.contourArea(c)
            if 8 < a < 500:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"]/M["m00"])+x1r
                    cy = int(M["m01"]/M["m00"])+y1r
                    if int(0.30*W) < cx < int(0.70*W):
                        center.append((cx, cy, a))
        if center:
            b = min(center, key=lambda x: x[2])
            print(f"  Seed (motion): frame {i+1} px=({b[0]},{b[1]})")
            return i+1, b[0], b[1]

    print(f"  Seed fallback: center of frame")
    return 0, W//2, H//2


# ============================================================
# EVALUATION
# ============================================================

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
        score    = (len(tracked)/len(errors)) * (sum(1 for e in tracked if e<=50)/len(tracked))
    )


def print_eval(name, ev):
    print(f"  {name:<40} {ev['tracked']:>3}/{ev['total']:<3}  "
          f"W30={ev['within30']:>3}  W50={ev['within50']:>3}  "
          f"Med={ev['median']:>6.1f}  Score={ev['score']:.4f}")


# ============================================================
# GRID SEARCH
# ============================================================

def grid_search(frames, seed_frame, seed_pos, labels):
    """Try all parameter combinations and rank by score."""
    param_grid = {
        "proc_noise":       [1.0, 3.0, 5.0],
        "meas_noise":       [8.0, 15.0, 25.0],
        "active_thresh":    [0.05, 0.08, 0.12],
        "stat_speed":       [3.0, 5.0, 8.0],
        "max_lost_active":  [5, 10, 15],
        "active_radius_base": [50, 70, 100],
    }

    # Generate combinations (sample to keep runtime reasonable)
    from itertools import product as iproduct
    import random
    random.seed(42)

    combos = []
    for pn, mn, at, ss, mla, arb in iproduct(
        param_grid["proc_noise"],
        param_grid["meas_noise"],
        param_grid["active_thresh"],
        param_grid["stat_speed"],
        param_grid["max_lost_active"],
        param_grid["active_radius_base"],
    ):
        combos.append(dict(proc_noise=pn, meas_noise=mn, active_thresh=at,
                           stat_speed=ss, max_lost_active=mla,
                           active_radius_base=arb))

    # Sample 40 random + always include some key ones
    key_combos = [
        dict(proc_noise=2.0, meas_noise=12.0, active_thresh=0.07, stat_speed=4.0,
             max_lost_active=8, active_radius_base=55),
        dict(proc_noise=1.0, meas_noise=8.0,  active_thresh=0.05, stat_speed=3.0,
             max_lost_active=10, active_radius_base=70),
        dict(proc_noise=3.0, meas_noise=15.0, active_thresh=0.08, stat_speed=5.0,
             max_lost_active=8, active_radius_base=70),
        dict(proc_noise=1.0, meas_noise=15.0, active_thresh=0.05, stat_speed=5.0,
             max_lost_active=15, active_radius_base=100),
        dict(proc_noise=5.0, meas_noise=25.0, active_thresh=0.05, stat_speed=8.0,
             max_lost_active=5,  active_radius_base=50),
    ]

    sample = random.sample(combos, min(35, len(combos)))
    all_combos = key_combos + sample

    print(f"  Running {len(all_combos)} parameter combinations...")
    results = []
    for i, params in enumerate(all_combos):
        pos = track_v2(frames, seed_frame, seed_pos, **params)
        ev  = evaluate(pos, labels)
        results.append((params, pos, ev))
        if (i+1) % 10 == 0:
            print(f"    {i+1}/{len(all_combos)} done...")

    results.sort(key=lambda x: x[2]["score"], reverse=True)
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",       required=True)
    parser.add_argument("--labels",      required=True)
    parser.add_argument("--frames-dir",  default=None)
    parser.add_argument("--out",         default="tracking_analysis_v2.txt")
    parser.add_argument("--render",      action="store_true",
                        help="Render winner video")
    args = parser.parse_args()

    # Load labels
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
        tmpdir = "/tmp/soccer_v2_frames"
        os.makedirs(tmpdir, exist_ok=True)
        existing = [f for f in os.listdir(tmpdir) if f.startswith("f_") and f.endswith(".jpg")]
        if existing:
            frames = sorted([os.path.join(tmpdir, f) for f in existing])
            print(f"Using {len(frames)} cached frames from {tmpdir}")
        else:
            print("Extracting frames...")
            pattern = os.path.join(tmpdir, "f_%04d.jpg")
            subprocess.run(["ffmpeg", "-i", args.video, "-vf", f"fps={FPS}",
                            "-q:v", "2", "-loglevel", "error", pattern], check=True)
            frames = sorted([os.path.join(tmpdir, f)
                             for f in os.listdir(tmpdir)
                             if f.startswith("f_") and f.endswith(".jpg")])
            print(f"Extracted {len(frames)} frames to {tmpdir}")
        with open("frames_dir_v2.txt", "w") as f:
            f.write(tmpdir)

    print("\nFinding seed...")
    seed_idx, seed_px, seed_py = find_seed(frames)
    print(f"Seed: frame {seed_idx} ({seed_idx/FPS:.1f}s) px=({seed_px},{seed_py})")

    # Quick baseline run
    print("\n--- Baseline run ---")
    t0 = time.time()
    pos_baseline = track_v2(frames, seed_idx, (seed_px, seed_py), verbose=False)
    ev_base = evaluate(pos_baseline, labels)
    print(f"Time: {time.time()-t0:.1f}s")
    print_eval("baseline", ev_base)

    # Grid search
    print("\n--- Grid search ---")
    t0 = time.time()
    grid = grid_search(frames, seed_idx, (seed_px, seed_py), labels)
    print(f"Grid search time: {time.time()-t0:.1f}s")

    print(f"\n{'='*72}")
    print(f"  TOP 10 RESULTS")
    print(f"{'='*72}")
    for params, pos, ev in grid[:10]:
        name = (f"pn={params['proc_noise']} mn={params['meas_noise']} "
                f"at={params['active_thresh']} ss={params['stat_speed']} "
                f"mla={params['max_lost_active']} arb={params['active_radius_base']}")
        print_eval(name, ev)

    winner_params, winner_pos, winner_ev = grid[0]
    print(f"\n{'='*72}")
    print(f"  WINNER params: {winner_params}")
    print(f"  Score={winner_ev['score']:.4f}  "
          f"Tracked={winner_ev['tracked']}/{winner_ev['total']}  "
          f"Within30={winner_ev['within30']}  "
          f"Within50={winner_ev['within50']}  "
          f"Median={winner_ev['median']:.1f}px")
    print(f"{'='*72}\n")

    # Per-frame detail
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

    # Save
    with open(args.out, "w") as f:
        f.write(f"Winner params: {winner_params}\n")
        f.write(f"Score: {winner_ev['score']:.4f}\n")
        f.write(f"Tracked: {winner_ev['tracked']}/{winner_ev['total']}\n")
        f.write(f"Within30: {winner_ev['within30']}\n")
        f.write(f"Within50: {winner_ev['within50']}\n")
        f.write(f"Median: {winner_ev['median']:.1f}px\n\n")
        f.write("All results:\n")
        for p, _, ev in grid:
            f.write(f"  {p}: score={ev['score']:.4f} tracked={ev['tracked']}/{ev['total']} "
                    f"w30={ev['within30']} w50={ev['within50']} median={ev['median']:.1f}\n")
    print(f"\nResults saved to {args.out}")

    if args.render:
        out_path = "tracking_v2_winner.mp4"
        print(f"\nRendering {out_path}...")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        ref = cv2.imread(frames[0])
        writer = cv2.VideoWriter(out_path, fourcc, FPS, (ref.shape[1], ref.shape[0]))
        for i, fp in enumerate(frames):
            img = cv2.imread(fp)
            if img is None:
                continue
            t = i / FPS
            cv2.putText(img, f"{t:.1f}s", (20,40),
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
        print(f"Saved: {out_path}  (green=pred, red=gt)")


if __name__ == "__main__":
    main()
