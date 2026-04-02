#!/usr/bin/env python3
"""
Ball tracking v4 - dual-mode tracker.

Root cause from v1-v3: motion detection fails for slow ball (< 3px/frame).
Template matching fails for fast ball (50-130px/frame search window too large).
Need BOTH, switched by velocity.

Architecture:
  TMATCH  - template matching, ±40px window. Works for slow/stationary ball.
  ACTIVE  - accumulated motion detection, adaptive radius. Works for fast ball.
  HOLD    - lost, holding last confirmed, running full-frame scan each frame.

State transitions:
  TMATCH → ACTIVE  : motion blob > KICK_DISP px from current position detected
  ACTIVE → TMATCH  : ball slows to < SLOW_SPEED for 5+ frames
  ACTIVE → HOLD    : no detection for max_lost_active frames
  HOLD   → TMATCH  : ball found (low speed) or ACTIVE (high speed)
  TMATCH → HOLD    : template match poor for max_lost_tmatch frames

Key params calibrated from 253-label ground truth:
  - Ball moves < 3px/frame in slow phase (use template window 40px)
  - Ball moves 20-130px/frame in active phase (use motion, radius 60-220px)
  - Kicks detectable by 3-frame accumulated motion > 30px from last position

Usage:
  .venv/bin/python3 analyze_tracking_v4.py --video Test.mp4 --labels ball_labels.json
  .venv/bin/python3 analyze_tracking_v4.py --video Test.mp4 --labels ball_labels.json --frames-dir /tmp/soccer_v2_frames --render
"""

import argparse
import json
import os
import subprocess
import time
import random
from itertools import product as iproduct

import cv2
import numpy as np

FPS = 10.0
W, H = 1920, 1080
FIELD_Y_MIN = int(0.18 * H)
FIELD_Y_MAX = int(0.84 * H)
BALL_AREA_MIN = 4
BALL_AREA_MAX = 900
K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))


# ──────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────

def dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def load_gray(path):
    f = cv2.imread(path)
    return f, cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f is not None else (None, None)


# ──────────────────────────────────────────────
# Template matching detector
# ──────────────────────────────────────────────

def tmatch_detect(curr_bgr, pred_px, pred_py, tmpl, radius=40):
    """
    Find ball via template matching within radius of prediction.
    Returns (px, py, score) or (None, None, 0).
    """
    x1 = max(0, pred_px-radius); x2 = min(W, pred_px+radius)
    y1 = max(0, pred_py-radius); y2 = min(H, pred_py+radius)
    roi = curr_bgr[y1:y2, x1:x2]
    th, tw = tmpl.shape[:2]
    if roi.shape[0] < th or roi.shape[1] < tw:
        return None, None, 0.0
    res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    fx = x1 + max_loc[0] + tw//2
    fy = y1 + max_loc[1] + th//2
    return fx, fy, float(max_val)


def get_template(bgr, px, py, half=11):
    x1,y1 = max(0,px-half), max(0,py-half)
    x2,y2 = min(W,px+half), min(H,py+half)
    patch = bgr[y1:y2, x1:x2]
    return patch.copy() if patch.size > 0 else None


# ──────────────────────────────────────────────
# Motion-based detector
# ──────────────────────────────────────────────

def accum_motion(frames, idx, step, k=3):
    """Frame idx vs frame (idx - k*step). Returns (motion_mask, curr_bgr, curr_gray) or None."""
    ref_idx = idx - k*step
    if not (0 <= ref_idx < len(frames)):
        return None
    curr = cv2.imread(frames[idx])
    ref  = cv2.imread(frames[ref_idx])
    if curr is None or ref is None:
        return None
    gc = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(ref,  cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gc, gr)
    _, mot = cv2.threshold(diff, 11, 255, cv2.THRESH_BINARY)
    mot = cv2.morphologyEx(mot, cv2.MORPH_OPEN, K3)
    mot = cv2.dilate(mot, K5, iterations=1)
    return mot, curr, gc


def motion_detect(mot_mask, curr_gray, curr_bgr, pred_px, pred_py, radius,
                  bright_thresh=70):
    """
    Find ball candidates in motion mask within radius of prediction.
    Returns sorted (px, py, score) list.
    """
    x1 = max(0,pred_px-radius); x2 = min(W,pred_px+radius)
    y1 = max(0,pred_py-radius); y2 = min(H,pred_py+radius)
    roi_mot  = mot_mask[y1:y2, x1:x2]
    roi_gray = curr_gray[y1:y2, x1:x2]
    roi_bgr  = curr_bgr[y1:y2, x1:x2]
    if roi_gray.size == 0:
        return []

    hsv   = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([30,25,25]), np.array([90,255,255]))
    ng    = cv2.bitwise_not(green)
    br    = (roi_gray > bright_thresh).astype(np.uint8) * 255
    cand  = cv2.bitwise_and(cv2.bitwise_and(roi_mot, ng), br)
    cand  = cv2.morphologyEx(cand, cv2.MORPH_OPEN, K3)

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
        d = dist((fx,fy),(pred_px,pred_py))
        if d > radius:
            continue
        p = cv2.arcLength(c, True)
        circ = 4*np.pi*area/(p**2) if p > 0 else 0
        patch = roi_gray[max(0,cy-5):cy+5, max(0,cx-5):cx+5]
        bv = float(np.mean(patch))/220.0 if patch.size > 0 else 0
        size_sc = np.exp(-((area-80)**2)/(2*200**2)) + 0.2
        sc = (circ+0.15)*bv*size_sc*(1.0-d/(radius+1))
        results.append((fx, fy, sc))
    return sorted(results, key=lambda r: r[2], reverse=True)


def full_frame_motion(frames, idx, step, k=3, bright_thresh=65):
    """Unbiased full-frame detection for recovery."""
    res = accum_motion(frames, idx, step, k=k)
    if res is None:
        return []
    mot, curr_bgr, curr_gray = res

    hsv   = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([30,25,25]), np.array([90,255,255]))
    ng    = cv2.bitwise_not(green)
    br    = (curr_gray > bright_thresh).astype(np.uint8)*255
    cand  = cv2.bitwise_and(cv2.bitwise_and(mot, ng), br)
    field = np.zeros_like(cand); field[FIELD_Y_MIN:FIELD_Y_MAX,:] = 255
    cand  = cv2.bitwise_and(cand, field)
    cand  = cv2.morphologyEx(cand, cv2.MORPH_OPEN, K3)

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
        size_sc = np.exp(-((area-80)**2)/(2*200**2)) + 0.2
        sc = (circ+0.15)*bv*size_sc
        results.append((cx, cy, sc))
    return sorted(results, key=lambda r: r[2], reverse=True)[:15]


# ──────────────────────────────────────────────
# Kick detector (for TMATCH → ACTIVE transition)
# ──────────────────────────────────────────────

def detect_kick(frames, idx, step, last_px, last_py, k=3,
                search_r=180, min_disp=35):
    """
    Detect if ball has been kicked (large fast motion from last_pos).
    Uses k-frame accumulated motion to amplify displacement.
    Returns (px, py) of new ball position, or None.
    """
    res = accum_motion(frames, idx, step, k=k)
    if res is None:
        return None
    mot, curr_bgr, curr_gray = res

    x1 = max(0,last_px-search_r); x2 = min(W,last_px+search_r)
    y1 = max(0,last_py-search_r); y2 = min(H,last_py+search_r)
    roi_mot  = mot[y1:y2, x1:x2]
    roi_gray = curr_gray[y1:y2, x1:x2]
    roi_bgr  = curr_bgr[y1:y2, x1:x2]

    hsv   = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([30,25,25]), np.array([90,255,255]))
    ng    = cv2.bitwise_not(green)
    br    = (roi_gray > 65).astype(np.uint8)*255
    cand  = cv2.bitwise_and(cv2.bitwise_and(roi_mot, ng), br)
    cand  = cv2.morphologyEx(cand, cv2.MORPH_OPEN, K3)

    cnts, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        if not (FIELD_Y_MIN <= fy <= FIELD_Y_MAX):
            continue
        disp = dist((fx,fy),(last_px,last_py))
        if disp < min_disp:
            continue  # not a kick, too close
        p = cv2.arcLength(c, True)
        circ = 4*np.pi*area/(p**2) if p > 0 else 0
        patch = roi_gray[max(0,cy-5):cy+5, max(0,cx-5):cx+5]
        bv = float(np.mean(patch))/220.0 if patch.size > 0 else 0
        sc = (circ+0.15)*bv*min(disp/80.0,1.0)
        if sc > best_sc:
            best_sc = sc; best = (fx, fy)
    return best if best_sc > 0.04 else None


# ──────────────────────────────────────────────
# Dual-mode tracker
# ──────────────────────────────────────────────

TMATCH = "TMATCH"
ACTIVE = "ACTIVE"
HOLD   = "HOLD"


def track_v4(frames, seed_frame, seed_pos,
             # Template matching params
             tmatch_radius=40,
             tmatch_thresh=0.28,
             tmatch_update_thresh=0.40,
             tmatch_half=11,
             max_lost_tmatch=20,
             # Active tracking params
             mot_k=3,
             active_radius_base=65,
             active_thresh=0.06,
             max_lost_active=12,
             # Kick detection
             kick_disp=35,
             kick_search_r=180,
             # Slow speed threshold to re-enter TMATCH
             slow_speed=8.0,
             # Hold / recovery
             max_hold=35):

    n = len(frames)
    positions = {seed_frame: seed_pos}

    def run_dir(start_idx, start_pos, step):
        # Initial template from seed frame
        seed_bgr = cv2.imread(frames[start_idx])
        tmpl = get_template(seed_bgr, start_pos[0], start_pos[1], half=tmatch_half)
        if tmpl is None:
            return

        state      = TMATCH
        last_pos   = start_pos
        velocity   = (0.0, 0.0)
        lost       = 0
        hold       = 0
        speed_hist = [0.0] * 5   # rolling speed history

        idx = start_idx + step
        while 0 <= idx < n:
            curr_bgr = cv2.imread(frames[idx])
            if curr_bgr is None:
                idx += step; continue

            # Velocity prediction
            pred_px = int(clamp(last_pos[0] + velocity[0], 0, W-1))
            pred_py = int(clamp(last_pos[1] + velocity[1], 0, H-1))

            # ── TMATCH ───────────────────────────────────────
            if state == TMATCH:
                fx, fy, sc = tmatch_detect(curr_bgr, pred_px, pred_py,
                                           tmpl, radius=tmatch_radius)

                if sc >= tmatch_thresh and fx is not None:
                    new_pos = (fx, fy)
                    positions[idx] = new_pos
                    vx = (fx - last_pos[0]) * 0.6 + velocity[0] * 0.4
                    vy = (fy - last_pos[1]) * 0.6 + velocity[1] * 0.4
                    velocity = (vx, vy)
                    cur_speed = (vx**2+vy**2)**0.5
                    speed_hist.append(cur_speed); speed_hist.pop(0)
                    lost = 0

                    # Update template if stable
                    if sc >= tmatch_update_thresh:
                        new_t = get_template(curr_bgr, fx, fy, half=tmatch_half)
                        if new_t is not None and new_t.shape == tmpl.shape:
                            tmpl = new_t
                    last_pos = new_pos
                else:
                    lost += 1
                    positions[idx] = last_pos  # hold
                    if lost > max_lost_tmatch:
                        state = HOLD; hold = 0; lost = 0

                # Check for kick (transition to ACTIVE)
                kick = detect_kick(frames, idx, step,
                                   last_pos[0], last_pos[1],
                                   k=mot_k, search_r=kick_search_r,
                                   min_disp=kick_disp)
                if kick is not None:
                    # Ball was kicked — switch to active
                    new_t = get_template(curr_bgr, kick[0], kick[1], half=tmatch_half)
                    if new_t is not None:
                        tmpl = new_t
                    positions[idx] = kick
                    velocity = (kick[0]-last_pos[0], kick[1]-last_pos[1])
                    last_pos = kick
                    state = ACTIVE; lost = 0

            # ── ACTIVE ───────────────────────────────────────
            elif state == ACTIVE:
                res = accum_motion(frames, idx, step, k=mot_k)
                if res is not None:
                    mot, _, curr_gray = res
                    spd = (velocity[0]**2+velocity[1]**2)**0.5
                    radius = max(active_radius_base, min(230, int(spd*2.5+active_radius_base)))
                    cands = motion_detect(mot, curr_gray, curr_bgr,
                                         pred_px, pred_py, radius)
                    if cands and cands[0][2] >= active_thresh:
                        fx, fy, sc = cands[0]
                        new_pos = (fx, fy)
                        positions[idx] = new_pos
                        vx = (fx-last_pos[0])*0.7 + velocity[0]*0.3
                        vy = (fy-last_pos[1])*0.7 + velocity[1]*0.3
                        velocity = (vx, vy)
                        cur_speed = (vx**2+vy**2)**0.5
                        speed_hist.append(cur_speed); speed_hist.pop(0)
                        lost = 0

                        # Update template at new position
                        new_t = get_template(curr_bgr, fx, fy, half=tmatch_half)
                        if new_t is not None and new_t.shape == tmpl.shape:
                            tmpl = new_t
                        last_pos = new_pos

                        # If ball is slow, switch back to TMATCH
                        avg_spd = sum(speed_hist)/len(speed_hist)
                        if avg_spd < slow_speed:
                            state = TMATCH; lost = 0
                    else:
                        lost += 1
                        if lost <= 4:
                            # Brief loss: trust velocity for a few frames
                            ipos = (clamp(pred_px,0,W-1), clamp(pred_py,0,H-1))
                            positions[idx] = ipos
                            last_pos = ipos
                        elif lost <= max_lost_active:
                            # Hold last confirmed
                            positions[idx] = last_pos
                            velocity = (velocity[0]*0.8, velocity[1]*0.8)  # decelerate
                        else:
                            state = HOLD; hold = 0; lost = 0
                else:
                    lost += 1
                    positions[idx] = last_pos
                    if lost > max_lost_active:
                        state = HOLD; hold = 0; lost = 0

            # ── HOLD ─────────────────────────────────────────
            elif state == HOLD:
                positions[idx] = last_pos  # default
                hold += 1
                velocity = (velocity[0]*0.5, velocity[1]*0.5)  # decay

                # Full-frame scan (unbiased)
                all_cands = full_frame_motion(frames, idx, step,
                                              k=min(mot_k+2, 6))
                if all_cands:
                    # Closest to last_pos
                    best = min(all_cands,
                               key=lambda c: dist((c[0],c[1]),last_pos))
                    d = dist((best[0],best[1]), last_pos)
                    accept_r = clamp(200 + hold*15, 200, 600)
                    if d < accept_r and best[2] > 0.05:
                        new_pos = (best[0], best[1])
                        positions[idx] = new_pos
                        spd = dist(new_pos, last_pos)
                        velocity = (new_pos[0]-last_pos[0], new_pos[1]-last_pos[1])
                        last_pos = new_pos

                        # Update template
                        new_t = get_template(curr_bgr, new_pos[0], new_pos[1],
                                             half=tmatch_half)
                        if new_t is not None:
                            tmpl = new_t

                        if spd < slow_speed:
                            state = TMATCH
                        else:
                            state = ACTIVE
                        hold = 0

                if hold > max_hold:
                    # Give up tracking
                    break

            idx += step

    run_dir(seed_frame, seed_pos, +1)
    run_dir(seed_frame, seed_pos, -1)
    return positions


# ──────────────────────────────────────────────
# Seed
# ──────────────────────────────────────────────

def find_seed(frames, avg_n=25):
    x1r,x2r = int(0.35*W),int(0.65*W)
    y1r,y2r = int(0.33*H),int(0.67*H)
    stack = []
    for i in range(min(avg_n, len(frames))):
        f = cv2.imread(frames[i])
        if f is not None:
            stack.append(f[y1r:y2r, x1r:x2r].astype(np.float32))
    if stack:
        avg  = np.mean(stack, axis=0).astype(np.uint8)
        gray = cv2.cvtColor(avg, cv2.COLOR_BGR2GRAY)
        blr  = cv2.GaussianBlur(gray,(5,5),1)
        circles = cv2.HoughCircles(blr, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                                   param1=30, param2=8, minRadius=3, maxRadius=13)
        if circles is not None:
            circles = np.round(circles[0]).astype(int)
            roi_cx=(x2r-x1r)/2; roi_cy=(y2r-y1r)/2
            best=None; best_sc=-1
            for cx,cy,r in circles:
                patch=gray[max(0,cy-r):cy+r, max(0,cx-r):cx+r]
                bv=float(np.mean(patch)) if patch.size>0 else 0
                d=((cx-roi_cx)**2+(cy-roi_cy)**2)**0.5
                sc=bv/(d+10)
                if sc>best_sc: best_sc=sc; best=(cx+x1r, cy+y1r)
            if best:
                print(f"  Seed (temporal avg): {best}")
                return 0, best[0], best[1]
    return 0, W//2, H//2


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

def evaluate(positions, labels):
    errors = []
    for fidx, gt in sorted(labels.items()):
        pred = positions.get(fidx)
        if pred is None:
            errors.append(None)
        else:
            errors.append(dist(pred, gt))
    tracked = [e for e in errors if e is not None]
    if not tracked:
        return dict(tracked=0,total=len(errors),within30=0,within50=0,
                    median=9999,mean=9999,score=0)
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
    print(f"  {name:<52}  {ev['tracked']:>3}/{ev['total']:<3}  "
          f"W30={ev['within30']:>3}  W50={ev['within50']:>3}  "
          f"Med={ev['median']:>6.1f}  Score={ev['score']:.4f}")


# ──────────────────────────────────────────────
# Grid search
# ──────────────────────────────────────────────

def grid_search(frames, seed_frame, seed_pos, labels):
    random.seed(42)
    hand_tuned = [
        # hand-tuned baseline
        dict(tmatch_radius=40, tmatch_thresh=0.28, tmatch_update_thresh=0.40,
             tmatch_half=11, max_lost_tmatch=20, mot_k=3,
             active_radius_base=65, active_thresh=0.06, max_lost_active=12,
             kick_disp=35, kick_search_r=180, slow_speed=8.0, max_hold=35),
        # looser template threshold
        dict(tmatch_radius=45, tmatch_thresh=0.22, tmatch_update_thresh=0.35,
             tmatch_half=12, max_lost_tmatch=25, mot_k=3,
             active_radius_base=80, active_thresh=0.05, max_lost_active=15,
             kick_disp=30, kick_search_r=200, slow_speed=10.0, max_hold=40),
        # tighter template, larger active radius
        dict(tmatch_radius=35, tmatch_thresh=0.32, tmatch_update_thresh=0.45,
             tmatch_half=11, max_lost_tmatch=15, mot_k=3,
             active_radius_base=80, active_thresh=0.06, max_lost_active=12,
             kick_disp=40, kick_search_r=200, slow_speed=8.0, max_hold=35),
        # k=2 for faster response
        dict(tmatch_radius=40, tmatch_thresh=0.28, tmatch_update_thresh=0.40,
             tmatch_half=11, max_lost_tmatch=20, mot_k=2,
             active_radius_base=65, active_thresh=0.06, max_lost_active=12,
             kick_disp=25, kick_search_r=160, slow_speed=8.0, max_hold=35),
        # k=4 for slower-kick detection
        dict(tmatch_radius=40, tmatch_thresh=0.25, tmatch_update_thresh=0.38,
             tmatch_half=13, max_lost_tmatch=25, mot_k=4,
             active_radius_base=75, active_thresh=0.05, max_lost_active=15,
             kick_disp=40, kick_search_r=200, slow_speed=6.0, max_hold=40),
        # very loose for maximum coverage
        dict(tmatch_radius=50, tmatch_thresh=0.20, tmatch_update_thresh=0.30,
             tmatch_half=13, max_lost_tmatch=30, mot_k=3,
             active_radius_base=90, active_thresh=0.04, max_lost_active=18,
             kick_disp=25, kick_search_r=220, slow_speed=12.0, max_hold=50),
    ]

    # Random grid
    g = {
        "tmatch_radius":       [35, 40, 50],
        "tmatch_thresh":       [0.22, 0.28, 0.34],
        "tmatch_update_thresh":[0.35, 0.42],
        "tmatch_half":         [10, 12],
        "max_lost_tmatch":     [15, 25],
        "mot_k":               [2, 3, 4],
        "active_radius_base":  [60, 80, 110],
        "active_thresh":       [0.04, 0.07],
        "max_lost_active":     [10, 15],
        "kick_disp":           [28, 38],
        "kick_search_r":       [160, 200],
        "slow_speed":          [6.0, 10.0],
        "max_hold":            [30, 45],
    }
    keys = list(g.keys())
    all_combos = list(iproduct(*g.values()))
    sample = [dict(zip(keys,c)) for c in random.sample(all_combos, min(34,len(all_combos)))]
    all_run = hand_tuned + sample

    print(f"  Running {len(all_run)} combinations...")
    results = []
    for i, params in enumerate(all_run):
        pos = track_v4(frames, seed_frame, seed_pos, **params)
        ev  = evaluate(pos, labels)
        results.append((params, pos, ev))
        if (i+1) % 10 == 0:
            best = max(results, key=lambda x: x[2]['score'])
            print(f"    {i+1}/{len(all_run)} done — best: "
                  f"score={best[2]['score']:.4f} "
                  f"w50={best[2]['within50']} "
                  f"med={best[2]['median']:.0f}px")

    results.sort(key=lambda x: x[2]["score"], reverse=True)
    return results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",       required=True)
    parser.add_argument("--labels",      required=True)
    parser.add_argument("--frames-dir",  default=None)
    parser.add_argument("--render",      action="store_true")
    parser.add_argument("--out",         default="tracking_analysis_v4.txt")
    args = parser.parse_args()

    with open(args.labels) as f:
        data = json.load(f)
    labels = {e["frame"]: (e["px"], e["py"]) for e in data["labels"]}
    print(f"Loaded {len(labels)} labels")

    # Frames
    fdir = args.frames_dir
    if not (fdir and os.path.isdir(fdir)):
        fdir = "/tmp/soccer_v2_frames"
    existing = [f for f in os.listdir(fdir) if f.startswith("f_")] \
               if os.path.isdir(fdir) else []
    if existing:
        frames = sorted([os.path.join(fdir,f) for f in existing])
        print(f"Using {len(frames)} cached frames from {fdir}")
    else:
        os.makedirs(fdir, exist_ok=True)
        print("Extracting frames...")
        pat = os.path.join(fdir,"f_%04d.jpg")
        subprocess.run(["ffmpeg","-i",args.video,"-vf",f"fps={FPS}",
                        "-q:v","2","-loglevel","error",pat], check=True)
        frames = sorted([os.path.join(fdir,f) for f in os.listdir(fdir)
                         if f.startswith("f_")])
        print(f"Extracted {len(frames)} frames")

    print("\nFinding seed...")
    seed_idx, seed_px, seed_py = find_seed(frames)
    print(f"Seed: frame {seed_idx} ({seed_idx/FPS:.1f}s) px=({seed_px},{seed_py})")

    print("\n--- Baseline v4 ---")
    t0 = time.time()
    base_pos = track_v4(frames, seed_idx, (seed_px, seed_py))
    base_ev  = evaluate(base_pos, labels)
    print(f"Time: {time.time()-t0:.1f}s")
    pev("baseline_v4", base_ev)

    print("\n--- Grid search v4 ---")
    t0 = time.time()
    grid = grid_search(frames, seed_idx, (seed_px, seed_py), labels)
    print(f"Grid time: {time.time()-t0:.1f}s")

    print(f"\n{'='*80}")
    print("  TOP 10:")
    print(f"{'='*80}")
    for params, pos, ev in grid[:10]:
        name = (f"tr={params['tmatch_radius']} tt={params['tmatch_thresh']} "
                f"k={params['mot_k']} rb={params['active_radius_base']} "
                f"kd={params['kick_disp']} ss={params['slow_speed']}")
        pev(name, ev)

    wp, winner_pos, wev = grid[0]
    print(f"\n{'='*80}")
    print(f"  WINNER: {wp}")
    print(f"  Score={wev['score']:.4f}  Tracked={wev['tracked']}/{wev['total']}  "
          f"W30={wev['within30']}  W50={wev['within50']}  Median={wev['median']:.1f}px")
    print(f"{'='*80}\n")

    print("Per-frame detail (winner):")
    for fidx, gt in sorted(labels.items()):
        pred = winner_pos.get(fidx)
        if pred is None:
            print(f"  frame {fidx:4d}: NOT TRACKED  gt=({gt[0]},{gt[1]})")
        else:
            err  = dist(pred, gt)
            flag = " <-- MISS" if err > 50 else ""
            print(f"  frame {fidx:4d}: err={err:6.1f}px  "
                  f"pred=({pred[0]:4d},{pred[1]:4d})  "
                  f"gt=({gt[0]:4d},{gt[1]:4d}){flag}")

    with open(args.out, "w") as f:
        f.write(f"Winner params: {wp}\n")
        f.write(f"Score: {wev['score']:.4f}\n")
        f.write(f"Tracked: {wev['tracked']}/{wev['total']}\n")
        f.write(f"Within30: {wev['within30']}\n")
        f.write(f"Within50: {wev['within50']}\n")
        f.write(f"Median: {wev['median']:.1f}px\n\n")
        f.write("All results:\n")
        for p,_,ev in grid:
            f.write(f"  {p}: score={ev['score']:.4f} tracked={ev['tracked']}/{ev['total']} "
                    f"w30={ev['within30']} w50={ev['within50']} median={ev['median']:.1f}\n")
    print(f"\nSaved to {args.out}")

    if args.render:
        out_path = "tracking_v4_winner.mp4"
        print(f"\nRendering {out_path}...")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        ref = cv2.imread(frames[0])
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
