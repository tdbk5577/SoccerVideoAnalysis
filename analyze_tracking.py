#!/usr/bin/env python3
"""
Exhaustive ball tracking analysis - tries 25+ methods against ground truth labels.
Run: .venv/bin/python3 analyze_tracking.py --video Test.mp4 --labels ball_labels.json

Methods tested:
  Group A: Detection-only (how to find the ball in a search window)
  Group B: Prediction-only (where to look)
  Group C: Full trackers (prediction + detection combined)
  Group D: OpenCV built-in trackers
  Group E: Adaptive radius variants of best methods
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from itertools import product

import cv2
import numpy as np

W, H = 1920, 1080
FPS  = 10.0


# ============================================================
# Frame extraction (cached to avoid re-extracting)
# ============================================================

def extract_frames(video_path, fps, out_dir):
    pattern = os.path.join(out_dir, "f_%04d.jpg")
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2", "-loglevel", "error", pattern
    ], check=True)
    frames = sorted([os.path.join(out_dir, f)
                     for f in os.listdir(out_dir)
                     if f.startswith("f_") and f.endswith(".jpg")])
    return frames


def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return {e["frame"]: (e["px"], e["py"]) for e in data["labels"]}


# ============================================================
# Evaluation
# ============================================================

def evaluate(positions, labels):
    """Returns dict with tracked_count, within30, within50, median_err, mean_err."""
    errors = []
    not_tracked = 0
    for fidx, gt in labels.items():
        pred = positions.get(fidx)
        if pred is None:
            not_tracked += 1
            errors.append(None)
        else:
            err = ((pred[0]-gt[0])**2 + (pred[1]-gt[1])**2)**0.5
            errors.append(err)
    tracked = [e for e in errors if e is not None]
    if not tracked:
        return dict(tracked=0, total=len(errors), within30=0, within50=0,
                    median=9999, mean=9999, score=0)
    return dict(
        tracked   = len(tracked),
        total     = len(errors),
        within30  = sum(1 for e in tracked if e <= 30),
        within50  = sum(1 for e in tracked if e <= 50),
        median    = float(np.median(tracked)),
        mean      = float(np.mean(tracked)),
        # composite score: weight coverage + accuracy
        score     = len(tracked)/len(errors) * (sum(1 for e in tracked if e<=50)/len(tracked))
    )


# ============================================================
# Detection helpers
# ============================================================

KERNEL_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
KERNEL_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))


def get_roi(frame, cx, cy, r, w=W, h=H):
    x1 = max(0, cx-r); y1 = max(0, cy-r)
    x2 = min(w, cx+r); y2 = min(h, cy+r)
    return frame[y1:y2, x1:x2], x1, y1


def motion_mask(curr_gray, prev_gray, thresh=15):
    diff = cv2.absdiff(curr_gray, prev_gray)
    _, m = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  KERNEL_3)
    m = cv2.dilate(m, KERNEL_5, iterations=1)
    return m


def non_green_mask(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([30,25,25]), np.array([90,255,255]))
    return cv2.bitwise_not(green)


def blobs_in_mask(mask, min_area=5, max_area=800):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if min_area <= a <= max_area:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                p  = cv2.arcLength(c, True)
                circ = 4*np.pi*a/(p**2) if p > 0 else 0
                out.append((cx, cy, a, circ))
    return out


# ============================================================
# Prediction helpers
# ============================================================

def predict_linear(history, n=2):
    """Linear velocity from last n frames."""
    if len(history) < 2:
        return history[-1]
    pts = history[-n:]
    vx = pts[-1][0] - pts[-2][0]
    vy = pts[-1][1] - pts[-2][1]
    return (history[-1][0] + vx, history[-1][1] + vy)


def predict_avg_velocity(history, n=4):
    """Average velocity over last n steps."""
    if len(history) < 2:
        return history[-1]
    pts = history[-n:]
    if len(pts) < 2:
        return history[-1]
    vx = (pts[-1][0] - pts[0][0]) / (len(pts)-1)
    vy = (pts[-1][1] - pts[0][1]) / (len(pts)-1)
    return (history[-1][0] + vx, history[-1][1] + vy)


def current_speed(history):
    if len(history) < 2:
        return 0
    return ((history[-1][0]-history[-2][0])**2 +
            (history[-1][1]-history[-2][1])**2)**0.5


def adaptive_radius(history, base=60, min_r=40, max_r=220):
    """Scale search radius with current ball speed."""
    spd = current_speed(history)
    r = max(min_r, min(max_r, base + spd * 2.0))
    return int(r)


# ============================================================
# DETECTION METHODS
# ============================================================

def detect_motion_bright(curr, prev, px, py, radius, bright_thresh=90, mot_thresh=15):
    """Motion + non-green + bright blobs."""
    roi_c, x1, y1 = get_roi(curr, px, py, radius)
    roi_p, _,  _  = get_roi(prev, px, py, radius)
    if roi_c.size == 0:
        return None, 0.0

    gc = cv2.cvtColor(roi_c, cv2.COLOR_BGR2GRAY)
    gp = cv2.cvtColor(roi_p, cv2.COLOR_BGR2GRAY)
    mot = motion_mask(gc, gp, thresh=mot_thresh)
    ng  = non_green_mask(roi_c)
    br  = (gc > bright_thresh).astype(np.uint8)*255
    cand = cv2.bitwise_and(cv2.bitwise_and(mot, ng), br)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, KERNEL_3)

    blobs = blobs_in_mask(cand)
    best_pos, best_sc = None, 0.0
    for cx, cy, area, circ in blobs:
        fx, fy = x1+cx, y1+cy
        dist = ((fx-px)**2+(fy-py)**2)**0.5
        if dist > radius:
            continue
        bpatch = gc[max(0,cy-4):cy+4, max(0,cx-4):cx+4]
        bv = float(np.mean(bpatch)) if bpatch.size > 0 else 0
        sc = (1-dist/(radius+1)) * (circ+0.3) * (bv/180)
        if sc > best_sc:
            best_sc = sc; best_pos = (fx, fy)

    return best_pos, best_sc


def detect_bright_nongeen(curr, px, py, radius):
    """Brightest non-green blob near prediction (works when stationary)."""
    roi, x1, y1 = get_roi(curr, px, py, radius)
    if roi.size == 0:
        return None, 0.0
    gc  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ng  = non_green_mask(roi)
    cand = cv2.bitwise_and(gc, gc, mask=ng)
    blurred = cv2.GaussianBlur(cand.astype(np.float32), (7,7), 2)
    _, max_val, _, max_loc = cv2.minMaxLoc(blurred)
    if max_val < 100:
        return None, 0.0
    fx, fy = x1+max_loc[0], y1+max_loc[1]
    dist = ((fx-px)**2+(fy-py)**2)**0.5
    sc = (1-dist/(radius+1)) * (max_val/255) * 0.5
    return (fx, fy), sc


def detect_hough_motion(curr, prev, px, py, radius):
    """Hough circles on motion mask."""
    roi_c, x1, y1 = get_roi(curr, px, py, radius)
    roi_p, _,  _  = get_roi(prev, px, py, radius)
    if roi_c.size == 0:
        return None, 0.0
    gc = cv2.cvtColor(roi_c, cv2.COLOR_BGR2GRAY)
    gp = cv2.cvtColor(roi_p, cv2.COLOR_BGR2GRAY)
    mot = motion_mask(gc, gp)
    # Apply motion as weight to gray
    masked = cv2.bitwise_and(gc, gc, mask=mot)
    blurred = cv2.GaussianBlur(masked, (5,5), 1)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,
                               dp=1, minDist=8, param1=25, param2=6,
                               minRadius=3, maxRadius=16)
    if circles is None:
        return None, 0.0
    circles = np.round(circles[0]).astype(int)
    best_pos, best_sc = None, 0.0
    for cx, cy, r in circles:
        fx, fy = x1+cx, y1+cy
        dist = ((fx-px)**2+(fy-py)**2)**0.5
        if dist > radius:
            continue
        patch = gc[max(0,cy-r):cy+r, max(0,cx-r):cx+r]
        bv = float(np.mean(patch)) if patch.size > 0 else 0
        sc = (1-dist/(radius+1)) * (bv/200)
        if sc > best_sc:
            best_sc = sc; best_pos = (fx, fy)
    return best_pos, best_sc


def detect_hough_static(curr, px, py, radius):
    """Hough circles on plain gray."""
    roi, x1, y1 = get_roi(curr, px, py, radius)
    if roi.size == 0:
        return None, 0.0
    gc = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gc, (5,5), 1)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,
                               dp=1, minDist=8, param1=30, param2=8,
                               minRadius=3, maxRadius=14)
    if circles is None:
        return None, 0.0
    circles = np.round(circles[0]).astype(int)
    best_pos, best_sc = None, 0.0
    for cx, cy, r in circles:
        fx, fy = x1+cx, y1+cy
        dist = ((fx-px)**2+(fy-py)**2)**0.5
        if dist > radius:
            continue
        patch = gc[max(0,cy-r):cy+r, max(0,cx-r):cx+r]
        bv = float(np.mean(patch)) if patch.size > 0 else 0
        sc = (1-dist/(radius+1)) * (bv/200)
        if sc > best_sc:
            best_sc = sc; best_pos = (fx, fy)
    return best_pos, best_sc


def detect_template(curr, px, py, radius, tmpl):
    """Template matching."""
    roi, x1, y1 = get_roi(curr, px, py, radius)
    th, tw = tmpl.shape[:2]
    if roi.shape[0] < th or roi.shape[1] < tw:
        return None, 0.0
    res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    fx = x1 + max_loc[0] + tw//2
    fy = y1 + max_loc[1] + th//2
    return (fx, fy), float(max_val)


def detect_motion_only(curr, prev, px, py, radius, mot_thresh=15):
    """Fastest-moving small blob, no brightness filter."""
    roi_c, x1, y1 = get_roi(curr, px, py, radius)
    roi_p, _,  _  = get_roi(prev, px, py, radius)
    if roi_c.size == 0:
        return None, 0.0
    gc = cv2.cvtColor(roi_c, cv2.COLOR_BGR2GRAY)
    gp = cv2.cvtColor(roi_p, cv2.COLOR_BGR2GRAY)
    mot = motion_mask(gc, gp, thresh=mot_thresh)
    blobs = blobs_in_mask(mot, max_area=600)
    best_pos, best_sc = None, 0.0
    for cx, cy, area, circ in blobs:
        fx, fy = x1+cx, y1+cy
        dist = ((fx-px)**2+(fy-py)**2)**0.5
        if dist > radius:
            continue
        sc = (1-dist/(radius+1)) * (circ+0.2)
        if sc > best_sc:
            best_sc = sc; best_pos = (fx, fy)
    return best_pos, best_sc


def detect_combined(curr, prev, px, py, radius, bright_thresh=90, mot_thresh=15):
    """Try motion+bright first, fall back to pure brightness."""
    pos, sc = detect_motion_bright(curr, prev, px, py, radius,
                                   bright_thresh=bright_thresh,
                                   mot_thresh=mot_thresh)
    if sc < 0.08:
        pos2, sc2 = detect_bright_nongeen(curr, px, py, radius)
        if sc2 > sc:
            pos, sc = pos2, sc2 * 0.85   # slight penalty for fallback
    return pos, sc


def detect_mog2(subtractor, curr, px, py, radius):
    """Background subtraction (MOG2)."""
    fg = subtractor.apply(curr)
    roi, x1, y1 = get_roi(fg, px, py, radius, w=fg.shape[1], h=fg.shape[0])
    if roi.size == 0:
        return None, 0.0
    roi_2d = roi if len(roi.shape) == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(roi_2d, 128, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, KERNEL_3)
    blobs = blobs_in_mask(bw, max_area=600)
    best_pos, best_sc = None, 0.0
    for cx, cy, area, circ in blobs:
        fx, fy = x1+cx, y1+cy
        dist = ((fx-px)**2+(fy-py)**2)**0.5
        if dist > radius:
            continue
        sc = (1-dist/(radius+1)) * (circ+0.2)
        if sc > best_sc:
            best_sc = sc; best_pos = (fx, fy)
    return best_pos, best_sc


# ============================================================
# GENERIC TRACKER RUNNER
# ============================================================

def run_tracker(frames, labels, seed_frame, seed_pos,
                detect_fn,          # fn(curr, prev, pred_px, pred_py, radius) -> (pos, score)
                predict_fn,         # fn(history) -> (px, py)
                radius_fn,          # fn(history) -> int
                score_thresh=0.10,
                max_lost=10):
    """
    Generic frame-by-frame tracker.
    detect_fn signature varies; wrap it before passing.
    """
    n = len(frames)
    positions = {}
    positions[seed_frame] = seed_pos

    def run_direction(start_idx, start_pos, step):
        history = [start_pos]
        lost    = 0
        idx     = start_idx + step
        while 0 <= idx < n:
            curr_img = cv2.imread(frames[idx])
            prev_idx = idx - step
            prev_img = cv2.imread(frames[prev_idx]) if 0 <= prev_idx < n else None
            if curr_img is None:
                break

            pred = predict_fn(history)
            pred_px, pred_py = int(pred[0]), int(pred[1])
            radius = radius_fn(history)

            pos, score = detect_fn(curr_img, prev_img, pred_px, pred_py, radius)

            if score >= score_thresh and pos is not None:
                positions[idx] = pos
                history.append(pos)
                lost = 0
            else:
                lost += 1
                if lost <= max_lost:
                    ipos = (pred_px, pred_py)
                    positions[idx] = ipos
                    history.append(ipos)
                else:
                    break

            if len(history) > 8:
                history.pop(0)
            idx += step

    run_direction(seed_frame, seed_pos, +1)
    run_direction(seed_frame, seed_pos, -1)
    return positions


# ============================================================
# SEED DETECTION
# ============================================================

def find_seed(frames, w=W, h=H, avg_n=25):
    """
    Temporal average + Hough circles for stationary ball,
    then motion-based fallback.
    Returns (frame_idx, px, py).
    """
    # Strategy 1: temporal average over first avg_n frames
    x1r, x2r = int(0.35*w), int(0.65*w)
    y1r, y2r = int(0.33*h), int(0.67*h)
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
                return 0, best[0], best[1]

    # Strategy 2: motion-based
    x1r, x2r = int(0.22*w), int(0.78*w)
    y1r, y2r = int(0.20*h), int(0.80*h)
    for i in range(min(60, len(frames)-3)):
        blobs_01 = []
        for ia, ib in [(i,i+1)]:
            fa = cv2.imread(frames[ia]); fb = cv2.imread(frames[ib])
            if fa is None or fb is None: continue
            ga = cv2.cvtColor(fa[y1r:y2r, x1r:x2r], cv2.COLOR_BGR2GRAY)
            gb = cv2.cvtColor(fb[y1r:y2r, x1r:x2r], cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(ga, gb)
            _, mot = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            mot = cv2.morphologyEx(mot, cv2.MORPH_OPEN, KERNEL_3)
            cnts, _ = cv2.findContours(mot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                a = cv2.contourArea(c)
                if 8 < a < 500:
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        blobs_01.append((int(M["m10"]/M["m00"])+x1r,
                                         int(M["m01"]/M["m00"])+y1r, a))
        center = [b for b in blobs_01
                  if int(0.30*w) < b[0] < int(0.70*w)
                  and int(0.25*h) < b[1] < int(0.75*h)]
        if center:
            # pick smallest (most ball-like)
            b = min(center, key=lambda x: x[2])
            return i+1, b[0], b[1]

    # Last resort: center of frame
    return 0, w//2, h//2


# ============================================================
# BUILD ALL METHODS
# ============================================================

def build_methods(frames, seed_frame, seed_pos):
    """Return list of (name, positions_dict) for all methods."""
    results = []

    def constant_radius(r):
        return lambda h: r

    def adaptive_r(base=80):
        return lambda h: adaptive_radius(h, base=base)

    def linear2(h): return predict_linear(h, 2)
    def linear3(h): return predict_linear(h, 3)
    def avg_vel4(h): return predict_avg_velocity(h, 4)

    # -------------------------------------------------------
    # Methods 1-6: motion+bright, fixed radius, linear2
    # -------------------------------------------------------
    for r, thresh, name_sfx in [
        (60,  0.10, "r60_t10"),
        (80,  0.10, "r80_t10"),
        (120, 0.10, "r120_t10"),
        (80,  0.15, "r80_t15"),
        (120, 0.15, "r120_t15"),
        (150, 0.10, "r150_t10"),
    ]:
        def dfn(curr, prev, px, py, radius, _r=r):
            return detect_motion_bright(curr, prev, px, py, _r)
        pos = run_tracker(frames, {}, seed_frame, seed_pos,
                          dfn, linear2, constant_radius(r),
                          score_thresh=thresh)
        results.append((f"motion_bright_{name_sfx}", pos))

    # -------------------------------------------------------
    # Methods 7-9: motion+bright + adaptive radius
    # -------------------------------------------------------
    for base, thresh, name_sfx in [
        (60, 0.10, "adapt60_t10"),
        (80, 0.10, "adapt80_t10"),
        (80, 0.08, "adapt80_t08"),
    ]:
        def dfn(curr, prev, px, py, radius, _b=base):
            return detect_motion_bright(curr, prev, px, py, radius)
        pos = run_tracker(frames, {}, seed_frame, seed_pos,
                          dfn, linear2, adaptive_r(base),
                          score_thresh=thresh)
        results.append((f"motion_bright_{name_sfx}", pos))

    # -------------------------------------------------------
    # Methods 10-13: combined (motion+bright fallback brightness)
    # -------------------------------------------------------
    for r, thresh, name_sfx in [
        (80,  0.08, "r80_t08"),
        (120, 0.08, "r120_t08"),
        (80,  0.10, "r80_t10"),
        (120, 0.10, "r120_t10"),
    ]:
        def dfn(curr, prev, px, py, radius, _r=r):
            return detect_combined(curr, prev, px, py, _r)
        pos = run_tracker(frames, {}, seed_frame, seed_pos,
                          dfn, linear2, constant_radius(r),
                          score_thresh=thresh)
        results.append((f"combined_{name_sfx}", pos))

    # -------------------------------------------------------
    # Methods 14-15: combined with adaptive radius
    # -------------------------------------------------------
    for base, thresh in [(80, 0.08), (100, 0.08)]:
        def dfn(curr, prev, px, py, radius, _base=base):
            return detect_combined(curr, prev, px, py, radius)
        pos = run_tracker(frames, {}, seed_frame, seed_pos,
                          dfn, linear2, adaptive_r(base),
                          score_thresh=thresh)
        results.append((f"combined_adapt{base}_t{int(thresh*100):02d}", pos))

    # -------------------------------------------------------
    # Methods 16-17: motion only (no brightness)
    # -------------------------------------------------------
    for r, thresh in [(80, 0.10), (120, 0.10)]:
        def dfn(curr, prev, px, py, radius, _r=r):
            return detect_motion_only(curr, prev, px, py, _r)
        pos = run_tracker(frames, {}, seed_frame, seed_pos,
                          dfn, linear2, constant_radius(r),
                          score_thresh=thresh)
        results.append((f"motion_only_r{r}", pos))

    # -------------------------------------------------------
    # Methods 18-19: Hough on motion mask
    # -------------------------------------------------------
    for r, thresh in [(80, 0.05), (120, 0.05)]:
        def dfn(curr, prev, px, py, radius, _r=r):
            return detect_hough_motion(curr, prev, px, py, _r)
        pos = run_tracker(frames, {}, seed_frame, seed_pos,
                          dfn, linear2, constant_radius(r),
                          score_thresh=thresh)
        results.append((f"hough_motion_r{r}", pos))

    # -------------------------------------------------------
    # Method 20: Hough static
    # -------------------------------------------------------
    def dfn_hs(curr, prev, px, py, radius):
        return detect_hough_static(curr, px, py, radius)
    pos = run_tracker(frames, {}, seed_frame, seed_pos,
                      dfn_hs, linear2, constant_radius(80),
                      score_thresh=0.05)
    results.append(("hough_static_r80", pos))

    # -------------------------------------------------------
    # Method 21: brightness only
    # -------------------------------------------------------
    def dfn_br(curr, prev, px, py, radius):
        return detect_bright_nongeen(curr, px, py, radius)
    pos = run_tracker(frames, {}, seed_frame, seed_pos,
                      dfn_br, linear2, constant_radius(80),
                      score_thresh=0.05)
    results.append(("bright_nongeen_r80", pos))

    # -------------------------------------------------------
    # Methods 22-23: different velocity predictors
    # -------------------------------------------------------
    def dfn_c(curr, prev, px, py, radius):
        return detect_combined(curr, prev, px, py, radius)
    for pred_fn, name in [(linear3, "linear3"), (avg_vel4, "avgvel4")]:
        pos = run_tracker(frames, {}, seed_frame, seed_pos,
                          dfn_c, pred_fn, constant_radius(100),
                          score_thresh=0.08)
        results.append((f"combined_{name}_r100", pos))

    # -------------------------------------------------------
    # Method 24: MOG2 background subtraction
    # -------------------------------------------------------
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=30, varThreshold=25, detectShadows=False)
    # Warm up
    for i in range(min(20, len(frames))):
        f = cv2.imread(frames[i])
        if f is not None:
            subtractor.apply(f)

    def dfn_mog(curr, prev, px, py, radius, _sub=subtractor):
        return detect_mog2(_sub, curr, px, py, radius)
    pos = run_tracker(frames, {}, seed_frame, seed_pos,
                      dfn_mog, linear2, constant_radius(80),
                      score_thresh=0.05)
    results.append(("mog2_r80", pos))

    # -------------------------------------------------------
    # Method 25: combined + extra-wide radius + low thresh
    # -------------------------------------------------------
    def dfn_wide(curr, prev, px, py, radius):
        return detect_combined(curr, prev, px, py, radius,
                               bright_thresh=80, mot_thresh=12)
    pos = run_tracker(frames, {}, seed_frame, seed_pos,
                      dfn_wide, linear2, adaptive_r(100),
                      score_thresh=0.06, max_lost=15)
    results.append(("combined_wide_adapt100", pos))

    # -------------------------------------------------------
    # Methods 26-28: OpenCV built-in trackers
    # -------------------------------------------------------
    for tracker_name in ["CSRT", "KCF", "MOSSE"]:
        pos = run_opencv_tracker(frames, seed_frame, seed_pos, tracker_name)
        results.append((f"opencv_{tracker_name}", pos))

    # -------------------------------------------------------
    # Method 29: two-stage — tight then wide fallback
    # -------------------------------------------------------
    pos = run_two_stage(frames, seed_frame, seed_pos)
    results.append(("two_stage_adaptive", pos))

    # -------------------------------------------------------
    # Method 30: velocity-boost on fast kick detection
    # -------------------------------------------------------
    pos = run_velocity_boost(frames, seed_frame, seed_pos)
    results.append(("velocity_boost", pos))

    return results


# ============================================================
# OPENCV BUILT-IN TRACKERS
# ============================================================

def run_opencv_tracker(frames, seed_frame, seed_pos, tracker_name):
    """Run an OpenCV built-in tracker seeded from ground truth."""
    positions = {}
    seed_img = cv2.imread(frames[seed_frame])
    if seed_img is None:
        return positions

    px, py = seed_pos
    half = 16
    bbox = (max(0, px-half), max(0, py-half), half*2, half*2)

    def make_tracker():
        if tracker_name == "CSRT":
            return cv2.TrackerCSRT_create()
        elif tracker_name == "KCF":
            return cv2.TrackerKCF_create()
        elif tracker_name == "MOSSE":
            return cv2.legacy.TrackerMOSSE_create() if hasattr(cv2, 'legacy') \
                   else cv2.TrackerMOSSE_create()
        return None

    # Forward
    try:
        tr = make_tracker()
        if tr is None:
            return positions
        tr.init(seed_img, bbox)
        positions[seed_frame] = seed_pos
        for i in range(seed_frame+1, len(frames)):
            img = cv2.imread(frames[i])
            if img is None:
                break
            ok, box = tr.update(img)
            if ok:
                x, y, bw, bh = [int(v) for v in box]
                positions[i] = (x + bw//2, y + bh//2)
            else:
                break
    except Exception:
        pass

    # Backward
    try:
        tr = make_tracker()
        if tr is None:
            return positions
        tr.init(seed_img, bbox)
        for i in range(seed_frame-1, -1, -1):
            img = cv2.imread(frames[i])
            if img is None:
                break
            ok, box = tr.update(img)
            if ok:
                x, y, bw, bh = [int(v) for v in box]
                positions[i] = (x + bw//2, y + bh//2)
            else:
                break
    except Exception:
        pass

    return positions


# ============================================================
# TWO-STAGE ADAPTIVE TRACKER
# ============================================================

def run_two_stage(frames, seed_frame, seed_pos):
    """
    Tight search first; on failure, expand to very wide radius.
    Distinguishes 'lost' from 'fast ball' by score gap.
    """
    n = len(frames)
    positions = {seed_frame: seed_pos}

    def run_dir(start, start_pos, step):
        history = [start_pos]
        lost = 0
        idx  = start + step
        while 0 <= idx < n:
            curr = cv2.imread(frames[idx])
            prev_idx = idx - step
            prev = cv2.imread(frames[prev_idx]) if 0 <= prev_idx < n else None
            if curr is None:
                break

            spd = current_speed(history)
            # Tight radius based on current speed
            tight_r = max(50, min(180, int(spd * 2.5 + 55)))

            pred = predict_linear(history)
            px, py = int(pred[0]), int(pred[1])

            pos, sc = detect_combined(curr, prev, px, py, tight_r)

            if sc < 0.08:
                # Wide fallback: ball may have been kicked hard
                pos, sc = detect_combined(curr, prev, px, py, min(220, tight_r*2))

            if sc >= 0.07 and pos is not None:
                positions[idx] = pos
                history.append(pos)
                lost = 0
            else:
                lost += 1
                if lost <= 12:
                    ipos = (px, py)
                    positions[idx] = ipos
                    history.append(ipos)
                else:
                    break

            if len(history) > 8:
                history.pop(0)
            idx += step

    run_dir(seed_frame, seed_pos, +1)
    run_dir(seed_frame, seed_pos, -1)
    return positions


# ============================================================
# VELOCITY-BOOST TRACKER
# ============================================================

def run_velocity_boost(frames, seed_frame, seed_pos):
    """
    When the ball suddenly moves much faster than predicted,
    expand search radius dramatically to catch the kick.
    """
    n = len(frames)
    positions = {seed_frame: seed_pos}

    def run_dir(start, start_pos, step):
        history  = [start_pos]
        lost     = 0
        idx      = start + step

        while 0 <= idx < n:
            curr = cv2.imread(frames[idx])
            prev_idx = idx - step
            prev = cv2.imread(frames[prev_idx]) if 0 <= prev_idx < n else None
            if curr is None:
                break

            spd  = current_speed(history)
            pred = predict_linear(history)
            px, py = int(pred[0]), int(pred[1])

            # Attempt 1: tight (trust velocity)
            r1 = max(45, min(160, int(spd * 2 + 50)))
            pos, sc = detect_combined(curr, prev, px, py, r1)

            # Attempt 2: wider (ball may have been kicked hard)
            if sc < 0.08:
                r2 = max(120, min(250, int(spd * 4 + 120)))
                pos2, sc2 = detect_combined(curr, prev, px, py, r2)
                if sc2 > sc:
                    pos, sc = pos2, sc2

            if sc >= 0.07 and pos is not None:
                positions[idx] = pos
                history.append(pos)
                lost = 0
            else:
                lost += 1
                if lost <= 12:
                    ipos = (px, py)
                    positions[idx] = ipos
                    history.append(ipos)
                else:
                    break

            if len(history) > 8:
                history.pop(0)
            idx += step

    run_dir(seed_frame, seed_pos, +1)
    run_dir(seed_frame, seed_pos, -1)
    return positions


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--frames-dir", default=None,
                        help="Pre-extracted frames dir (skip re-extraction)")
    parser.add_argument("--out", default="tracking_analysis.txt")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    print(f"Loaded {len(labels)} ground-truth labels")

    if args.frames_dir and os.path.isdir(args.frames_dir):
        frames = sorted([os.path.join(args.frames_dir, f)
                         for f in os.listdir(args.frames_dir)
                         if f.startswith("f_") and f.endswith(".jpg")])
        print(f"Using pre-extracted {len(frames)} frames from {args.frames_dir}")
        tmpdir_ctx = None
    else:
        import tempfile
        tmpdir_ctx = tempfile.mkdtemp(prefix="soccer_frames_")
        print(f"Extracting frames to {tmpdir_ctx} ...")
        frames = extract_frames(args.video, FPS, tmpdir_ctx)
        print(f"Extracted {len(frames)} frames")
        # Save dir for reuse
        with open("frames_dir.txt", "w") as f:
            f.write(tmpdir_ctx)
        print(f"Frames dir saved to frames_dir.txt — reuse with --frames-dir {tmpdir_ctx}")

    print(f"\nFinding seed position...")
    seed_idx, seed_px, seed_py = find_seed(frames)
    print(f"Seed: frame {seed_idx} ({seed_idx/FPS:.1f}s) px=({seed_px},{seed_py})")

    print(f"\nBuilding and running {30} tracking methods...\n")
    method_results = build_methods(frames, seed_idx, (seed_px, seed_py))

    print(f"\n{'='*72}")
    print(f"  RESULTS (sorted by score = coverage × within50_rate)")
    print(f"{'='*72}")
    print(f"  {'Method':<35} {'Tracked':>7} {'W30':>5} {'W50':>5} {'Med':>6} {'Score':>7}")
    print(f"  {'-'*35} {'-'*7} {'-'*5} {'-'*5} {'-'*6} {'-'*7}")

    scored = []
    for name, positions in method_results:
        ev = evaluate(positions, labels)
        scored.append((name, positions, ev))

    scored.sort(key=lambda x: x[2]["score"], reverse=True)

    for name, _, ev in scored:
        print(f"  {name:<35} {ev['tracked']:>3}/{ev['total']:<3} "
              f"{ev['within30']:>5} {ev['within50']:>5} "
              f"{ev['median']:>6.1f} {ev['score']:>7.3f}")

    # Save full per-frame breakdown for winner
    winner_name, winner_pos, winner_ev = scored[0]
    print(f"\n{'='*72}")
    print(f"  WINNER: {winner_name}")
    print(f"  Score={winner_ev['score']:.3f}  "
          f"Tracked={winner_ev['tracked']}/{winner_ev['total']}  "
          f"Within30={winner_ev['within30']}  "
          f"Median={winner_ev['median']:.1f}px")
    print(f"{'='*72}\n")

    print("  Per-frame detail (winner):")
    for fidx, gt in sorted(labels.items()):
        pred = winner_pos.get(fidx)
        if pred is None:
            print(f"    frame {fidx:3d}: NOT TRACKED  gt=({gt[0]},{gt[1]})")
        else:
            err = ((pred[0]-gt[0])**2+(pred[1]-gt[1])**2)**0.5
            flag = " <-- MISS" if err > 50 else ""
            print(f"    frame {fidx:3d}: err={err:5.1f}px  "
                  f"pred=({pred[0]:4d},{pred[1]:4d})  "
                  f"gt=({gt[0]:4d},{gt[1]:4d}){flag}")

    # Save results to file
    with open(args.out, "w") as f:
        f.write(f"Winner: {winner_name}\n")
        f.write(f"Score: {winner_ev['score']:.4f}\n")
        f.write(f"Tracked: {winner_ev['tracked']}/{winner_ev['total']}\n")
        f.write(f"Within30: {winner_ev['within30']}\n")
        f.write(f"Within50: {winner_ev['within50']}\n")
        f.write(f"Median error: {winner_ev['median']:.1f}px\n\n")
        f.write("All methods:\n")
        for name, _, ev in scored:
            f.write(f"  {name}: score={ev['score']:.3f} "
                    f"tracked={ev['tracked']}/{ev['total']} "
                    f"w30={ev['within30']} w50={ev['within50']} "
                    f"median={ev['median']:.1f}\n")

    print(f"\nResults saved to {args.out}")

    # Render winner video
    print("\nRendering winner tracking video...")
    out_path = f"tracking_winner_{winner_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ref = cv2.imread(frames[0])
    ww, hh = ref.shape[1], ref.shape[0]
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (ww, hh))
    for i, fp in enumerate(frames):
        img = cv2.imread(fp)
        if img is None:
            continue
        t = i / FPS
        cv2.putText(img, f"{t:.1f}s", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        pos = winner_pos.get(i)
        if pos:
            cv2.circle(img, pos, 14, (0,255,0), 3)
            cv2.circle(img, pos,  3, (0,255,0), -1)
        # Draw GT if labeled
        gt = labels.get(i)
        if gt:
            cv2.circle(img, gt, 8, (0,0,255), 2)
        writer.write(img)
    writer.release()
    print(f"Saved: {out_path}  (green=predicted, red=ground truth)")


if __name__ == "__main__":
    main()
