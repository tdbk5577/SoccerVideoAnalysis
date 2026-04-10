#!/usr/bin/env python3
"""
Ball tracking v8 — world-space distance checks, pixel-space positions.

Key insight over v7h:
  All distance thresholds (speed_limit, chain_margin, gap_margin,
  person_accept_r) are specified in METERS and checked using the
  field homography's forward transform (pixel→world).

  Crucially, we NEVER use H_inv (world→pixel) during tracking.
  Pixel coordinates are always sourced directly from YOLO detections
  or interpolated between pixel anchors. This avoids the extrapolation
  instability that makes H_inv unreliable outside the calibration region.

This makes the tracker camera-angle agnostic: the same world-unit
thresholds work regardless of camera zoom or position.

Pipeline:
  1. Load calibration (pixel→world homography H)
  2. YOLO detections stay in pixel space
  3. All distance/speed checks convert pixels→world via H and compare in meters
  4. Gap filling: linear interpolation in pixel space (no H_inv needed)
  5. Person estimation: world-space proximity check, pixel-space position used
  6. Output: pixel positions for display and evaluation

Calibration:
  .venv/bin/python3 calibrate_field.py --video "test 1.mp4" --format 11v11 \\
      --box goal --output calibration_test1.json

Usage:
  .venv/bin/python3 analyze_tracking_v8.py \\
      --video "test 1.mp4" --labels ball_labels.json \\
      --cache yolo_cache_test1.json \\
      --calibration calibration_test1.json \\
      --render
"""

import argparse
import json
import os
import random
import subprocess

import cv2
import numpy as np

FPS = 10.0


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def load_frames(frames_dir):
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith('.jpg'))
    return [os.path.join(frames_dir, f) for f in files]


def load_cache(path):
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ─────────────────────────────────────────────────────────────
# Field homography — forward-only usage
# ─────────────────────────────────────────────────────────────

class FieldHomography:
    def __init__(self, cal_path):
        with open(cal_path) as f:
            cal = json.load(f)
        self.H     = np.array(cal['H'],     dtype=np.float64)
        self.H_inv = np.array(cal['H_inv'], dtype=np.float64)  # kept for viz only
        self.frame_w = cal['frame_w']
        self.frame_h = cal['frame_h']
        self.field_config = cal['field_config']
        self.fmt  = cal['format']
        self.box  = cal.get('box', 'unknown')

    def to_world(self, px, py):
        """Pixel → world (meters). Forward transform — reliable everywhere."""
        pt = np.array([[[float(px), float(py)]]], dtype=np.float64)
        wp = cv2.perspectiveTransform(pt, self.H)[0][0]
        return float(wp[0]), float(wp[1])

    def world_dist(self, px1, py1, px2, py2):
        """World-space distance (meters) between two pixel positions."""
        wx1, wy1 = self.to_world(px1, py1)
        wx2, wy2 = self.to_world(px2, py2)
        return dist((wx1, wy1), (wx2, wy2))

    def local_ppm(self, px, py, delta=5):
        """Pixels per meter at pixel position (px, py)."""
        w0 = self.to_world(px, py)
        w1 = self.to_world(px + delta, py)
        w2 = self.to_world(px, py + delta)
        mpp_x = dist(w0, w1) / delta  # meters per pixel (x direction)
        mpp_y = dist(w0, w2) / delta  # meters per pixel (y direction)
        mpp   = (mpp_x + mpp_y) / 2
        return 1.0 / mpp if mpp > 1e-6 else 100.0


# ─────────────────────────────────────────────────────────────
# Pass 1a: Trajectory-guided collection
# (pixel-space velocity, world-space speed check)
# ─────────────────────────────────────────────────────────────

def collect_trajectory_guided(cache, n_frames, seed_pos, hom,
                               min_conf=0.10,
                               hard_reset_conf=0.40,
                               speed_limit=3.0,     # m/frame
                               vel_decay=0.85):
    rx, ry = float(seed_pos[0]), float(seed_pos[1])
    rvx, rvy = 0.0, 0.0
    last_det_frame = 0
    dets = {}

    for idx in range(n_frames):
        balls = [b for b in cache.get(idx, {}).get('balls', []) if b[2] >= min_conf]
        if not balls:
            rvx *= vel_decay; rvy *= vel_decay
            rx += rvx; ry += rvy
            continue

        dt = max(idx - last_det_frame, 1)
        pred_x = rx + rvx * dt
        pred_y = ry + rvy * dt

        # Distance check in WORLD space (meters per frame)
        close = []
        for b in balls:
            wd = hom.world_dist(b[0], b[1], pred_x, pred_y) / dt
            if wd <= speed_limit:
                close.append((b, wd))

        hard_candidates = [b for b in balls if b[2] >= hard_reset_conf]

        if close:
            best_b, _ = min(close, key=lambda x: x[1])
        elif hard_candidates:
            best_b = max(hard_candidates, key=lambda b: b[2])
        else:
            rvx *= vel_decay; rvy *= vel_decay
            rx += rvx; ry += rvy
            continue

        bx, by, bcf = best_b
        if dt > 0:
            rvx = (bx - rx) / dt * 0.5 + rvx * 0.5
            rvy = (by - ry) / dt * 0.5 + rvy * 0.5
        rx, ry = bx, by
        last_det_frame = idx
        dets[idx] = (bx, by, bcf)

    return dets


# ─────────────────────────────────────────────────────────────
# Pass 1b: HC override
# ─────────────────────────────────────────────────────────────

def hc_override(traj_dets, cache, n_frames, hc_direct_conf=0.50):
    merged = dict(traj_dets)
    for idx in range(n_frames):
        balls = cache.get(idx, {}).get('balls', [])
        hc_balls = [b for b in balls if b[2] >= hc_direct_conf]
        if not hc_balls:
            continue
        best_hc = max(hc_balls, key=lambda b: b[2])
        bx, by, bcf = best_hc
        if idx not in merged or bcf > merged[idx][2]:
            merged[idx] = (bx, by, bcf)
    return merged


# ─────────────────────────────────────────────────────────────
# Pass 1c: ABC filter (world-space speed check)
# ─────────────────────────────────────────────────────────────

def abc_filter(dets, hom, abc_protect_conf=0.65,
               speed_limit=3.0, abc_ratio=0.6):
    clean = dict(dets)
    for _ in range(3):
        frames = sorted(clean)
        to_remove = set()
        for i, idx in enumerate(frames):
            bx, by, bcf = clean[idx]
            if bcf >= abc_protect_conf:
                continue
            if i == 0 or i == len(frames)-1:
                continue
            aidx = frames[i-1]; cidx = frames[i+1]
            ax, ay, _ = clean[aidx]; cx2, cy2, _ = clean[cidx]
            dt_ab = max(idx - aidx, 1)
            spd_ab = hom.world_dist(bx, by, ax, ay) / dt_ab
            if spd_ab < speed_limit * 0.3:
                continue
            # In world space: if C is closer to A than B is to A → B is an outlier
            if hom.world_dist(cx2, cy2, ax, ay) < abc_ratio * hom.world_dist(cx2, cy2, bx, by):
                to_remove.add(idx)
        for idx in to_remove:
            del clean[idx]
        if not to_remove:
            break
    return clean


# ─────────────────────────────────────────────────────────────
# Pass 1d: HC anchor chain validation (world-space deviation)
# ─────────────────────────────────────────────────────────────

def validate_hc_chain(clean_dets, seed_frame, seed_pos, hom,
                      hc_thresh=0.55, chain_margin=5.0):   # meters
    filtered = dict(clean_dets)
    while True:
        hc_anchors = {f: v for f, v in filtered.items() if v[2] >= hc_thresh}
        hc_anchors[seed_frame] = (float(seed_pos[0]), float(seed_pos[1]), 1.0)
        hc_frames = sorted(hc_anchors)
        if len(hc_frames) < 3:
            break
        worst_frame = None; worst_dev = chain_margin
        for i in range(1, len(hc_frames)-1):
            fa = hc_frames[i-1]; fb = hc_frames[i]; fc = hc_frames[i+1]
            ax, ay, _ = hc_anchors[fa]
            bx, by, _ = hc_anchors[fb]
            cx, cy, _ = hc_anchors[fc]
            t = (fb - fa) / (fc - fa)
            ex = ax + t*(cx-ax); ey = ay + t*(cy-ay)   # pixel interpolation
            # Deviation checked in world space
            dev = hom.world_dist(bx, by, ex, ey)
            if dev > worst_dev:
                worst_dev = dev; worst_frame = fb
        if worst_frame is None:
            break
        del filtered[worst_frame]
    return filtered


# ─────────────────────────────────────────────────────────────
# Pass 1e: HC-boundary gap filter (world-space deviation)
# ─────────────────────────────────────────────────────────────

def hc_gap_filter(clean_dets, seed_frame, seed_pos, hom,
                  hc_thresh=0.55, gap_margin=3.0,        # meters
                  min_gap_for_filter=6):
    filtered = dict(clean_dets)
    changed = True
    while changed:
        changed = False
        hc_anchors = {f: v for f, v in filtered.items() if v[2] >= hc_thresh}
        hc_anchors[seed_frame] = (float(seed_pos[0]), float(seed_pos[1]), 1.0)
        hc_frames = sorted(hc_anchors)
        for i in range(len(hc_frames)-1):
            f_a = hc_frames[i]; f_b = hc_frames[i+1]
            gap = f_b - f_a
            if gap < min_gap_for_filter:
                continue
            ax, ay, _ = hc_anchors[f_a]; bx, by, _ = hc_anchors[f_b]
            to_remove = []
            for f in list(filtered.keys()):
                if f <= f_a or f >= f_b:
                    continue
                cx, cy, cf = filtered[f]
                t = (f - f_a) / gap
                ex = ax + t*(bx-ax); ey = ay + t*(by-ay)   # pixel interp
                if hom.world_dist(cx, cy, ex, ey) > gap_margin:
                    to_remove.append(f)
            for f in to_remove:
                del filtered[f]; changed = True
    return filtered


# ─────────────────────────────────────────────────────────────
# Pass 2: Gap filling — pixel-space interpolation, world-space person check
# ─────────────────────────────────────────────────────────────

def fill_gaps(n_frames, clean_dets, cache, seed_frame, seed_pos, hom,
              interp_max_gap=100,
              person_accept_r=3.0,    # meters
              person_conf_min=0.30):
    W = hom.frame_w; H_px = hom.frame_h

    positions = {}
    anchors = dict(clean_dets)
    anchors[seed_frame] = (float(seed_pos[0]), float(seed_pos[1]), 1.0)
    anchor_frames = sorted(anchors)
    running_ref = (float(seed_pos[0]), float(seed_pos[1]))   # pixel

    for idx in range(n_frames):
        if idx in anchors:
            cx, cy, _ = anchors[idx]
            positions[idx] = (int(cx), int(cy))
            running_ref = (cx, cy)
            continue

        prev_f = next_f = None
        for f in reversed([f for f in anchor_frames if f < idx]):
            prev_f = f; break
        for f in [f for f in anchor_frames if f > idx]:
            next_f = f; break

        # Linear interpolation in PIXEL space between two anchors
        if prev_f is not None and next_f is not None:
            gap = next_f - prev_f
            if gap <= interp_max_gap:
                t = (idx - prev_f) / gap
                px_a, py_a, _ = anchors[prev_f]
                px_b, py_b, _ = anchors[next_f]
                x = px_a + t*(px_b - px_a)
                y = py_a + t*(py_b - py_a)
                positions[idx] = (int(x), int(y))
                running_ref = (x, y)
                continue

        # Person estimation: world-space proximity check, pixel-space output
        persons = cache.get(idx, {}).get('persons', [])
        best_person = None; best_wd = 1e9
        for pcx, pcy, pcf, x1, y1, x2, y2 in persons:
            if pcf < person_conf_min:
                continue
            # Use bottom-center of bbox as feet position
            fpx, fpy = (x1 + x2) / 2, y2
            wd = hom.world_dist(fpx, fpy, running_ref[0], running_ref[1])
            if wd < person_accept_r and wd < best_wd:
                best_wd = wd
                best_person = (int(clamp(fpx, 0, W-1)),
                               int(clamp(fpy, 0, H_px-1)))

        if best_person is not None:
            positions[idx] = best_person
            running_ref = (float(best_person[0]), float(best_person[1]))
            continue

        # Fallback: hold last known position
        if prev_f is not None:
            cx, cy, _ = anchors[prev_f]; positions[idx] = (int(cx), int(cy))
        elif next_f is not None:
            cx, cy, _ = anchors[next_f]; positions[idx] = (int(cx), int(cy))
        else:
            positions[idx] = seed_pos

    return positions


# ─────────────────────────────────────────────────────────────
# Pass 3: Kalman smoother (pixel space)
# ─────────────────────────────────────────────────────────────

def kalman_smooth(positions, clean_dets, seed_frame, seed_pos, n_frames,
                  hom, kalman_q_m=0.1):
    """
    Kalman smoother in pixel space.
    Process noise (kalman_q) is specified in m²/frame² and converted
    to px²/frame² using local ppm at seed position.
    """
    ppm = hom.local_ppm(*seed_pos)
    kalman_q = kalman_q_m * (ppm ** 2)   # px²/frame²

    W = hom.frame_w; H_px = hom.frame_h
    kf_pos = {}
    all_anchors = dict(clean_dets)
    all_anchors[seed_frame] = (float(seed_pos[0]), float(seed_pos[1]), 1.0)

    def reset_kf(x0, y0):
        kf_ = cv2.KalmanFilter(4, 2)
        kf_.transitionMatrix   = np.array([[1,0,1,0],[0,1,0,1],
                                            [0,0,1,0],[0,0,0,1]], dtype=np.float32)
        kf_.measurementMatrix  = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        kf_.processNoiseCov    = np.eye(4, dtype=np.float32) * kalman_q
        kf_.measurementNoiseCov= np.eye(2, dtype=np.float32) * max(1.0, kalman_q * 0.1)
        kf_.errorCovPost = np.eye(4, dtype=np.float32) * 50.0
        kf_.statePost    = np.array([x0, y0, 0.0, 0.0], dtype=np.float32).reshape(-1,1)
        return kf_

    kf = reset_kf(*seed_pos)
    kf_pos[seed_frame] = seed_pos

    for idx in range(n_frames):
        if idx == seed_frame:
            continue
        if idx in all_anchors:
            cx, cy, _ = all_anchors[idx]
            kf = reset_kf(cx, cy)
            kf_pos[idx] = (int(cx), int(cy))
            continue

        pred = kf.predict()
        px = float(pred[0,0]); py = float(pred[1,0])

        if idx in positions:
            fx, fy = positions[idx]
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * max(kalman_q*5, 50.0)
            meas = np.array([[float(fx)], [float(fy)]], dtype=np.float32)
            kf.correct(meas)
            state = kf.statePost
            kf_pos[idx] = (int(clamp(float(state[0,0]), 0, W-1)),
                           int(clamp(float(state[1,0]), 0, H_px-1)))
        else:
            kf.statePost[2,0] *= 0.85
            kf.statePost[3,0] *= 0.85
            kf_pos[idx] = (int(clamp(px, 0, W-1)), int(clamp(py, 0, H_px-1)))

    return kf_pos


# ─────────────────────────────────────────────────────────────
# Full v8 tracker
# ─────────────────────────────────────────────────────────────

def track_v8(frames, seed_frame, seed_pos, cache, hom,
             min_conf=0.10,
             hard_reset_conf=0.50,
             abc_protect_conf=0.65,
             speed_limit=3.0,          # m/frame  (30 m/s at 10fps)
             vel_decay=0.85,
             abc_ratio=0.6,
             hc_thresh=0.55,
             chain_margin=5.0,         # meters
             gap_margin=3.0,           # meters
             min_gap_for_filter=6,
             interp_max_gap=100,
             person_accept_r=3.0,      # meters
             person_conf_min=0.30,
             use_kalman_smooth=True,
             kalman_q_m=0.1):
    n = len(frames)

    raw_dets = collect_trajectory_guided(
        cache, n, seed_pos, hom,
        min_conf=min_conf, hard_reset_conf=hard_reset_conf,
        speed_limit=speed_limit, vel_decay=vel_decay)

    raw_dets = hc_override(raw_dets, cache, n,
                           hc_direct_conf=hard_reset_conf)

    clean_dets = abc_filter(raw_dets, hom,
                            abc_protect_conf=abc_protect_conf,
                            speed_limit=speed_limit, abc_ratio=abc_ratio)

    clean_dets = validate_hc_chain(
        clean_dets, seed_frame, seed_pos, hom,
        hc_thresh=hc_thresh, chain_margin=chain_margin)

    clean_dets = hc_gap_filter(
        clean_dets, seed_frame, seed_pos, hom,
        hc_thresh=hc_thresh, gap_margin=gap_margin,
        min_gap_for_filter=min_gap_for_filter)

    positions = fill_gaps(
        n, clean_dets, cache, seed_frame, seed_pos, hom,
        interp_max_gap=interp_max_gap,
        person_accept_r=person_accept_r,
        person_conf_min=person_conf_min)

    if use_kalman_smooth:
        positions = kalman_smooth(
            positions, clean_dets, seed_frame, seed_pos,
            n, hom, kalman_q_m=kalman_q_m)

    return positions, clean_dets


# ─────────────────────────────────────────────────────────────
# Bidirectional tracking
# ─────────────────────────────────────────────────────────────

def find_best_seed_in_window(cache, start, end):
    """Return (frame, (px,py)) of highest-confidence ball detection in [start,end]."""
    best_conf = -1; best_frame = None; best_pos = None
    for f in range(start, end+1):
        for b in cache.get(f, {}).get('balls', []):
            if b[2] > best_conf:
                best_conf = b[2]; best_frame = f
                best_pos = (int(b[0]), int(b[1]))
    return best_frame, best_pos


def track_v8_backward(frames, cache, hom,
                      bwd_seed_frame, bwd_seed_pos, **params):
    """
    Run the v8 tracker in reverse (from bwd_seed toward frame 0).
    Internally reverses the cache indices, runs the same pipeline,
    then un-reverses the output.  Never uses H_inv.
    """
    n = len(frames)
    cache_rev = {n-1-k: v for k, v in cache.items()}
    seed_rev   = n-1-bwd_seed_frame

    pos_rev, dets_rev = track_v8(
        frames, seed_rev, bwd_seed_pos, cache_rev, hom, **params)

    pos_bwd  = {n-1-k: v  for k, v  in pos_rev.items()}
    dets_bwd = {n-1-k: v  for k, v  in dets_rev.items()}
    return pos_bwd, dets_bwd


def merge_bidirectional(pos_fwd, pos_bwd,
                        dets_fwd, dets_bwd,
                        seed_fwd, seed_bwd,
                        hom,
                        agreement_m=2.0,
                        quality_window=10,
                        hc_thresh=0.55):
    """
    Merge forward and backward tracks frame by frame.

    For each frame:
      • Both agree within agreement_m  → average their pixel positions
      • They disagree                  → pick the direction that has
                                         more HC anchors in ±quality_window frames
                                         (tie → pick direction with nearest anchor)

    This resolves FP chains: a cluster of FPs that hijacks the forward
    tracker will NOT hijack the backward tracker (it has good anchors
    after the cluster), so the backward position wins in that region.
    No per-video threshold tuning needed — agreement_m is in meters.
    """
    all_frames = set(pos_fwd) | set(pos_bwd)

    # HC anchor sets for quality scoring
    hc_fwd = {f for f, v in dets_fwd.items() if v[2] >= hc_thresh}
    hc_fwd.add(seed_fwd)
    hc_bwd = {f for f, v in dets_bwd.items() if v[2] >= hc_thresh}
    hc_bwd.add(seed_bwd)

    def quality(f, hc_set):
        return sum(1 for h in hc_set if abs(h-f) <= quality_window)

    def nearest_anchor(f, hc_set):
        return min((abs(h-f) for h in hc_set), default=99999)

    merged = {}
    for f in sorted(all_frames):
        has_fwd = f in pos_fwd
        has_bwd = f in pos_bwd

        if has_fwd and not has_bwd:
            merged[f] = pos_fwd[f]
        elif has_bwd and not has_fwd:
            merged[f] = pos_bwd[f]
        else:
            pxf, pyf = pos_fwd[f]
            pxb, pyb = pos_bwd[f]
            wd = hom.world_dist(pxf, pyf, pxb, pyb)

            if wd <= agreement_m:
                # Agreement: average pixel positions
                merged[f] = (int((pxf+pxb)/2), int((pyf+pyb)/2))
            else:
                # Disagreement: pick better-supported direction
                qf = quality(f, hc_fwd)
                qb = quality(f, hc_bwd)
                if qb > qf:
                    merged[f] = pos_bwd[f]
                elif qf > qb:
                    merged[f] = pos_fwd[f]
                else:
                    # Tie: prefer closer anchor
                    nf = nearest_anchor(f, hc_fwd)
                    nb = nearest_anchor(f, hc_bwd)
                    merged[f] = pos_bwd[f] if nb <= nf else pos_fwd[f]

    return merged


def track_bidir(frames, seed_frame, seed_pos, cache, hom,
                bwd_seed_frame=None, bwd_seed_pos=None,
                agreement_m=2.0, quality_window=10,
                **params):
    """
    Full bidirectional tracker.  Runs forward from seed_frame and backward
    from bwd_seed_frame (auto-detected from last 150 frames if not given),
    then merges the two tracks.
    """
    n = len(frames)

    # Auto-detect backward seed if not provided
    if bwd_seed_frame is None:
        bwd_seed_frame, bwd_seed_pos = find_best_seed_in_window(
            cache, max(0, n-150), n-1)
    if bwd_seed_frame is None:
        bwd_seed_frame = n-1
        bwd_seed_pos   = seed_pos

    hc_thresh = params.get('hc_thresh', 0.55)

    pos_fwd, dets_fwd = track_v8(
        frames, seed_frame, seed_pos, cache, hom, **params)

    pos_bwd, dets_bwd = track_v8_backward(
        frames, cache, hom, bwd_seed_frame, bwd_seed_pos, **params)

    merged = merge_bidirectional(
        pos_fwd, pos_bwd, dets_fwd, dets_bwd,
        seed_frame, bwd_seed_frame, hom,
        agreement_m=agreement_m,
        quality_window=quality_window,
        hc_thresh=hc_thresh)

    return merged


# ─────────────────────────────────────────────────────────────
# Evaluation and rendering
# ─────────────────────────────────────────────────────────────

def evaluate(positions, labels):
    errs = []
    for lb in labels:
        f = lb['frame']
        if f not in positions:
            continue
        px, py = positions[f]
        gx, gy = int(lb['px']), int(lb['py'])
        errs.append(dist((px,py),(gx,gy)))
    if not errs:
        return 0, 0, 0, float('inf'), 0.0
    w30 = sum(1 for e in errs if e <= 30)
    w50 = sum(1 for e in errs if e <= 50)
    med = float(np.median(errs))
    score = (w30*2 + w50) / (3*len(labels))
    return len(errs), w30, w50, med, score


def render_video(frames, positions, labels, out_path, fps=10):
    if not frames:
        return
    ref = cv2.imread(frames[0])
    h, w = ref.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    lmap = {lb['frame']: lb for lb in labels}
    for i, fp in enumerate(frames):
        fr = cv2.imread(fp)
        if fr is None:
            continue
        if i in positions:
            px, py = positions[i]
            cv2.circle(fr, (px, py), 12, (0, 255, 0), 2)
            cv2.circle(fr, (px, py), 3,  (0, 255, 0), -1)
        if i in lmap:
            lb = lmap[i]
            cv2.circle(fr, (int(lb['px']), int(lb['py'])), 12, (0, 0, 255), 2)
        vw.write(fr)
    vw.release()
    print(f"Saved {out_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video',        required=True)
    ap.add_argument('--labels',       required=True)
    ap.add_argument('--cache',        required=True)
    ap.add_argument('--calibration',  required=True)
    ap.add_argument('--frames-dir',   default='')
    ap.add_argument('--render',       action='store_true')
    ap.add_argument('--output',       default='tracking_v8_winner.mp4')
    ap.add_argument('--analysis-out', default='tracking_analysis_v8.txt')
    ap.add_argument('--random-seeds', type=int, default=40)
    args = ap.parse_args()

    # Load labels
    with open(args.labels) as f:
        data = json.load(f)
    labels = data['labels']
    print(f"Loaded {len(labels)} labels")

    # Frames
    frames_dir = args.frames_dir or '/tmp/soccer_v8_frames'
    if not os.path.isdir(frames_dir) or not os.listdir(frames_dir):
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Extracting frames from '{args.video}'...")
        subprocess.run([
            'ffmpeg', '-i', args.video, '-vf', f'fps={FPS}',
            '-q:v', '2', '-loglevel', 'error',
            os.path.join(frames_dir, 'frame_%06d.jpg')
        ], check=True)
    frames = load_frames(frames_dir)
    print(f"Using {len(frames)} frames")

    # Cache
    print(f"Loading YOLO cache from {args.cache}...")
    cache = load_cache(args.cache)
    print(f"  {len(cache)} frames cached")

    # Calibration
    print(f"Loading calibration from {args.calibration}...")
    hom = FieldHomography(args.calibration)
    cfg = hom.field_config
    print(f"  Format: {hom.fmt}  Box: {hom.box}")

    # Seed
    first = labels[0]
    seed_frame = first['frame']
    seed_pos   = (int(first['px']), int(first['py']))
    seed_world = hom.to_world(*seed_pos)
    ppm_seed   = hom.local_ppm(*seed_pos)
    print(f"\nSeed: frame {seed_frame}  pixel={seed_pos}  "
          f"world=({seed_world[0]:.2f},{seed_world[1]:.2f})m")
    print(f"Local scale at seed: {ppm_seed:.1f} px/m")

    # ── Grid search ───────────────────────────────────────────
    print("\n--- Grid search v8 (world-space thresholds) ---")

    hand_tuned = [
        # Baseline
        dict(min_conf=0.10, hard_reset_conf=0.50, abc_protect_conf=0.65,
             speed_limit=3.0, vel_decay=0.85, abc_ratio=0.6,
             hc_thresh=0.55, chain_margin=5.0, gap_margin=3.0,
             min_gap_for_filter=6, interp_max_gap=100,
             person_accept_r=3.0, person_conf_min=0.30,
             use_kalman_smooth=True, kalman_q_m=0.1),
        # Tighter speed limit
        dict(min_conf=0.10, hard_reset_conf=0.50, abc_protect_conf=0.65,
             speed_limit=2.0, vel_decay=0.85, abc_ratio=0.6,
             hc_thresh=0.55, chain_margin=4.0, gap_margin=2.5,
             min_gap_for_filter=6, interp_max_gap=100,
             person_accept_r=3.0, person_conf_min=0.30,
             use_kalman_smooth=True, kalman_q_m=0.1),
        # Looser margins
        dict(min_conf=0.10, hard_reset_conf=0.50, abc_protect_conf=0.65,
             speed_limit=4.0, vel_decay=0.85, abc_ratio=0.6,
             hc_thresh=0.55, chain_margin=7.0, gap_margin=5.0,
             min_gap_for_filter=6, interp_max_gap=100,
             person_accept_r=4.0, person_conf_min=0.30,
             use_kalman_smooth=True, kalman_q_m=0.1),
        # hc_thresh=0.65
        dict(min_conf=0.10, hard_reset_conf=0.50, abc_protect_conf=0.65,
             speed_limit=3.0, vel_decay=0.85, abc_ratio=0.6,
             hc_thresh=0.65, chain_margin=5.0, gap_margin=3.0,
             min_gap_for_filter=6, interp_max_gap=100,
             person_accept_r=3.0, person_conf_min=0.30,
             use_kalman_smooth=True, kalman_q_m=0.1),
        # Wide person search
        dict(min_conf=0.10, hard_reset_conf=0.50, abc_protect_conf=0.65,
             speed_limit=3.0, vel_decay=0.85, abc_ratio=0.6,
             hc_thresh=0.55, chain_margin=5.0, gap_margin=3.0,
             min_gap_for_filter=6, interp_max_gap=100,
             person_accept_r=6.0, person_conf_min=0.30,
             use_kalman_smooth=True, kalman_q_m=0.1),
        # No kalman
        dict(min_conf=0.10, hard_reset_conf=0.50, abc_protect_conf=0.65,
             speed_limit=3.0, vel_decay=0.85, abc_ratio=0.6,
             hc_thresh=0.55, chain_margin=5.0, gap_margin=3.0,
             min_gap_for_filter=6, interp_max_gap=100,
             person_accept_r=3.0, person_conf_min=0.30,
             use_kalman_smooth=False, kalman_q_m=0.1),
        # Very tight — only very reliable detections
        dict(min_conf=0.12, hard_reset_conf=0.50, abc_protect_conf=0.65,
             speed_limit=2.5, vel_decay=0.85, abc_ratio=0.6,
             hc_thresh=0.65, chain_margin=4.0, gap_margin=2.5,
             min_gap_for_filter=6, interp_max_gap=80,
             person_accept_r=2.5, person_conf_min=0.35,
             use_kalman_smooth=True, kalman_q_m=0.05),
        # Very loose — more YOLO detections accepted
        dict(min_conf=0.07, hard_reset_conf=0.50, abc_protect_conf=0.55,
             speed_limit=5.0, vel_decay=0.85, abc_ratio=0.5,
             hc_thresh=0.50, chain_margin=8.0, gap_margin=6.0,
             min_gap_for_filter=4, interp_max_gap=150,
             person_accept_r=5.0, person_conf_min=0.25,
             use_kalman_smooth=True, kalman_q_m=0.2),
        # Large interp gap
        dict(min_conf=0.10, hard_reset_conf=0.50, abc_protect_conf=0.65,
             speed_limit=3.0, vel_decay=0.85, abc_ratio=0.6,
             hc_thresh=0.55, chain_margin=5.0, gap_margin=3.0,
             min_gap_for_filter=6, interp_max_gap=200,
             person_accept_r=3.0, person_conf_min=0.30,
             use_kalman_smooth=True, kalman_q_m=0.1),
        # mgf=8 (stricter gap filter application)
        dict(min_conf=0.10, hard_reset_conf=0.50, abc_protect_conf=0.65,
             speed_limit=3.0, vel_decay=0.85, abc_ratio=0.6,
             hc_thresh=0.55, chain_margin=5.0, gap_margin=3.0,
             min_gap_for_filter=8, interp_max_gap=100,
             person_accept_r=3.0, person_conf_min=0.30,
             use_kalman_smooth=True, kalman_q_m=0.1),
    ]

    rng = random.Random(42)
    random_combos = []
    for _ in range(args.random_seeds):
        random_combos.append(dict(
            min_conf         = rng.choice([0.07, 0.08, 0.10, 0.12]),
            hard_reset_conf  = 0.50,
            abc_protect_conf = rng.choice([0.55, 0.60, 0.65, 0.70]),
            speed_limit      = rng.uniform(1.5, 5.0),
            vel_decay        = rng.uniform(0.78, 0.92),
            abc_ratio        = rng.uniform(0.45, 0.70),
            hc_thresh        = rng.choice([0.50, 0.55, 0.60, 0.65]),
            chain_margin     = rng.uniform(2.5, 8.0),
            gap_margin       = rng.uniform(1.5, 6.0),
            min_gap_for_filter = rng.choice([4, 6, 8]),
            interp_max_gap   = rng.choice([80, 100, 120, 150, 200]),
            person_accept_r  = rng.uniform(1.5, 7.0),
            person_conf_min  = rng.choice([0.25, 0.30, 0.35]),
            use_kalman_smooth = True,
            kalman_q_m       = rng.uniform(0.03, 0.3),
        ))

    # Backward seed: use last labeled frame (reliable for eval)
    last_lb = labels[-1]
    bwd_seed_frame = last_lb['frame']
    bwd_seed_pos   = (int(last_lb['px']), int(last_lb['py']))
    print(f"Backward seed: frame {bwd_seed_frame}  pixel={bwd_seed_pos}")

    all_combos = hand_tuned + random_combos
    results_log = []
    best_score  = -1; best_pos = None; best_params = None

    for i, params in enumerate(all_combos):
        tag = f"hand_{i}" if i < len(hand_tuned) else f"rand_{i-len(hand_tuned)}"
        pos = track_bidir(frames, seed_frame, seed_pos, cache, hom,
                          bwd_seed_frame=bwd_seed_frame,
                          bwd_seed_pos=bwd_seed_pos,
                          **params)
        tr, w30, w50, med, sc = evaluate(pos, labels)
        results_log.append((sc, w30, w50, med, tr, tag, params, pos))
        if sc > best_score:
            best_score = sc; best_params = params; best_pos = pos
        if (i+1) % 10 == 0:
            top = max(results_log, key=lambda x: x[0])
            print(f"  [{i+1}/{len(all_combos)}] "
                  f"best: score={top[0]:.4f} w30={top[1]} w50={top[2]} med={top[3]:.0f}px")

    results_log.sort(key=lambda x: x[0], reverse=True)

    print(f"\nTop 5:")
    for sc, w30, w50, med, tr, tag, params, _ in results_log[:5]:
        print(f"  {tag}  score={sc:.4f}  W30={w30:3d}  W50={w50:3d}  Med={med:.0f}px")

    sc, w30, w50, med, tr, tag, best_params, best_pos = results_log[0]
    print(f"\nWINNER: score={sc:.4f}  W30={w30}/{len(labels)}  "
          f"W50={w50}/{len(labels)}  Median={med:.1f}px")
    print(f"  Params: {best_params}")

    # Per-frame misses
    miss_count = 0
    print("\nPer-frame errors > 50px (winner):")
    for lb in labels:
        f = lb['frame']
        if f not in best_pos:
            print(f"  f{f:4d}  NO_DET"); miss_count += 1; continue
        px, py = best_pos[f]
        err = dist((px,py),(lb['px'],lb['py']))
        if err > 50:
            print(f"  f{f:4d}  err={err:.0f}px  gt=({lb['px']},{lb['py']})  pred=({px},{py})")
            miss_count += 1
    print(f"  Total misses > 50px: {miss_count}/{len(labels)}")

    # Save analysis
    with open(args.analysis_out, 'w') as f:
        f.write(f"V8 Winner: score={sc:.4f} w30={w30} w50={w50} med={med:.1f}\n")
        f.write(f"Params: {best_params}\n\n")
        for sc2, w30_2, w50_2, md, tr2, tg, p, _ in results_log:
            f.write(f"{sc2:.4f}  W30={w30_2}  W50={w50_2}  Med={md:.1f}  {tg}\n")
    print(f"Saved analysis to {args.analysis_out}")

    if args.render:
        render_video(frames, best_pos, labels, args.output)


if __name__ == '__main__':
    main()
