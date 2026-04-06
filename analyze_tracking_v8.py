#!/usr/bin/env python3
"""
Ball tracking v8 — homography-based world-coordinate tracking.

Key change from v7h:
  All pixel-space tracking parameters (speed_limit, chain_margin, gap_margin,
  person_accept_r) are replaced by world-space parameters in METERS.
  This makes the tracker camera-angle agnostic.

Pipeline:
  1. Load field calibration (computed by calibrate_field.py)
  2. Transform YOLO cache: pixel → world coords (meters on pitch)
  3. Run full v7h tracking pipeline in world space
  4. Transform results back to pixels for evaluation and display

World coordinate system (defined by calibrate_field.py):
  Origin = center of center circle
  +X     = rightward along field width (facing near goal)
  +Y     = toward near goal
  Units  = meters

World-space physical limits:
  Ball max speed  ≈ 30 m/s = 3 m/frame @ 10fps  → speed_limit = 5 m/frame
  Chain/gap error ≈ ± 3-5m tolerance             → margins in meters

Usage:
  # First, calibrate once per camera setup:
  .venv/bin/python3 calibrate_field.py --video "test 1.mp4" \\
      --format 11v11 --output calibration_test1.json

  # Then run tracker:
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
# Homography wrapper
# ─────────────────────────────────────────────────────────────

class FieldHomography:
    def __init__(self, cal_path):
        with open(cal_path) as f:
            cal = json.load(f)
        self.H     = np.array(cal['H'],     dtype=np.float64)
        self.H_inv = np.array(cal['H_inv'], dtype=np.float64)
        self.frame_w = cal['frame_w']
        self.frame_h = cal['frame_h']
        self.field_config = cal['field_config']
        self.fmt = cal['format']

    def to_world(self, px, py):
        """Transform a single pixel point → world (meters)."""
        pt = np.array([[[float(px), float(py)]]], dtype=np.float64)
        wp = cv2.perspectiveTransform(pt, self.H)[0][0]
        return float(wp[0]), float(wp[1])

    def to_pixel(self, wx, wy):
        """Transform a world point (meters) → pixel."""
        pt = np.array([[[float(wx), float(wy)]]], dtype=np.float64)
        pp = cv2.perspectiveTransform(pt, self.H_inv)[0][0]
        return int(clamp(pp[0], 0, self.frame_w - 1)), \
               int(clamp(pp[1], 0, self.frame_h - 1))

    def transform_cache_to_world(self, cache_px):
        """
        Return a new cache where ball positions are in world coords (meters)
        and person positions use the BOTTOM-CENTER of each bounding box
        (feet on ground plane) also transformed to world coords.
        """
        cache_world = {}
        for idx, entry in cache_px.items():
            balls_w = []
            for bx, by, cf in entry.get('balls', []):
                wx, wy = self.to_world(bx, by)
                balls_w.append([wx, wy, cf])

            persons_w = []
            for cx, cy, cf, x1, y1, x2, y2 in entry.get('persons', []):
                # Use bottom-center of bbox as feet position on ground
                feet_px = (x1 + x2) / 2
                feet_py = y2
                wx, wy = self.to_world(feet_px, feet_py)
                # Store world coords in same structure; set bbox to zeros
                persons_w.append([wx, wy, cf, 0.0, 0.0, 0.0, 0.0])

            cache_world[idx] = {'balls': balls_w, 'persons': persons_w}
        return cache_world


# ─────────────────────────────────────────────────────────────
# Tracker pipeline (world-space version of v7h)
# All pixel distances replaced with meters.
# ─────────────────────────────────────────────────────────────

def collect_trajectory_guided(cache, n_frames, seed_pos,
                               min_conf=0.10,
                               hard_reset_conf=0.40,
                               speed_limit=5.0,      # m/frame
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

        close = [(b, dist(b[:2], (pred_x, pred_y)) / dt)
                 for b in balls
                 if dist(b[:2], (pred_x, pred_y)) / dt <= speed_limit]
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


def abc_filter(dets, abc_protect_conf=0.65, speed_limit=5.0, abc_ratio=0.6):
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
            spd_ab = dist((bx,by),(ax,ay)) / dt_ab
            if spd_ab < speed_limit * 0.3:
                continue
            if dist((cx2,cy2),(ax,ay)) < abc_ratio * dist((cx2,cy2),(bx,by)):
                to_remove.add(idx)
        for idx in to_remove:
            del clean[idx]
        if not to_remove:
            break
    return clean


def validate_hc_chain(clean_dets, seed_frame, seed_pos,
                      hc_thresh=0.55, chain_margin=5.0):   # 5m
    filtered = dict(clean_dets)
    while True:
        hc_anchors = {f: v for f, v in filtered.items() if v[2] >= hc_thresh}
        hc_anchors[seed_frame] = (float(seed_pos[0]), float(seed_pos[1]), 1.0)
        hc_frames = sorted(hc_anchors)
        if len(hc_frames) < 3:
            break
        worst_frame = None
        worst_dev = chain_margin
        for i in range(1, len(hc_frames)-1):
            fa = hc_frames[i-1]; fb = hc_frames[i]; fc = hc_frames[i+1]
            ax, ay, _ = hc_anchors[fa]
            bx, by, _ = hc_anchors[fb]
            cx, cy, _ = hc_anchors[fc]
            t = (fb - fa) / (fc - fa)
            ex = ax + t*(cx-ax); ey = ay + t*(cy-ay)
            dev = dist((bx,by),(ex,ey))
            if dev > worst_dev:
                worst_dev = dev; worst_frame = fb
        if worst_frame is None:
            break
        del filtered[worst_frame]
    return filtered


def hc_gap_filter(clean_dets, seed_frame, seed_pos,
                  hc_thresh=0.55, gap_margin=4.0,    # 4m
                  min_gap_for_filter=8):
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
                ex = ax + t*(bx-ax); ey = ay + t*(by-ay)
                if dist((cx,cy),(ex,ey)) > gap_margin:
                    to_remove.append(f)
            for f in to_remove:
                del filtered[f]; changed = True
    return filtered


def fill_gaps(n_frames, clean_dets, cache_world, seed_frame, seed_pos,
              interp_max_gap=100,
              person_accept_r=3.0,      # 3m radius
              person_conf_min=0.30,
              field_bounds=None):        # (x_min, x_max, y_min, y_max) in meters
    """Gap filling in world coordinates.
    Person estimation uses feet (bottom of bbox) already transformed to world coords.
    """
    if field_bounds is None:
        field_bounds = (-60, 60, -40, 90)   # generous bounds in meters

    x_min, x_max, y_min, y_max = field_bounds

    positions = {}
    anchors = dict(clean_dets)
    anchors[seed_frame] = (float(seed_pos[0]), float(seed_pos[1]), 1.0)
    anchor_frames = sorted(anchors)
    running_ref = (float(seed_pos[0]), float(seed_pos[1]))

    for idx in range(n_frames):
        if idx in anchors:
            wx, wy, _ = anchors[idx]
            positions[idx] = (wx, wy)
            running_ref = (wx, wy)
            continue

        prev_f = next_f = None
        for f in reversed([f for f in anchor_frames if f < idx]):
            prev_f = f; break
        for f in [f for f in anchor_frames if f > idx]:
            next_f = f; break

        if prev_f is not None and next_f is not None:
            gap = next_f - prev_f
            if gap <= interp_max_gap:
                t = (idx - prev_f) / gap
                wx_a, wy_a, _ = anchors[prev_f]
                wx_b, wy_b, _ = anchors[next_f]
                wx = wx_a + t*(wx_b - wx_a)
                wy = wy_a + t*(wy_b - wy_a)
                positions[idx] = (wx, wy)
                running_ref = (wx, wy)
                continue

        # Person estimation (world-space feet positions)
        persons = cache_world.get(idx, {}).get('persons', [])
        best_person = None; best_d = 1e9
        for wx, wy, pcf, *_ in persons:
            if pcf < person_conf_min:
                continue
            d = dist((wx, wy), running_ref)
            if d < person_accept_r and d < best_d:
                best_d = d; best_person = (wx, wy)

        if best_person is not None:
            fx, fy = best_person
            fx = clamp(fx, x_min, x_max)
            fy = clamp(fy, y_min, y_max)
            positions[idx] = (fx, fy)
            running_ref = (fx, fy)
            continue

        if prev_f is not None:
            wx, wy, _ = anchors[prev_f]; positions[idx] = (wx, wy)
        elif next_f is not None:
            wx, wy, _ = anchors[next_f]; positions[idx] = (wx, wy)
        else:
            positions[idx] = seed_pos

    return positions


def kalman_smooth_world(positions, clean_dets, seed_frame, seed_pos,
                        n_frames, kalman_q=0.1):
    """Kalman smoother in world coordinates. q is in m²."""
    kf_pos = {}
    all_anchors = dict(clean_dets)
    all_anchors[seed_frame] = (float(seed_pos[0]), float(seed_pos[1]), 1.0)

    def reset_kf(x0, y0):
        kf_ = cv2.KalmanFilter(4, 2)
        kf_.transitionMatrix  = np.array([[1,0,1,0],[0,1,0,1],
                                           [0,0,1,0],[0,0,0,1]], dtype=np.float32)
        kf_.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        kf_.processNoiseCov    = np.eye(4, dtype=np.float32) * kalman_q
        kf_.measurementNoiseCov= np.eye(2, dtype=np.float32) * 0.05  # 5cm anchor noise
        kf_.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        kf_.statePost    = np.array([x0, y0, 0.0, 0.0], dtype=np.float32).reshape(-1,1)
        return kf_

    kf = reset_kf(*seed_pos)
    kf_pos[seed_frame] = seed_pos

    for idx in range(n_frames):
        if idx == seed_frame:
            continue
        if idx in all_anchors:
            wx, wy, _ = all_anchors[idx]
            kf = reset_kf(wx, wy)
            kf_pos[idx] = (wx, wy)
            continue

        pred = kf.predict()
        px = float(pred[0,0]); py = float(pred[1,0])

        if idx in positions:
            fx, fy = positions[idx]
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5  # 0.5m fill noise
            meas = np.array([[float(fx)], [float(fy)]], dtype=np.float32)
            kf.correct(meas)
            state = kf.statePost
            kf_pos[idx] = (float(state[0,0]), float(state[1,0]))
        else:
            kf.statePost[2,0] *= 0.85
            kf.statePost[3,0] *= 0.85
            kf_pos[idx] = (px, py)

    return kf_pos


# ─────────────────────────────────────────────────────────────
# Full v8 tracker
# ─────────────────────────────────────────────────────────────

def track_v8(frames, seed_frame, seed_pos_px, cache_px, homography,
             min_conf=0.10,
             hard_reset_conf=0.50,
             abc_protect_conf=0.65,
             speed_limit=5.0,          # m/frame  (≈50 m/s at 10fps — very generous)
             vel_decay=0.85,
             abc_ratio=0.6,
             hc_thresh=0.55,
             chain_margin=5.0,         # meters
             gap_margin=4.0,           # meters
             min_gap_for_filter=6,
             interp_max_gap=100,
             person_accept_r=3.0,      # meters
             person_conf_min=0.30,
             use_kalman_smooth=True,
             kalman_q=0.1):
    """
    Track ball in world coordinates using field homography.
    Returns positions as pixel coordinates for evaluation and display.
    """
    n = len(frames)

    # Convert cache and seed to world coordinates
    cache_world = homography.transform_cache_to_world(cache_px)
    seed_world  = homography.to_world(*seed_pos_px)

    raw_dets = collect_trajectory_guided(
        cache_world, n, seed_world,
        min_conf=min_conf, hard_reset_conf=hard_reset_conf,
        speed_limit=speed_limit, vel_decay=vel_decay)

    raw_dets = hc_override(raw_dets, cache_world, n,
                           hc_direct_conf=hard_reset_conf)

    clean_dets = abc_filter(raw_dets, abc_protect_conf=abc_protect_conf,
                            speed_limit=speed_limit, abc_ratio=abc_ratio)

    clean_dets = validate_hc_chain(
        clean_dets, seed_frame, seed_world,
        hc_thresh=hc_thresh, chain_margin=chain_margin)

    clean_dets = hc_gap_filter(
        clean_dets, seed_frame, seed_world,
        hc_thresh=hc_thresh, gap_margin=gap_margin,
        min_gap_for_filter=min_gap_for_filter)

    positions_world = fill_gaps(
        n, clean_dets, cache_world, seed_frame, seed_world,
        interp_max_gap=interp_max_gap,
        person_accept_r=person_accept_r,
        person_conf_min=person_conf_min)

    if use_kalman_smooth:
        positions_world = kalman_smooth_world(
            positions_world, clean_dets, seed_frame, seed_world,
            n, kalman_q=kalman_q)

    # Convert world positions → pixels
    positions_px = {}
    for f, (wx, wy) in positions_world.items():
        positions_px[f] = homography.to_pixel(wx, wy)

    return positions_px


# ─────────────────────────────────────────────────────────────
# Evaluation (pixel-space, same as before)
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


# ─────────────────────────────────────────────────────────────
# Renderer
# ─────────────────────────────────────────────────────────────

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
    ap.add_argument('--cache',        required=True,
                    help='YOLO cache JSON for this video')
    ap.add_argument('--calibration',  required=True,
                    help='Field calibration JSON from calibrate_field.py')
    ap.add_argument('--frames-dir',   default='')
    ap.add_argument('--render',       action='store_true')
    ap.add_argument('--output',       default='tracking_v8_winner.mp4')
    ap.add_argument('--analysis-out', default='tracking_analysis_v8.txt')
    ap.add_argument('--random-seeds', type=int, default=30)
    args = ap.parse_args()

    # Load labels
    with open(args.labels) as f:
        data = json.load(f)
    labels = data['labels']
    lmap   = {lb['frame']: lb for lb in labels}
    print(f"Loaded {len(labels)} labels")

    # Extract / load frames
    frames_dir = args.frames_dir or '/tmp/soccer_v8_frames'
    if not os.path.isdir(frames_dir) or not os.listdir(frames_dir):
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Extracting frames from '{args.video}'...")
        subprocess.run([
            'ffmpeg', '-i', args.video,
            '-vf', f'fps={FPS}',
            '-q:v', '2', '-loglevel', 'error',
            os.path.join(frames_dir, 'frame_%06d.jpg')
        ], check=True)
    frames = load_frames(frames_dir)
    print(f"Using {len(frames)} frames from {frames_dir}")

    # Load YOLO cache
    print(f"Loading YOLO cache from {args.cache}...")
    cache_px = load_cache(args.cache)
    print(f"  {len(cache_px)} frames cached")

    # Load calibration / homography
    print(f"Loading field calibration from {args.calibration}...")
    hom = FieldHomography(args.calibration)
    print(f"  Format: {hom.fmt}")
    cfg = hom.field_config
    print(f"  Center circle r={cfg['center_circle_r']}m  "
          f"Penalty area={cfg['penalty_area_w']}×{cfg['penalty_area_d']}m")

    # Seed: first label
    first = labels[0]
    seed_frame = first['frame']
    seed_pos   = (int(first['px']), int(first['py']))
    seed_world = hom.to_world(*seed_pos)
    print(f"\nSeed: frame {seed_frame}  pixel={seed_pos}  "
          f"world=({seed_world[0]:.2f},{seed_world[1]:.2f})m")

    # ── Baseline run ──────────────────────────────────────────
    print("\n--- Baseline v8 (default params) ---")
    pos = track_v8(frames, seed_frame, seed_pos, cache_px, hom)
    tr, w30, w50, med, sc = evaluate(pos, labels)
    print(f"  W30={w30}/{len(labels)}  W50={w50}/{len(labels)}  "
          f"Median={med:.1f}px  Score={sc:.4f}")

    # ── Grid search ───────────────────────────────────────────
    print("\n--- Grid search v8 ---")

    hand_tuned = [
        # Baseline (default)
        dict(min_conf=0.10, hard_reset_conf=0.50, abc_protect_conf=0.65,
             speed_limit=5.0, vel_decay=0.85, abc_ratio=0.6,
             hc_thresh=0.55, chain_margin=5.0, gap_margin=4.0,
             min_gap_for_filter=6, interp_max_gap=100,
             person_accept_r=3.0, person_conf_min=0.30,
             use_kalman_smooth=True, kalman_q=0.1),
        # Tighter speed limit (3 m/frame ≈ 30m/s)
        dict(min_conf=0.10, hard_reset_conf=0.50, abc_protect_conf=0.65,
             speed_limit=3.0, vel_decay=0.85, abc_ratio=0.6,
             hc_thresh=0.55, chain_margin=4.0, gap_margin=3.0,
             min_gap_for_filter=6, interp_max_gap=100,
             person_accept_r=3.0, person_conf_min=0.30,
             use_kalman_smooth=True, kalman_q=0.1),
        # Looser chain/gap margins
        dict(min_conf=0.10, hard_reset_conf=0.50, abc_protect_conf=0.65,
             speed_limit=5.0, vel_decay=0.85, abc_ratio=0.6,
             hc_thresh=0.55, chain_margin=7.0, gap_margin=6.0,
             min_gap_for_filter=6, interp_max_gap=100,
             person_accept_r=3.0, person_conf_min=0.30,
             use_kalman_smooth=True, kalman_q=0.1),
        # hc_thresh=0.65 (more conservative HC anchors)
        dict(min_conf=0.10, hard_reset_conf=0.50, abc_protect_conf=0.65,
             speed_limit=5.0, vel_decay=0.85, abc_ratio=0.6,
             hc_thresh=0.65, chain_margin=5.0, gap_margin=4.0,
             min_gap_for_filter=6, interp_max_gap=100,
             person_accept_r=3.0, person_conf_min=0.30,
             use_kalman_smooth=True, kalman_q=0.1),
        # Wider person search radius
        dict(min_conf=0.10, hard_reset_conf=0.50, abc_protect_conf=0.65,
             speed_limit=5.0, vel_decay=0.85, abc_ratio=0.6,
             hc_thresh=0.55, chain_margin=5.0, gap_margin=4.0,
             min_gap_for_filter=6, interp_max_gap=100,
             person_accept_r=5.0, person_conf_min=0.30,
             use_kalman_smooth=True, kalman_q=0.1),
        # No kalman smooth
        dict(min_conf=0.10, hard_reset_conf=0.50, abc_protect_conf=0.65,
             speed_limit=5.0, vel_decay=0.85, abc_ratio=0.6,
             hc_thresh=0.55, chain_margin=5.0, gap_margin=4.0,
             min_gap_for_filter=6, interp_max_gap=100,
             person_accept_r=3.0, person_conf_min=0.30,
             use_kalman_smooth=False, kalman_q=0.1),
        # min_conf=0.12 (same as v7h)
        dict(min_conf=0.12, hard_reset_conf=0.50, abc_protect_conf=0.65,
             speed_limit=5.0, vel_decay=0.85, abc_ratio=0.6,
             hc_thresh=0.65, chain_margin=5.0, gap_margin=4.0,
             min_gap_for_filter=6, interp_max_gap=100,
             person_accept_r=3.0, person_conf_min=0.30,
             use_kalman_smooth=True, kalman_q=0.1),
        # Large interp_max_gap (more interpolation)
        dict(min_conf=0.10, hard_reset_conf=0.50, abc_protect_conf=0.65,
             speed_limit=5.0, vel_decay=0.85, abc_ratio=0.6,
             hc_thresh=0.55, chain_margin=5.0, gap_margin=4.0,
             min_gap_for_filter=6, interp_max_gap=200,
             person_accept_r=3.0, person_conf_min=0.30,
             use_kalman_smooth=True, kalman_q=0.1),
    ]

    rng = random.Random(42)
    random_combos = []
    for _ in range(args.random_seeds):
        random_combos.append(dict(
            min_conf      = rng.choice([0.08, 0.10, 0.12]),
            hard_reset_conf = 0.50,
            abc_protect_conf = rng.choice([0.55, 0.60, 0.65, 0.70]),
            speed_limit   = rng.uniform(3.0, 7.0),
            vel_decay     = rng.uniform(0.80, 0.92),
            abc_ratio     = rng.uniform(0.5, 0.7),
            hc_thresh     = rng.choice([0.50, 0.55, 0.60, 0.65]),
            chain_margin  = rng.uniform(3.0, 8.0),
            gap_margin    = rng.uniform(2.5, 6.0),
            min_gap_for_filter = rng.choice([4, 6, 8]),
            interp_max_gap = rng.choice([80, 100, 120, 200]),
            person_accept_r = rng.uniform(2.0, 6.0),
            person_conf_min = rng.choice([0.25, 0.30, 0.35]),
            use_kalman_smooth = True,
            kalman_q      = rng.uniform(0.05, 0.3),
        ))

    all_combos = hand_tuned + random_combos
    results_log = []
    best_score = -1; best_pos = None; best_params = None

    for i, params in enumerate(all_combos):
        tag = f"hand_{i}" if i < len(hand_tuned) else f"rand_{i-len(hand_tuned)}"
        pos = track_v8(frames, seed_frame, seed_pos, cache_px, hom, **params)
        tr2, w30, w50, med, sc = evaluate(pos, labels)
        results_log.append((sc, w30, w50, med, tr2, tag, params, pos))
        if sc > best_score:
            best_score = sc; best_params = params; best_pos = pos
        if (i+1) % 10 == 0:
            top = max(results_log, key=lambda x: x[0])
            print(f"  [{i+1}/{len(all_combos)}] "
                  f"best: score={top[0]:.4f} w30={top[1]} w50={top[2]} med={top[3]:.0f}px")

    results_log.sort(key=lambda x: x[0], reverse=True)

    print(f"\nTop 5 results:")
    for sc, w30, w50, med, tr2, tag, params, _ in results_log[:5]:
        print(f"  {tag}  score={sc:.4f}  W30={w30:3d}  W50={w50:3d}  Med={med:.0f}px")

    sc, w30, w50, med, tr2, tag, best_params, best_pos = results_log[0]
    print(f"\nWINNER: {best_params}")
    print(f"  Score={sc:.4f}  W30={w30}/{len(labels)}  "
          f"W50={w50}/{len(labels)}  Median={med:.1f}px")

    # Per-frame detail
    print("\nPer-frame errors > 50px (winner):")
    miss_count = 0
    for lb in labels:
        f = lb['frame']
        if f not in best_pos:
            print(f"  f{f:4d}  NO_DET")
            miss_count += 1
            continue
        px, py = best_pos[f]
        err = dist((px,py),(lb['px'],lb['py']))
        if err > 50:
            print(f"  f{f:4d}  err={err:.0f}px  gt=({lb['px']},{lb['py']})  pred=({px},{py})")
            miss_count += 1
    print(f"  Total misses > 50px: {miss_count}")

    # Save analysis
    with open(args.analysis_out, 'w') as f:
        f.write(f"V8 Winner: score={sc:.4f} w30={w30} w50={w50} med={med:.1f}\n")
        f.write(f"Params: {best_params}\n\n")
        for sc2, w30_2, w50_2, md, tr3, tg, p, _ in results_log:
            f.write(f"{sc2:.4f}  W30={w30_2}  W50={w50_2}  Med={md:.1f}  {tg}\n")
    print(f"Saved analysis to {args.analysis_out}")

    # Render
    if args.render:
        render_video(frames, best_pos, labels, args.output)


if __name__ == '__main__':
    main()
