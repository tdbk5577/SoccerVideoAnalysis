#!/usr/bin/env python3
"""
Ball tracking v7 — Two-pass YOLO@1920 + gap-filling.

Architecture (fundamentally different from v6):
  PASS 1 — Collect: run YOLOv8x@1920 on ALL frames, gather raw ball detections.
            Clean outliers: reject a detection whose speed from nearest neighbor
            exceeds hard_speed_limit AND whose confidence is below hard_reset_conf.
  PASS 2 — Fill: for every frame, get best position estimate:
            (a) If clean YOLO detection in this frame → use it directly.
            (b) Gap between two clean detections, gap <= interp_max_gap → linear interp.
            (c) No detection near (or large gap) → person-foot estimation:
                find person nearest to last known ball, ball ≈ (person_cx, person_cy - y_off).
            (d) No person → use last clean position.
  Optional PASS 3 — Smooth: optional light Gaussian smoothing of final trajectory.

Key v6 improvements:
  - imgsz=1920 → more detections (204/253 at 1920 vs ~107 W50 at 1280).
  - No Kalman divergence: only Kalman-smooth within segments between clean anchors.
  - Hard reset: high-conf detection always accepted regardless of distance.
  - Person estimation updates BOTH x and y (v6 only updated y).
  - Velocity check only for low-conf detections (high-conf always accepted).

Usage:
  .venv/bin/python3 analyze_tracking_v7.py --video Test.mp4 \\
      --labels ball_labels.json --frames-dir /tmp/soccer_v2_frames --render
"""

import argparse
import json
import os
import random
import time

import cv2
import numpy as np

FPS  = 10.0
W, H = 1920, 1080

YOLO_CACHE_FILE = 'yolo_cache_v7.json'
BALL_CLASS   = 32
PERSON_CLASS = 0


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


# ─────────────────────────────────────────────────────────────
# YOLO inference + caching
# ─────────────────────────────────────────────────────────────

def run_yolo_all_frames(frames, model_name='yolov8x.pt', imgsz=1920, conf=0.05):
    from ultralytics import YOLO
    model = YOLO(model_name)
    cache = {}
    n = len(frames)
    t0 = time.time()
    for i, fp in enumerate(frames):
        fr = cv2.imread(fp)
        if fr is None:
            cache[i] = {'balls': [], 'persons': []}
            continue
        r = model(fr, verbose=False, conf=conf, imgsz=imgsz)[0]
        balls, persons = [], []
        for j, cls in enumerate(r.boxes.cls):
            cls_id = int(cls)
            box = r.boxes.xyxy[j].cpu().numpy()
            cf  = float(r.boxes.conf[j])
            cx  = float((box[0]+box[2])/2)
            cy  = float((box[1]+box[3])/2)
            if cls_id == BALL_CLASS:
                balls.append([cx, cy, cf])
            elif cls_id == PERSON_CLASS:
                persons.append([cx, cy, cf,
                                float(box[0]), float(box[1]),
                                float(box[2]), float(box[3])])
        cache[i] = {'balls': balls, 'persons': persons}
        if (i+1) % 100 == 0:
            elapsed = time.time()-t0
            eta = elapsed/(i+1)*(n-i-1)
            print(f"    YOLO {i+1}/{n}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")
    return cache


def save_cache(cache, path):
    with open(path, 'w') as f:
        json.dump({str(k): v for k, v in cache.items()}, f)
    print(f"Saved YOLO cache to {path}")


def load_cache(path):
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ─────────────────────────────────────────────────────────────
# Pass 1: collect & clean YOLO detections
# ─────────────────────────────────────────────────────────────

def collect_detections(cache, n_frames,
                        min_conf=0.10,
                        hard_reset_conf=0.40,
                        speed_limit=250.0):
    """
    For each frame, pick best YOLO ball detection.
    Returns:
      raw_dets : {frame_idx: (cx, cy, conf)}  — best detection per frame
      clean_dets: {frame_idx: (cx, cy, conf)} — after outlier removal
    """
    # Best detection per frame
    raw_dets = {}
    for idx in range(n_frames):
        balls = cache.get(idx, {}).get('balls', [])
        if not balls:
            continue
        # Pick highest confidence
        best = max(balls, key=lambda b: b[2])
        if best[2] >= min_conf:
            raw_dets[idx] = tuple(best)

    # Outlier removal: two-pass trajectory consistency
    # A detection is an outlier if speed to BOTH neighbors > speed_limit
    # AND confidence < hard_reset_conf.
    frames_list = sorted(raw_dets)
    clean_dets = dict(raw_dets)  # start with all, then remove outliers

    for pass_num in range(3):
        clean_frames = sorted(clean_dets)
        to_remove = set()
        for i, idx in enumerate(clean_frames):
            cx, cy, cf = clean_dets[idx]
            if cf >= hard_reset_conf:
                continue  # Never remove high-conf detections
            # Speed to previous clean detection
            prev_speed = 1e9
            if i > 0:
                pidx = clean_frames[i-1]
                pcx, pcy, _ = clean_dets[pidx]
                dt = max(idx - pidx, 1)
                prev_speed = dist((cx,cy),(pcx,pcy)) / dt
            # Speed to next clean detection
            next_speed = 1e9
            if i < len(clean_frames)-1:
                nidx = clean_frames[i+1]
                ncx, ncy, _ = clean_dets[nidx]
                dt = max(nidx - idx, 1)
                next_speed = dist((cx,cy),(ncx,ncy)) / dt
            # Remove if fast relative to both neighbors (true outlier)
            if prev_speed > speed_limit and next_speed > speed_limit:
                to_remove.add(idx)
        for idx in to_remove:
            del clean_dets[idx]
        if not to_remove:
            break

    return raw_dets, clean_dets


# ─────────────────────────────────────────────────────────────
# Pass 2: gap filling
# ─────────────────────────────────────────────────────────────

def fill_gaps(n_frames, clean_dets, cache, seed_frame, seed_pos,
              interp_max_gap=20,
              y_off=51.0,
              x_off=0.0,
              person_accept_r=150.0,
              person_conf_min=0.3):
    """
    Returns positions dict: {frame_idx: (x, y)}.
    Strategy per frame:
      1. Direct clean detection → use it.
      2. Between two clean anchors, gap <= interp_max_gap → linear interpolation.
      3. Else → person foot estimation.
      4. Fallback → nearest anchor (hold last known).
    """
    positions = {}

    # Anchor points: seed + clean detections
    anchors = dict(clean_dets)
    anchors[seed_frame] = (float(seed_pos[0]), float(seed_pos[1]), 1.0)

    anchor_frames = sorted(anchors)

    for idx in range(n_frames):
        # Direct clean detection?
        if idx in anchors:
            cx, cy, _ = anchors[idx]
            positions[idx] = (int(cx), int(cy))
            continue

        # Find bracketing anchors
        prev_f = next_f = None
        for f in reversed([f for f in anchor_frames if f < idx]):
            prev_f = f; break
        for f in [f for f in anchor_frames if f > idx]:
            next_f = f; break

        used_interp = False
        if prev_f is not None and next_f is not None:
            gap = next_f - prev_f
            if gap <= interp_max_gap:
                # Linear interpolation
                t = (idx - prev_f) / gap
                px_a, py_a, _ = anchors[prev_f]
                px_b, py_b, _ = anchors[next_f]
                x = px_a + t * (px_b - px_a)
                y = py_a + t * (py_b - py_a)
                positions[idx] = (int(x), int(y))
                used_interp = True

        if used_interp:
            continue

        # Person-foot estimation
        # Determine reference point (last known or interp-estimated)
        if prev_f is not None:
            ref_x, ref_y, _ = anchors[prev_f]
        elif next_f is not None:
            ref_x, ref_y, _ = anchors[next_f]
        else:
            ref_x, ref_y = seed_pos

        persons = cache.get(idx, {}).get('persons', [])
        best_person = None
        best_d = 1e9
        for pcx, pcy, pcf, x1, y1, x2, y2 in persons:
            if pcf < person_conf_min:
                continue
            # Ball at feet: empirically ball_y ≈ player_center_y - 51px
            feet_y = pcy - y_off
            feet_x = pcx + x_off
            d = dist((feet_x, feet_y), (ref_x, ref_y))
            if d < person_accept_r and d < best_d:
                best_d = d
                best_person = (pcx, pcy, feet_x, feet_y)

        if best_person is not None:
            _, _, fx, fy = best_person
            positions[idx] = (int(clamp(fx, 0, W-1)), int(clamp(fy, 0, H-1)))
            continue

        # Hold last known / nearest anchor
        if prev_f is not None:
            cx, cy, _ = anchors[prev_f]
            positions[idx] = (int(cx), int(cy))
        elif next_f is not None:
            cx, cy, _ = anchors[next_f]
            positions[idx] = (int(cx), int(cy))
        else:
            positions[idx] = seed_pos

    return positions


# ─────────────────────────────────────────────────────────────
# Optional Pass 3: Kalman smoother within segments
# ─────────────────────────────────────────────────────────────

def kalman_smooth_segments(positions, clean_dets, seed_frame, seed_pos,
                            n_frames, kalman_q=1.0, r_det=5.0, r_fill=40.0):
    """
    Run a Kalman smoother forward, resetting at each clean detection anchor.
    This smooths inter-detection interpolations while preserving YOLO accuracy.
    """
    kf_pos = {}
    kf = None

    def reset_kf(x0, y0):
        kf_ = cv2.KalmanFilter(4, 2)
        kf_.transitionMatrix  = np.array([[1,0,1,0],[0,1,0,1],
                                           [0,0,1,0],[0,0,0,1]], dtype=np.float32)
        kf_.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        kf_.processNoiseCov    = np.eye(4, dtype=np.float32) * kalman_q
        kf_.measurementNoiseCov= np.eye(2, dtype=np.float32) * r_det
        kf_.errorCovPost = np.eye(4, dtype=np.float32) * 50.0
        kf_.statePost    = np.array([x0, y0, 0.0, 0.0], dtype=np.float32).reshape(-1,1)
        return kf_

    # Initialize at seed
    all_anchors = dict(clean_dets)
    all_anchors[seed_frame] = (float(seed_pos[0]), float(seed_pos[1]), 1.0)
    kf = reset_kf(*seed_pos)
    kf_pos[seed_frame] = seed_pos

    for idx in range(n_frames):
        if idx == seed_frame:
            continue
        if idx in all_anchors:
            # Hard reset at each clean detection
            cx, cy, _ = all_anchors[idx]
            kf = reset_kf(cx, cy)
            kf_pos[idx] = (int(cx), int(cy))
            continue

        pred = kf.predict()
        px = float(pred[0,0]); py = float(pred[1,0])

        if idx in positions:
            # Fill position: correct Kalman with higher noise
            fx, fy = positions[idx]
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r_fill
            meas = np.array([[float(fx)], [float(fy)]], dtype=np.float32)
            kf.correct(meas)
            state = kf.statePost
            sx = int(clamp(float(state[0,0]), 0, W-1))
            sy = int(clamp(float(state[1,0]), 0, H-1))
            kf_pos[idx] = (sx, sy)
        else:
            # No position: pure prediction with velocity decay
            state = kf.statePost
            kf.statePost[2,0] *= 0.85  # decay vx
            kf.statePost[3,0] *= 0.85  # decay vy
            bx = int(clamp(px, 0, W-1))
            by = int(clamp(py, 0, H-1))
            kf_pos[idx] = (bx, by)

    return kf_pos


# ─────────────────────────────────────────────────────────────
# Full tracker v7
# ─────────────────────────────────────────────────────────────

def track_v7(frames, seed_frame, seed_pos, cache,
             min_conf=0.10,
             hard_reset_conf=0.40,
             speed_limit=300.0,
             interp_max_gap=20,
             y_off=51.0,
             x_off=0.0,
             person_accept_r=150.0,
             person_conf_min=0.30,
             use_kalman_smooth=True,
             kalman_q=1.0):
    n = len(frames)

    # Pass 1
    raw_dets, clean_dets = collect_detections(
        cache, n,
        min_conf=min_conf,
        hard_reset_conf=hard_reset_conf,
        speed_limit=speed_limit)

    # Pass 2
    positions = fill_gaps(
        n, clean_dets, cache, seed_frame, seed_pos,
        interp_max_gap=interp_max_gap,
        y_off=y_off, x_off=x_off,
        person_accept_r=person_accept_r,
        person_conf_min=person_conf_min)

    # Pass 3 (optional)
    if use_kalman_smooth:
        positions = kalman_smooth_segments(
            positions, clean_dets, seed_frame, seed_pos,
            n, kalman_q=kalman_q)

    return positions


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate(positions, labels, tol30=30, tol50=50):
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
# Render
# ─────────────────────────────────────────────────────────────

def render_video(frames, positions, labels, out_path, fps=10):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    lmap = {lb['frame']: lb for lb in labels}
    for i, fp in enumerate(frames):
        fr = cv2.imread(fp)
        if fr is None:
            continue
        if i in positions:
            px, py = positions[i]
            cv2.circle(fr, (px, py), 10, (0,255,0), 2)
        if i in lmap:
            lb = lmap[i]
            cv2.circle(fr, (int(lb['px']), int(lb['py'])), 10, (0,0,255), 2)
        vw.write(fr)
    vw.release()
    print(f"Saved {out_path}  (green=pred, red=gt)")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video',       required=True)
    ap.add_argument('--labels',      required=True)
    ap.add_argument('--frames-dir',  default='')
    ap.add_argument('--render',      action='store_true')
    ap.add_argument('--rerun-yolo',  action='store_true')
    ap.add_argument('--model',       default='yolov8x.pt')
    ap.add_argument('--imgsz',       type=int, default=1920)
    args = ap.parse_args()

    with open(args.labels) as f:
        data = json.load(f)
    labels = data['labels']
    lmap   = {lb['frame']: lb for lb in labels}
    print(f"Loaded {len(labels)} labels")

    frames_dir = args.frames_dir or '/tmp/soccer_v7_frames'
    if args.frames_dir and os.path.isdir(args.frames_dir):
        frames = load_frames(args.frames_dir)
        print(f"Using {len(frames)} cached frames from {args.frames_dir}")
    else:
        os.makedirs(frames_dir, exist_ok=True)
        os.system(f'ffmpeg -i "{args.video}" -vf fps={FPS} '
                  f'-q:v 2 "{frames_dir}/frame_%06d.jpg" -y -loglevel error')
        frames = load_frames(frames_dir)
        print(f"Extracted {len(frames)} frames")

    # YOLO cache
    if os.path.exists(YOLO_CACHE_FILE) and not args.rerun_yolo:
        print(f"\nLoading YOLO cache from {YOLO_CACHE_FILE}...")
        cache = load_cache(YOLO_CACHE_FILE)
        print(f"  Loaded {len(cache)} frames")
    else:
        print(f"\nRunning {args.model} imgsz={args.imgsz} on {len(frames)} frames...")
        print(f"  This may take 30-45 minutes at imgsz={args.imgsz}.")
        t0 = time.time()
        cache = run_yolo_all_frames(frames, model_name=args.model,
                                    imgsz=args.imgsz, conf=0.05)
        save_cache(cache, YOLO_CACHE_FILE)
        print(f"  Done in {time.time()-t0:.1f}s")

    # Seed
    seed_pos   = (int(lmap[0]['px']), int(lmap[0]['py'])) if 0 in lmap else (W//2, H//2)
    seed_frame = 0
    print(f"\nSeed: frame {seed_frame} → {seed_pos}")

    # Diagnostic: how many labeled frames have YOLO detections?
    raw_dets_all, clean_dets_all = collect_detections(
        cache, len(frames), min_conf=0.10, hard_reset_conf=0.40, speed_limit=300.0)
    labeled_with_raw   = sum(1 for lb in labels if lb['frame'] in raw_dets_all)
    labeled_with_clean = sum(1 for lb in labels if lb['frame'] in clean_dets_all)
    print(f"  YOLO raw dets in labeled frames:   {labeled_with_raw}/{len(labels)}")
    print(f"  YOLO clean dets in labeled frames: {labeled_with_clean}/{len(labels)}")

    # Check accuracy of raw YOLO on labeled frames
    yolo_errs = []
    for lb in labels:
        f = lb['frame']
        if f in clean_dets_all:
            cx, cy, _ = clean_dets_all[f]
            yolo_errs.append(dist((cx,cy),(lb['px'],lb['py'])))
    if yolo_errs:
        yolo_w30 = sum(1 for e in yolo_errs if e<=30)
        yolo_w50 = sum(1 for e in yolo_errs if e<=50)
        print(f"  YOLO clean accuracy: W30={yolo_w30}  W50={yolo_w50}  "
              f"Med={np.median(yolo_errs):.1f}px  (of {len(yolo_errs)} detected)")

    # ── Baseline ─────────────────────────────────────────────
    print("\n--- Baseline v7 ---")
    t0 = time.time()
    pos = track_v7(frames, seed_frame, seed_pos, cache)
    tr, w30, w50, med, sc = evaluate(pos, labels)
    print(f"  Time: {time.time()-t0:.1f}s")
    print(f"  baseline_v7  {tr}/{len(labels)}  W30={w30:3d}  W50={w50:3d}  "
          f"Med={med:.1f}  Score={sc:.4f}")

    # ── Grid search ──────────────────────────────────────────
    print("\n--- Grid search v7 ---")

    hand_tuned = [
        # (a) Conservative: only high-conf, wide interp
        dict(min_conf=0.10, hard_reset_conf=0.40, speed_limit=300, interp_max_gap=20,
             y_off=51, x_off=0, person_accept_r=150, person_conf_min=0.30,
             use_kalman_smooth=True,  kalman_q=1.0),
        # (b) Aggressive: low conf, lots of interpolation
        dict(min_conf=0.08, hard_reset_conf=0.30, speed_limit=400, interp_max_gap=30,
             y_off=48, x_off=0, person_accept_r=200, person_conf_min=0.25,
             use_kalman_smooth=True,  kalman_q=0.5),
        # (c) No Kalman smooth
        dict(min_conf=0.10, hard_reset_conf=0.40, speed_limit=300, interp_max_gap=20,
             y_off=51, x_off=0, person_accept_r=150, person_conf_min=0.30,
             use_kalman_smooth=False, kalman_q=1.0),
        # (d) Short interp, heavy person
        dict(min_conf=0.12, hard_reset_conf=0.35, speed_limit=250, interp_max_gap=10,
             y_off=54, x_off=0, person_accept_r=100, person_conf_min=0.35,
             use_kalman_smooth=True,  kalman_q=2.0),
        # (e) Wide everything
        dict(min_conf=0.08, hard_reset_conf=0.25, speed_limit=500, interp_max_gap=40,
             y_off=51, x_off=5, person_accept_r=250, person_conf_min=0.20,
             use_kalman_smooth=True,  kalman_q=0.3),
        # (f) Tight conf, big gap
        dict(min_conf=0.15, hard_reset_conf=0.50, speed_limit=200, interp_max_gap=50,
             y_off=51, x_off=0, person_accept_r=120, person_conf_min=0.40,
             use_kalman_smooth=True,  kalman_q=1.5),
        # (g) Medium everything
        dict(min_conf=0.10, hard_reset_conf=0.35, speed_limit=350, interp_max_gap=25,
             y_off=49, x_off=-5, person_accept_r=180, person_conf_min=0.30,
             use_kalman_smooth=True,  kalman_q=0.8),
        # (h) No person estimation
        dict(min_conf=0.10, hard_reset_conf=0.40, speed_limit=300, interp_max_gap=20,
             y_off=51, x_off=0, person_accept_r=0, person_conf_min=1.0,
             use_kalman_smooth=True,  kalman_q=1.0),
    ]

    random_combos = []
    for _ in range(62):
        random_combos.append(dict(
            min_conf        = random.choice([0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]),
            hard_reset_conf = random.choice([0.25, 0.30, 0.40, 0.50, 0.60]),
            speed_limit     = random.choice([150, 200, 300, 400, 500, 700]),
            interp_max_gap  = random.choice([5, 10, 15, 20, 30, 40, 60]),
            y_off           = random.choice([40, 44, 47, 51, 54, 58, 65, 75]),
            x_off           = random.choice([-10, -5, 0, 5, 10]),
            person_accept_r = random.choice([60, 100, 150, 200, 300]),
            person_conf_min = random.choice([0.20, 0.25, 0.30, 0.40]),
            use_kalman_smooth = random.choice([True, False]),
            kalman_q        = random.choice([0.2, 0.5, 1.0, 2.0, 5.0]),
        ))

    all_combos = hand_tuned + random_combos
    print(f"  Running {len(all_combos)} combinations...")

    results_log = []
    best_score = -1; best_pos = None; best_params = None

    for ci, params in enumerate(all_combos):
        pos = track_v7(frames, seed_frame, seed_pos, cache, **params)
        tr, w30, w50, med, sc = evaluate(pos, labels)
        tag = (f"mc={params['min_conf']:.2f} hr={params['hard_reset_conf']:.2f} "
               f"sl={params['speed_limit']:.0f} ig={params['interp_max_gap']} "
               f"yo={params['y_off']} par={params['person_accept_r']} "
               f"ks={params['use_kalman_smooth']}")
        results_log.append((sc, w30, w50, med, tr, tag, params, pos))
        if sc > best_score:
            best_score = sc; best_params = params; best_pos = pos
        if (ci+1) % 10 == 0:
            top = sorted(results_log, key=lambda r: r[0], reverse=True)[0]
            print(f"    {ci+1}/{len(all_combos)} done — "
                  f"best: score={top[0]:.4f} w30={top[1]} w50={top[2]} med={top[3]:.0f}px")

    results_log.sort(key=lambda r: r[0], reverse=True)

    print(f"\n{'='*80}")
    print(f"  TOP 10:")
    print(f"{'='*80}")
    for sc, w30, w50, med, tr, tag, _, _ in results_log[:10]:
        print(f"  {tag}  {tr}/{len(labels)}  W30={w30:3d}  W50={w50:3d}  "
              f"Med={med:.1f}  Score={sc:.4f}")

    sc, w30, w50, med, tr, tag, best_params, best_pos = results_log[0]
    print(f"\n{'='*80}")
    print(f"  WINNER: {best_params}")
    print(f"  Score={sc:.4f}  Tracked={tr}/{len(labels)}  "
          f"W30={w30}  W50={w50}  Median={med:.1f}px")
    print(f"{'='*80}")

    # Per-frame detail
    print("\nPer-frame detail (winner):")
    for f_idx in sorted(lmap):
        lb = lmap[f_idx]
        gx, gy = int(lb['px']), int(lb['py'])
        if f_idx in best_pos:
            px, py = best_pos[f_idx]
            err = dist((px,py),(gx,gy))
            flag = ' <-- MISS' if err > 50 else ''
            print(f"  frame {f_idx:4d}: err={err:6.1f}px  "
                  f"pred=({px:4d},{py:4d})  gt=({gx:4d},{gy:4d}){flag}")
        else:
            print(f"  frame {f_idx:4d}: NO PRED  gt=({gx:4d},{gy:4d}) <-- MISS")

    with open('tracking_analysis_v7.txt', 'w') as f:
        f.write(f"V7 Winner: score={sc:.4f} w30={w30} w50={w50} med={med:.1f}\n")
        f.write(f"Params: {best_params}\n\n")
        for s, w3, w5, md, t, tg, _, _ in results_log[:20]:
            f.write(f"{s:.4f}  W30={w3}  W50={w5}  Med={md:.1f}  {tg}\n")
    print("\nSaved to tracking_analysis_v7.txt")

    if args.render:
        print("\nRendering tracking_v7_winner.mp4...")
        render_video(frames, best_pos, labels, 'tracking_v7_winner.mp4')


if __name__ == '__main__':
    main()
