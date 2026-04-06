#!/usr/bin/env python3
"""
Ball tracking v6 — YOLOv8x + person-foot estimation + Kalman.

Architecture:
  1. Pre-run YOLOv8x (imgsz=1280) on ALL 1471 frames → cache ball & person detections.
  2. Filter false-positive YOLO ball detections by:
     a. Confidence threshold (min_conf)
     b. Velocity consistency: reject if implied speed > max_speed px/frame
  3. For frames with no ball detection, try person-based estimation:
     - Find person nearest Kalman-predicted ball position
     - Ball ≈ (person_cx - x_off, person_cy - y_off)  [y_off ≈ 51px empirically]
     - Accept only if estimate is within person_accept_r of Kalman prediction
  4. Kalman filter (4-state x,y,vx,vy) integrates all detections.
     - Low process noise → smooth trajectory, respects physics
     - Adaptive measurement noise: YOLO ball=R_ball, person estimate=R_person
  5. Grid search over: min_conf, max_speed, q, r_ball, r_person, y_off, person_accept_r

Key insight from v5 analysis:
  - Ball is NOT reliably white (dark panels, shadow); white-blob fails.
  - Template matching drifts within 1-2 frames.
  - YOLO sports-ball (class 32) is highly accurate (d<10px) but only for ~60% of frames.
  - For stationary phase (frames 0-175): player detected ~50px below ball in y.

Usage:
  # Fast eval (use cached YOLO results if available):
  .venv/bin/python3 analyze_tracking_v6.py --video Test.mp4 \\
      --labels ball_labels.json --frames-dir /tmp/soccer_v2_frames --render

  # Force re-run YOLO (slow, ~20-30 min):
  .venv/bin/python3 analyze_tracking_v6.py --video Test.mp4 \\
      --labels ball_labels.json --frames-dir /tmp/soccer_v2_frames --render --rerun-yolo
"""

import argparse
import json
import os
import random
import time

import cv2
import numpy as np

FPS   = 10.0
W, H  = 1920, 1080
FIELD_Y_MIN = int(0.18 * H)
FIELD_Y_MAX = int(0.86 * H)

YOLO_CACHE_FILE = 'yolo_cache_v6.json'

BALL_CLASS   = 32   # COCO sports ball
PERSON_CLASS = 0    # COCO person


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def load_frames(frames_dir):
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith('.jpg'))
    return [os.path.join(frames_dir, f) for f in files]


# ─────────────────────────────────────────────────────────────
# YOLO inference + caching
# ─────────────────────────────────────────────────────────────

def run_yolo_all_frames(frames, model_name='yolov8x.pt', imgsz=1280, conf=0.08):
    """
    Run YOLO on every frame. Returns dict:
      cache[i] = {
        'balls':   [(cx,cy,conf), ...],
        'persons': [(cx,cy,conf,x1,y1,x2,y2), ...]
      }
    """
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
        balls   = []
        persons = []
        for j, cls in enumerate(r.boxes.cls):
            cls_id = int(cls)
            box  = r.boxes.xyxy[j].cpu().numpy()
            cf   = float(r.boxes.conf[j])
            cx   = float((box[0]+box[2])/2)
            cy   = float((box[1]+box[3])/2)
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
# Kalman factory
# ─────────────────────────────────────────────────────────────

def make_kalman(x0, y0, q=1.0, r=10.0):
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix  = np.array([[1,0,1,0],[0,1,0,1],
                                      [0,0,1,0],[0,0,0,1]], dtype=np.float32)
    kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
    kf.processNoiseCov    = np.eye(4, dtype=np.float32) * q
    kf.measurementNoiseCov= np.eye(2, dtype=np.float32) * r
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 50.0
    kf.statePost    = np.array([x0, y0, 0.0, 0.0], dtype=np.float32).reshape(-1,1)
    return kf


# ─────────────────────────────────────────────────────────────
# Core tracker v6
# ─────────────────────────────────────────────────────────────

def track_v6(frames, seed_frame, seed_pos, cache,
             min_conf=0.20,
             max_speed=160.0,
             kalman_q=0.8,
             r_ball=8.0,
             r_person=200.0,
             y_off=51.0,
             x_off=0.0,
             person_accept_r=120.0,
             conf_scale=True):
    """
    Track ball using YOLO detections + person-foot estimation + Kalman.

    conf_scale: if True, scale measurement noise inversely with confidence.
    """
    n = len(frames)
    positions = {}

    kf = make_kalman(seed_pos[0], seed_pos[1], q=kalman_q, r=r_ball)
    positions[seed_frame] = seed_pos
    last_pos  = seed_pos
    velocity  = (0.0, 0.0)

    def kalman_state():
        x = float(kf.statePost[0, 0])
        y = float(kf.statePost[1, 0])
        vx = float(kf.statePost[2, 0])
        vy = float(kf.statePost[3, 0])
        return x, y, vx, vy

    for idx in range(n):
        if idx == seed_frame:
            continue

        pred = kf.predict()
        px = float(pred[0, 0])
        py = float(pred[1, 0])

        frame_data = cache.get(idx, {'balls': [], 'persons': []})
        balls   = frame_data.get('balls', [])
        persons = frame_data.get('persons', [])

        # ── Best YOLO ball detection ────────────────────────────
        best_ball = None
        best_d    = 1e9
        for cx, cy, cf in balls:
            if cf < min_conf:
                continue
            d = dist((cx, cy), (px, py))
            # Velocity check: how fast would the ball need to move?
            speed_implied = dist((cx, cy), last_pos)
            if speed_implied > max_speed:
                continue
            if d < best_d:
                best_d = d
                best_ball = (cx, cy, cf)

        if best_ball is not None:
            bx, by, bcf = best_ball
            # Adaptive measurement noise based on confidence
            if conf_scale:
                r_eff = r_ball / max(bcf, 0.05)
            else:
                r_eff = r_ball
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r_eff
            meas = np.array([[bx], [by]], dtype=np.float32)
            kf.correct(meas)
            vx = (bx - last_pos[0]) * 0.65 + velocity[0] * 0.35
            vy = (by - last_pos[1]) * 0.65 + velocity[1] * 0.35
            velocity = (vx, vy)
            last_pos = (int(bx), int(by))
            positions[idx] = last_pos
            continue

        # ── Person-based ball estimation ────────────────────────
        best_person = None
        best_pd     = 1e9
        for cx, cy, cf, x1, y1, x2, y2 in persons:
            # Estimate ball from person
            ball_est_x = cx + x_off
            ball_est_y = cy - y_off
            d = dist((ball_est_x, ball_est_y), (px, py))
            if d < person_accept_r and d < best_pd:
                best_pd = d
                best_person = (cx, cy, cf, x1, y1, x2, y2)

        if best_person is not None:
            cx, cy, cf, x1, y1, x2, y2 = best_person
            ball_est_x = cx + x_off
            ball_est_y = cy - y_off
            # Only correct y (x from Kalman prediction is often more accurate)
            # Use asymmetric measurement: y from person, x from Kalman prediction
            kf.measurementNoiseCov = np.array([[r_ball * 4, 0],
                                                [0, r_person]],
                                               dtype=np.float32)
            meas = np.array([[px], [ball_est_y]], dtype=np.float32)
            kf.correct(meas)
            new_state = kf.statePost
            bx = int(clamp(float(new_state[0,0]), 0, W-1))
            by = int(clamp(float(new_state[1,0]), 0, H-1))
            positions[idx] = (bx, by)
            last_pos = (bx, by)
            continue

        # ── No detection: pure Kalman prediction ───────────────
        bx = int(clamp(px, 0, W-1))
        by = int(clamp(py, 0, H-1))
        # Decay velocity gently
        vx = velocity[0] * 0.92
        vy = velocity[1] * 0.92
        velocity = (vx, vy)
        positions[idx] = (bx, by)

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
        errs.append(dist((px, py), (gx, gy)))
    if not errs:
        return 0, 0, 0, float('inf'), 0.0
    w30 = sum(1 for e in errs if e <= tol30)
    w50 = sum(1 for e in errs if e <= tol50)
    med = float(np.median(errs))
    score = (w30 * 2 + w50) / (3 * len(labels))
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
            cv2.circle(fr, (px, py), 10, (0, 255, 0), 2)
        if i in lmap:
            lb = lmap[i]
            cv2.circle(fr, (int(lb['px']), int(lb['py'])), 10, (0, 0, 255), 2)
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
    ap.add_argument('--rerun-yolo',  action='store_true',
                    help='Force re-run YOLO even if cache exists')
    ap.add_argument('--model',       default='yolov8x.pt')
    ap.add_argument('--imgsz',       type=int, default=1280)
    args = ap.parse_args()

    with open(args.labels) as f:
        data = json.load(f)
    labels = data['labels']
    lmap   = {lb['frame']: lb for lb in labels}
    print(f"Loaded {len(labels)} labels")

    # ── Load frames ──────────────────────────────────────────
    frames_dir = args.frames_dir or '/tmp/soccer_v6_frames'
    if args.frames_dir and os.path.isdir(args.frames_dir):
        frames = load_frames(args.frames_dir)
        print(f"Using {len(frames)} cached frames from {args.frames_dir}")
    else:
        os.makedirs(frames_dir, exist_ok=True)
        os.system(f'ffmpeg -i "{args.video}" -vf fps={FPS} '
                  f'-q:v 2 "{frames_dir}/frame_%06d.jpg" -y -loglevel error')
        frames = load_frames(frames_dir)
        print(f"Extracted {len(frames)} frames")

    # ── YOLO inference (all frames) ──────────────────────────
    if os.path.exists(YOLO_CACHE_FILE) and not args.rerun_yolo:
        print(f"\nLoading YOLO cache from {YOLO_CACHE_FILE}...")
        cache = load_cache(YOLO_CACHE_FILE)
        print(f"  Loaded {len(cache)} frames of YOLO data")
    else:
        print(f"\nRunning {args.model} imgsz={args.imgsz} on {len(frames)} frames...")
        print("  This may take 20-30 minutes for 1471 frames.")
        t0 = time.time()
        cache = run_yolo_all_frames(frames, model_name=args.model,
                                    imgsz=args.imgsz, conf=0.08)
        save_cache(cache, YOLO_CACHE_FILE)
        print(f"  Done in {time.time()-t0:.1f}s")

    # Seed from first label
    if 0 in lmap:
        seed_pos = (int(lmap[0]['px']), int(lmap[0]['py']))
    else:
        seed_pos = (W // 2, H // 2)
    seed_frame = 0
    print(f"\nSeed: frame {seed_frame} → {seed_pos}")

    # ── Baseline run ─────────────────────────────────────────
    print("\n--- Baseline v6 ---")
    t0 = time.time()
    pos = track_v6(frames, seed_frame, seed_pos, cache)
    tr, w30, w50, med, sc = evaluate(pos, labels)
    print(f"  Time: {time.time()-t0:.1f}s")
    print(f"  baseline_v6  {tr}/{len(labels)}  W30={w30:3d}  W50={w50:3d}  "
          f"Med={med:.1f}  Score={sc:.4f}")

    # ── Grid search ──────────────────────────────────────────
    print("\n--- Grid search v6 ---")

    hand_tuned = [
        dict(min_conf=0.20, max_speed=160, kalman_q=0.8, r_ball=8.0,
             r_person=200, y_off=51.0, x_off=0.0, person_accept_r=120, conf_scale=True),
        dict(min_conf=0.15, max_speed=180, kalman_q=1.2, r_ball=5.0,
             r_person=150, y_off=51.0, x_off=0.0, person_accept_r=100, conf_scale=True),
        dict(min_conf=0.25, max_speed=150, kalman_q=0.5, r_ball=10.0,
             r_person=250, y_off=51.0, x_off=0.0, person_accept_r=140, conf_scale=True),
        dict(min_conf=0.12, max_speed=200, kalman_q=2.0, r_ball=6.0,
             r_person=180, y_off=51.0, x_off=-5.0, person_accept_r=130, conf_scale=True),
        dict(min_conf=0.18, max_speed=170, kalman_q=1.0, r_ball=8.0,
             r_person=300, y_off=53.0, x_off=0.0, person_accept_r=150, conf_scale=True),
        dict(min_conf=0.20, max_speed=160, kalman_q=0.8, r_ball=8.0,
             r_person=200, y_off=51.0, x_off=0.0, person_accept_r=120, conf_scale=False),
        dict(min_conf=0.30, max_speed=140, kalman_q=0.3, r_ball=15.0,
             r_person=400, y_off=49.0, x_off=0.0, person_accept_r=80, conf_scale=True),
        dict(min_conf=0.10, max_speed=220, kalman_q=3.0, r_ball=4.0,
             r_person=100, y_off=51.0, x_off=0.0, person_accept_r=160, conf_scale=True),
    ]

    random_combos = []
    for _ in range(42):
        random_combos.append(dict(
            min_conf      = random.choice([0.08, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40]),
            max_speed     = random.choice([120, 140, 160, 180, 200, 240]),
            kalman_q      = random.choice([0.2, 0.5, 1.0, 2.0, 4.0, 8.0]),
            r_ball        = random.choice([3.0, 6.0, 10.0, 20.0, 50.0]),
            r_person      = random.choice([80, 150, 250, 400, 800]),
            y_off         = random.choice([45.0, 48.0, 51.0, 54.0, 58.0]),
            x_off         = random.choice([-10.0, -5.0, 0.0, 5.0]),
            person_accept_r = random.choice([60, 100, 140, 180, 240]),
            conf_scale    = random.choice([True, False]),
        ))

    all_combos = hand_tuned + random_combos
    print(f"  Running {len(all_combos)} combinations...")

    results_log = []
    best_score = -1; best_pos = None; best_params = None

    for ci, params in enumerate(all_combos):
        pos = track_v6(frames, seed_frame, seed_pos, cache, **params)
        tr, w30, w50, med, sc = evaluate(pos, labels)
        tag = (f"mc={params['min_conf']:.2f} ms={params['max_speed']:.0f} "
               f"q={params['kalman_q']} rb={params['r_ball']} "
               f"rp={params['r_person']} yo={params['y_off']} "
               f"par={params['person_accept_r']} cs={params['conf_scale']}")
        results_log.append((sc, w30, w50, med, tr, tag, params, pos))
        if sc > best_score:
            best_score = sc; best_params = params; best_pos = pos
        if (ci + 1) % 10 == 0:
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
            err = dist((px, py), (gx, gy))
            flag = ' <-- MISS' if err > 50 else ''
            print(f"  frame {f_idx:4d}: err={err:6.1f}px  "
                  f"pred=({px:4d},{py:4d})  gt=({gx:4d},{gy:4d}){flag}")
        else:
            print(f"  frame {f_idx:4d}: NO PRED  gt=({gx:4d},{gy:4d}) <-- MISS")

    with open('tracking_analysis_v6.txt', 'w') as f:
        f.write(f"V6 Winner: score={sc:.4f} w30={w30} w50={w50} med={med:.1f}\n")
        f.write(f"Params: {best_params}\n\n")
        for s, w3, w5, md, t, tg, _, _ in results_log[:20]:
            f.write(f"{s:.4f}  W30={w3}  W50={w5}  Med={md:.1f}  {tg}\n")
    print("\nSaved to tracking_analysis_v6.txt")

    if args.render:
        print("\nRendering tracking_v6_winner.mp4...")
        render_video(frames, best_pos, labels, 'tracking_v6_winner.mp4')


if __name__ == '__main__':
    main()
