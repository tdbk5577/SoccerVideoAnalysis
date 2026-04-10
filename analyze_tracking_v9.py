#!/usr/bin/env python3
"""
Ball tracking v9 — TrackNet detections at 15fps, pixel-space tracker.

Self-contained: no dependency on v8, no field calibration file, no person fallback.
TrackNet provides dense ball detections; the tracker handles gap-filling and smoothing.

Prerequisites:
  1. frames_15fps/ directory with 15fps frames  (build_cache_15fps.py)
  2. tracknet_model.pth                          (tracknet.py train)
  3. tracknet_cache_15fps.json                   (tracknet.py infer)

Usage:
  .venv/bin/python3 analyze_tracking_v9.py \\
      --video "test 1.mp4" \\
      --labels ball_labels.json \\
      --tracknet_cache tracknet_cache_15fps.json \\
      [--render] [--random-seeds 40]
"""

import argparse
import json
import os
import random
import subprocess

import cv2
import numpy as np

FPS        = 15.0
FRAMES_DIR = 'frames_15fps'


# ── Helpers ───────────────────────────────────────────────────────────────────

def dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5


def load_frames(frames_dir):
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith('.jpg'))
    return [os.path.join(frames_dir, f) for f in files]


def load_cache(path):
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def remap_labels_15fps(labels_data, fps=FPS):
    seen, out = set(), []
    for lb in sorted(labels_data['labels'], key=lambda x: x['t']):
        f15 = int(round(lb['t'] * fps))
        if f15 not in seen:
            seen.add(f15)
            out.append({'frame': f15, 't': lb['t'], 'px': lb['px'], 'py': lb['py']})
    return out


# ── Tracker ───────────────────────────────────────────────────────────────────

def collect_anchors(cache, n_frames, seed_frame, seed_pos,
                    min_conf=0.10, hard_reset_conf=0.50,
                    speed_limit=150.0, vel_decay=0.85):
    """
    Pass 1: velocity-guided anchor collection in pixel space.
    speed_limit: max pixels/frame (150px/frame @ 15fps ≈ 144 km/h ball)
    """
    rx, ry  = float(seed_pos[0]), float(seed_pos[1])
    rvx, rvy = 0.0, 0.0
    last_det = seed_frame
    anchors  = {seed_frame: (rx, ry, 1.0)}

    for idx in range(n_frames):
        if idx == seed_frame:
            continue
        balls = [b for b in cache.get(idx, {}).get('balls', []) if b[2] >= min_conf]
        if not balls:
            rvx *= vel_decay; rvy *= vel_decay
            rx  += rvx;       ry  += rvy
            continue

        dt     = max(idx - last_det, 1)
        pred_x = rx + rvx * dt
        pred_y = ry + rvy * dt

        close = [(b, dist((b[0],b[1]),(pred_x,pred_y))/dt)
                  for b in balls
                  if dist((b[0],b[1]),(pred_x,pred_y))/dt <= speed_limit]

        if close:
            best, _ = min(close, key=lambda x: x[1])
        else:
            hc = [b for b in balls if b[2] >= hard_reset_conf]
            if hc:
                best = max(hc, key=lambda b: b[2])
            else:
                rvx *= vel_decay; rvy *= vel_decay
                rx  += rvx;       ry  += rvy
                continue

        bx, by, bc = best
        rvx = (bx-rx)/dt * 0.5 + rvx * 0.5
        rvy = (by-ry)/dt * 0.5 + rvy * 0.5
        rx, ry   = bx, by
        last_det = idx
        anchors[idx] = (bx, by, bc)

    return anchors


def fill_gaps(n_frames, anchors, interp_max_gap=200):
    """Pass 2: linear pixel interpolation between anchors."""
    anchor_frames = sorted(anchors)
    positions = {}

    for idx in range(n_frames):
        if idx in anchors:
            cx, cy, _ = anchors[idx]
            positions[idx] = (int(cx), int(cy))
            continue

        prev_f = next((f for f in reversed(anchor_frames) if f < idx), None)
        next_f = next((f for f in anchor_frames if f > idx), None)

        if prev_f is not None and next_f is not None and (next_f - prev_f) <= interp_max_gap:
            t = (idx - prev_f) / (next_f - prev_f)
            ax, ay, _ = anchors[prev_f]
            bx, by, _ = anchors[next_f]
            positions[idx] = (int(ax + t*(bx-ax)), int(ay + t*(by-ay)))
        elif prev_f is not None:
            cx, cy, _ = anchors[prev_f]
            positions[idx] = (int(cx), int(cy))
        elif next_f is not None:
            cx, cy, _ = anchors[next_f]
            positions[idx] = (int(cx), int(cy))

    return positions


def kalman_smooth(positions, anchors, n_frames, kalman_q=1.0):
    """Pass 3: Kalman smoother in pixel space."""
    W, H = 1920, 1080

    def reset_kf(x0, y0):
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        kf.processNoiseCov     = np.eye(4, dtype=np.float32) * kalman_q
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * max(1.0, kalman_q*0.1)
        kf.errorCovPost        = np.eye(4, dtype=np.float32) * 50.0
        kf.statePost           = np.array([x0, y0, 0.0, 0.0], dtype=np.float32).reshape(-1,1)
        return kf

    first_anchor = min(anchors)
    ax0, ay0, _  = anchors[first_anchor]
    kf = reset_kf(ax0, ay0)
    kf_pos = {}

    for idx in range(n_frames):
        if idx in anchors:
            cx, cy, _ = anchors[idx]
            kf = reset_kf(cx, cy)
            kf_pos[idx] = (int(cx), int(cy))
            continue

        pred = kf.predict()
        px, py = float(pred[0,0]), float(pred[1,0])

        if idx in positions:
            fx, fy = positions[idx]
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * max(kalman_q*5, 50.0)
            kf.correct(np.array([[float(fx)], [float(fy)]], dtype=np.float32))
            s = kf.statePost
            kf_pos[idx] = (int(np.clip(s[0,0], 0, W-1)), int(np.clip(s[1,0], 0, H-1)))
        else:
            kf.statePost[2,0] *= 0.85
            kf.statePost[3,0] *= 0.85
            kf_pos[idx] = (int(np.clip(px, 0, W-1)), int(np.clip(py, 0, H-1)))

    return kf_pos


def track(cache, n_frames, seed_frame, seed_pos,
          min_conf=0.10, hard_reset_conf=0.50,
          speed_limit=150.0, vel_decay=0.85,
          interp_max_gap=200,
          use_kalman=True, kalman_q=2.0):

    anchors   = collect_anchors(cache, n_frames, seed_frame, seed_pos,
                                 min_conf=min_conf, hard_reset_conf=hard_reset_conf,
                                 speed_limit=speed_limit, vel_decay=vel_decay)
    positions = fill_gaps(n_frames, anchors, interp_max_gap=interp_max_gap)

    if use_kalman:
        positions = kalman_smooth(positions, anchors, n_frames, kalman_q=kalman_q)

    return positions


def track_backward(cache, n_frames, bwd_seed_frame, bwd_seed_pos, **params):
    cache_rev = {n_frames-1-k: v for k, v in cache.items()}
    seed_rev  = n_frames - 1 - bwd_seed_frame

    pos_rev = track(cache_rev, n_frames, seed_rev, bwd_seed_pos, **params)
    return {n_frames-1-k: v for k, v in pos_rev.items()}


def track_bidir(cache, n_frames, seed_frame, seed_pos,
                bwd_seed_frame, bwd_seed_pos,
                agreement_px=30.0, **params):
    """Forward + backward track, merge by agreement."""
    pos_fwd = track(cache, n_frames, seed_frame, seed_pos, **params)
    pos_bwd = track_backward(cache, n_frames, bwd_seed_frame, bwd_seed_pos, **params)

    merged = {}
    for f in range(n_frames):
        has_f = f in pos_fwd
        has_b = f in pos_bwd
        if has_f and not has_b:
            merged[f] = pos_fwd[f]
        elif has_b and not has_f:
            merged[f] = pos_bwd[f]
        else:
            pf, pb = pos_fwd[f], pos_bwd[f]
            if dist(pf, pb) <= agreement_px:
                merged[f] = (int((pf[0]+pb[0])/2), int((pf[1]+pb[1])/2))
            else:
                # Pick direction with nearest anchor (seed proximity)
                df = abs(f - seed_frame)
                db = abs(f - bwd_seed_frame)
                merged[f] = pos_fwd[f] if df <= db else pos_bwd[f]
    return merged


# ── Evaluation & rendering ────────────────────────────────────────────────────

def evaluate(positions, labels):
    errs = [dist(positions[lb['frame']], (lb['px'], lb['py']))
            for lb in labels if lb['frame'] in positions]
    if not errs:
        return 0, 0, 0, float('inf'), 0.0
    w30  = sum(1 for e in errs if e <= 30)
    w50  = sum(1 for e in errs if e <= 50)
    med  = float(np.median(errs))
    score = (w30*2 + w50) / (3*len(labels))
    return len(errs), w30, w50, med, score


def render_video(frames, positions, labels, out_path, fps=15):
    ref = cv2.imread(frames[0])
    h, w = ref.shape[:2]
    vw  = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
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
    print(f'Saved {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video',          required=True)
    ap.add_argument('--labels',         required=True)
    ap.add_argument('--tracknet_cache', required=True)
    ap.add_argument('--frames_dir',     default=FRAMES_DIR)
    ap.add_argument('--render',         action='store_true')
    ap.add_argument('--output',         default='tracking_v9_winner.mp4')
    ap.add_argument('--analysis_out',   default='tracking_analysis_v9.txt')
    ap.add_argument('--random-seeds',   type=int, default=40)
    args = ap.parse_args()

    # Labels
    with open(args.labels) as f:
        labels_data = json.load(f)
    labels = remap_labels_15fps(labels_data)
    print(f'Labels: {len(labels)} remapped to 15fps  '
          f'(frames {labels[0]["frame"]}..{labels[-1]["frame"]})')

    # Frames
    frames_dir = args.frames_dir
    if not os.path.isdir(frames_dir) or not os.listdir(frames_dir):
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Extracting frames from '{args.video}' at {FPS}fps...")
        subprocess.run(['ffmpeg', '-i', args.video, '-vf', f'fps={FPS}',
                        '-q:v', '2', '-loglevel', 'error',
                        os.path.join(frames_dir, 'frame_%06d.jpg')], check=True)
    frames = load_frames(frames_dir)
    n = len(frames)
    print(f'Frames: {n}')

    # TrackNet cache
    print(f'Loading {args.tracknet_cache}...')
    cache = load_cache(args.tracknet_cache)
    total_dets = sum(len(v.get('balls',[])) for v in cache.values())
    hc_dets    = sum(1 for v in cache.values() for b in v.get('balls',[]) if b[2] >= 0.5)
    print(f'  {len(cache)} frames  {total_dets} ball dets  {hc_dets} with conf>=0.5')

    # Seeds
    seed_frame  = labels[0]['frame']
    seed_pos    = (int(labels[0]['px']), int(labels[0]['py']))
    bwd_seed_frame = labels[-1]['frame']
    bwd_seed_pos   = (int(labels[-1]['px']), int(labels[-1]['py']))
    print(f'Seed: frame {seed_frame} {seed_pos}')
    print(f'BWD seed: frame {bwd_seed_frame} {bwd_seed_pos}')

    # Parameter search
    print(f'\n--- Parameter search v9 (TrackNet + 15fps) ---')

    hand_tuned = [
        dict(min_conf=0.10, hard_reset_conf=0.50, speed_limit=150, vel_decay=0.85,
             interp_max_gap=200, use_kalman=True, kalman_q=2.0, agreement_px=30),
        dict(min_conf=0.15, hard_reset_conf=0.50, speed_limit=120, vel_decay=0.85,
             interp_max_gap=150, use_kalman=True, kalman_q=1.0, agreement_px=25),
        dict(min_conf=0.07, hard_reset_conf=0.40, speed_limit=200, vel_decay=0.80,
             interp_max_gap=300, use_kalman=True, kalman_q=5.0, agreement_px=40),
        dict(min_conf=0.10, hard_reset_conf=0.50, speed_limit=150, vel_decay=0.85,
             interp_max_gap=200, use_kalman=False, kalman_q=2.0, agreement_px=30),
        dict(min_conf=0.20, hard_reset_conf=0.60, speed_limit=100, vel_decay=0.90,
             interp_max_gap=100, use_kalman=True, kalman_q=0.5, agreement_px=20),
        dict(min_conf=0.05, hard_reset_conf=0.40, speed_limit=250, vel_decay=0.75,
             interp_max_gap=400, use_kalman=True, kalman_q=10.0, agreement_px=50),
    ]

    rng = random.Random(42)
    random_combos = [dict(
        min_conf          = rng.choice([0.05, 0.08, 0.10, 0.12, 0.15, 0.20]),
        hard_reset_conf   = rng.choice([0.35, 0.40, 0.45, 0.50, 0.55, 0.60]),
        speed_limit       = rng.uniform(80, 250),
        vel_decay         = rng.uniform(0.75, 0.95),
        interp_max_gap    = rng.choice([100, 150, 200, 250, 300, 400]),
        use_kalman        = True,
        kalman_q          = rng.uniform(0.3, 15.0),
        agreement_px      = rng.uniform(15, 60),
    ) for _ in range(args.random_seeds)]

    all_combos  = hand_tuned + random_combos
    results_log = []
    best_score  = -1
    best_pos    = None
    best_params = None

    for i, params in enumerate(all_combos):
        tag = f'hand_{i}' if i < len(hand_tuned) else f'rand_{i-len(hand_tuned)}'
        pos = track_bidir(cache, n, seed_frame, seed_pos,
                          bwd_seed_frame, bwd_seed_pos, **params)
        tr, w30, w50, med, sc = evaluate(pos, labels)
        results_log.append((sc, w30, w50, med, tag, params, pos))
        if sc > best_score:
            best_score, best_params, best_pos = sc, params, pos
        if (i+1) % 10 == 0:
            top = max(results_log, key=lambda x: x[0])
            print(f'  [{i+1}/{len(all_combos)}] best: score={top[0]:.4f} '
                  f'w30={top[1]} w50={top[2]} med={top[3]:.0f}px')

    results_log.sort(key=lambda x: x[0], reverse=True)

    print(f'\nTop 5:')
    for sc, w30, w50, med, tag, params, _ in results_log[:5]:
        print(f'  {tag}  score={sc:.4f}  W30={w30}  W50={w50}  Med={med:.0f}px')

    sc, w30, w50, med, tag, best_params, best_pos = results_log[0]
    print(f'\nWINNER: score={sc:.4f}  W30={w30}/{len(labels)}  '
          f'W50={w50}/{len(labels)}  Median={med:.1f}px')
    print(f'  Params: {best_params}')

    miss_count = 0
    print('\nErrors > 50px:')
    for lb in labels:
        f = lb['frame']
        if f not in best_pos:
            print(f'  f{f:4d}  NO_DET'); miss_count += 1; continue
        px, py = best_pos[f]
        err = dist((px,py),(lb['px'],lb['py']))
        if err > 50:
            print(f'  f{f:4d}  err={err:.0f}px'); miss_count += 1
    print(f'  Total > 50px: {miss_count}/{len(labels)}')

    with open(args.analysis_out, 'w') as f:
        f.write(f'V9 Winner: score={sc:.4f} w30={w30} w50={w50} med={med:.1f}\n')
        f.write(f'Params: {best_params}\n\n')
        for sc2, w30_2, w50_2, md, tg, p, _ in results_log:
            f.write(f'{sc2:.4f}  W30={w30_2}  W50={w50_2}  Med={md:.1f}  {tg}\n')
    print(f'Saved {args.analysis_out}')

    if args.render:
        render_video(frames, best_pos, labels, args.output, fps=int(FPS))


if __name__ == '__main__':
    main()
