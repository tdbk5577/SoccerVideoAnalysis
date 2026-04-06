#!/usr/bin/env python3
"""
Ball tracking v7b — v7 + A→B→C bidirectional consistency filter + running ref.

Key new features vs v7:
  1. A→B→C filter: if C (next det) is closer to A (prev det) than to B (current det)
     AND B is fast from A, then B is a false positive → remove it.
     This handles the "tracker jumps right, ball stays left" failure mode.
  2. Running reference in fill_gaps: update ref after each person-estimated frame
     so the ref tracks the ball through the stationary phase.
  3. Grid search includes interp_max_gap up to 100 frames.

Uses yolo_cache_v7.json (YOLO@1920, already computed).
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
# Pass 1: collect & clean YOLO detections
# ─────────────────────────────────────────────────────────────

def collect_detections(cache, n_frames,
                        min_conf=0.10,
                        hard_reset_conf=0.40,
                        speed_limit=250.0,
                        abc_ratio=0.6):
    """
    Returns clean_dets: {frame_idx: (cx, cy, conf)}

    Two-stage cleaning:
      Stage 1 (speed): remove if speed > speed_limit from BOTH neighbors
                       (unless conf >= hard_reset_conf)
      Stage 2 (A→B→C): if B is fast from A and C is "closer to A than B",
                        then B is a false positive:
                        condition: dist(C,A) < abc_ratio * dist(C,B)
                        AND speed(A→B) > speed_limit * 0.5
    """
    # Best detection per frame
    raw_dets = {}
    for idx in range(n_frames):
        balls = cache.get(idx, {}).get('balls', [])
        if not balls:
            continue
        best = max(balls, key=lambda b: b[2])
        if best[2] >= min_conf:
            raw_dets[idx] = tuple(best)

    clean_dets = dict(raw_dets)

    # Stage 1: speed filter (3 passes)
    for _ in range(3):
        clean_frames = sorted(clean_dets)
        to_remove = set()
        for i, idx in enumerate(clean_frames):
            cx, cy, cf = clean_dets[idx]
            if cf >= hard_reset_conf:
                continue
            prev_speed = 1e9
            if i > 0:
                pidx = clean_frames[i-1]
                pcx, pcy, _ = clean_dets[pidx]
                prev_speed = dist((cx,cy),(pcx,pcy)) / max(idx-pidx, 1)
            next_speed = 1e9
            if i < len(clean_frames)-1:
                nidx = clean_frames[i+1]
                ncx, ncy, _ = clean_dets[nidx]
                next_speed = dist((cx,cy),(ncx,ncy)) / max(nidx-idx, 1)
            if prev_speed > speed_limit and next_speed > speed_limit:
                to_remove.add(idx)
        for idx in to_remove:
            del clean_dets[idx]
        if not to_remove:
            break

    # Stage 2: A→B→C filter (2 passes)
    for _ in range(2):
        clean_frames = sorted(clean_dets)
        to_remove = set()
        for i, idx in enumerate(clean_frames):
            bx, by, bcf = clean_dets[idx]
            if bcf >= hard_reset_conf:
                continue
            if i == 0 or i == len(clean_frames)-1:
                continue
            aidx = clean_frames[i-1]
            cidx = clean_frames[i+1]
            ax, ay, _ = clean_dets[aidx]
            cx2, cy2, _ = clean_dets[cidx]

            # Speed A→B (per frame)
            dt_ab = max(idx - aidx, 1)
            speed_ab = dist((bx,by),(ax,ay)) / dt_ab

            if speed_ab < speed_limit * 0.4:
                continue  # B is not fast from A, skip

            dca = dist((cx2,cy2),(ax,ay))
            dcb = dist((cx2,cy2),(bx,by))

            # If C is much closer to A than to B: B is a false positive
            if dca < abc_ratio * dcb:
                to_remove.add(idx)

        for idx in to_remove:
            del clean_dets[idx]
        if not to_remove:
            break

    return clean_dets


# ─────────────────────────────────────────────────────────────
# Pass 2: gap filling with running reference
# ─────────────────────────────────────────────────────────────

def fill_gaps(n_frames, clean_dets, cache, seed_frame, seed_pos,
              interp_max_gap=20,
              y_off=51.0,
              x_off=0.0,
              person_accept_r=150.0,
              person_conf_min=0.30):
    positions = {}
    anchors = dict(clean_dets)
    anchors[seed_frame] = (float(seed_pos[0]), float(seed_pos[1]), 1.0)
    anchor_frames = sorted(anchors)

    # Running reference for person estimation (updated each frame)
    running_ref = (float(seed_pos[0]), float(seed_pos[1]))

    for idx in range(n_frames):
        # Direct anchor?
        if idx in anchors:
            cx, cy, _ = anchors[idx]
            positions[idx] = (int(cx), int(cy))
            running_ref = (cx, cy)
            continue

        # Find bracketing anchors
        prev_f = next_f = None
        for f in reversed([f for f in anchor_frames if f < idx]):
            prev_f = f; break
        for f in [f for f in anchor_frames if f > idx]:
            next_f = f; break

        # Linear interpolation if gap is small
        if prev_f is not None and next_f is not None:
            gap = next_f - prev_f
            if gap <= interp_max_gap:
                t = (idx - prev_f) / gap
                px_a, py_a, _ = anchors[prev_f]
                px_b, py_b, _ = anchors[next_f]
                x = px_a + t * (px_b - px_a)
                y = py_a + t * (py_b - py_a)
                positions[idx] = (int(x), int(y))
                running_ref = (x, y)
                continue

        # Person-foot estimation using running_ref
        persons = cache.get(idx, {}).get('persons', [])
        best_person = None
        best_d = 1e9
        for pcx, pcy, pcf, x1, y1, x2, y2 in persons:
            if pcf < person_conf_min:
                continue
            feet_y = pcy - y_off
            feet_x = pcx + x_off
            d = dist((feet_x, feet_y), running_ref)
            if d < person_accept_r and d < best_d:
                best_d = d
                best_person = (feet_x, feet_y)

        if best_person is not None:
            fx, fy = best_person
            bx = int(clamp(fx, 0, W-1))
            by = int(clamp(fy, 0, H-1))
            positions[idx] = (bx, by)
            running_ref = (float(bx), float(by))
            continue

        # Hold last known
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
# Pass 3: Kalman smoother within segments
# ─────────────────────────────────────────────────────────────

def kalman_smooth_segments(positions, clean_dets, seed_frame, seed_pos,
                            n_frames, kalman_q=1.0, r_det=5.0, r_fill=50.0):
    kf_pos = {}
    all_anchors = dict(clean_dets)
    all_anchors[seed_frame] = (float(seed_pos[0]), float(seed_pos[1]), 1.0)

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
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r_fill
            meas = np.array([[float(fx)], [float(fy)]], dtype=np.float32)
            kf.correct(meas)
            state = kf.statePost
            sx = int(clamp(float(state[0,0]), 0, W-1))
            sy = int(clamp(float(state[1,0]), 0, H-1))
            kf_pos[idx] = (sx, sy)
        else:
            kf.statePost[2,0] *= 0.85
            kf.statePost[3,0] *= 0.85
            kf_pos[idx] = (int(clamp(px, 0, W-1)), int(clamp(py, 0, H-1)))

    return kf_pos


# ─────────────────────────────────────────────────────────────
# Full tracker
# ─────────────────────────────────────────────────────────────

def track_v7b(frames, seed_frame, seed_pos, cache,
              min_conf=0.10,
              hard_reset_conf=0.40,
              speed_limit=250.0,
              abc_ratio=0.6,
              interp_max_gap=40,
              y_off=51.0,
              x_off=0.0,
              person_accept_r=150.0,
              person_conf_min=0.30,
              use_kalman_smooth=True,
              kalman_q=1.0):
    n = len(frames)

    clean_dets = collect_detections(
        cache, n,
        min_conf=min_conf,
        hard_reset_conf=hard_reset_conf,
        speed_limit=speed_limit,
        abc_ratio=abc_ratio)

    positions = fill_gaps(
        n, clean_dets, cache, seed_frame, seed_pos,
        interp_max_gap=interp_max_gap,
        y_off=y_off, x_off=x_off,
        person_accept_r=person_accept_r,
        person_conf_min=person_conf_min)

    if use_kalman_smooth:
        positions = kalman_smooth_segments(
            positions, clean_dets, seed_frame, seed_pos,
            n, kalman_q=kalman_q)

    return positions


# ─────────────────────────────────────────────────────────────
# Evaluation
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
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video',       required=True)
    ap.add_argument('--labels',      required=True)
    ap.add_argument('--frames-dir',  default='')
    ap.add_argument('--render',      action='store_true')
    args = ap.parse_args()

    with open(args.labels) as f:
        data = json.load(f)
    labels = data['labels']
    lmap   = {lb['frame']: lb for lb in labels}
    print(f"Loaded {len(labels)} labels")

    frames_dir = args.frames_dir or '/tmp/soccer_v2_frames'
    frames = load_frames(frames_dir)
    print(f"Using {len(frames)} frames from {frames_dir}")

    print(f"Loading YOLO cache from {YOLO_CACHE_FILE}...")
    cache = load_cache(YOLO_CACHE_FILE)
    print(f"  Loaded {len(cache)} frames")

    seed_pos   = (int(lmap[0]['px']), int(lmap[0]['py'])) if 0 in lmap else (W//2, H//2)
    seed_frame = 0

    # Diagnostic: how many frames remain after ABC filter
    clean_baseline = collect_detections(
        cache, len(frames), min_conf=0.10, hard_reset_conf=0.40,
        speed_limit=250.0, abc_ratio=0.6)
    labeled_clean = sum(1 for lb in labels if lb['frame'] in clean_baseline)
    print(f"  Clean dets in labeled frames (baseline): {labeled_clean}/{len(labels)}")

    # Baseline
    print("\n--- Baseline v7b ---")
    pos = track_v7b(frames, seed_frame, seed_pos, cache)
    tr, w30, w50, med, sc = evaluate(pos, labels)
    print(f"  W30={w30}  W50={w50}  Med={med:.1f}  Score={sc:.4f}")

    # Grid search
    print("\n--- Grid search v7b ---")

    hand_tuned = [
        dict(min_conf=0.10, hard_reset_conf=0.40, speed_limit=250, abc_ratio=0.6,
             interp_max_gap=80, y_off=51, x_off=0, person_accept_r=150,
             person_conf_min=0.30, use_kalman_smooth=True, kalman_q=1.0),
        dict(min_conf=0.10, hard_reset_conf=0.40, speed_limit=200, abc_ratio=0.5,
             interp_max_gap=100, y_off=51, x_off=0, person_accept_r=120,
             person_conf_min=0.30, use_kalman_smooth=True, kalman_q=0.5),
        dict(min_conf=0.08, hard_reset_conf=0.35, speed_limit=300, abc_ratio=0.7,
             interp_max_gap=60, y_off=48, x_off=0, person_accept_r=180,
             person_conf_min=0.25, use_kalman_smooth=True, kalman_q=1.5),
        dict(min_conf=0.12, hard_reset_conf=0.45, speed_limit=200, abc_ratio=0.55,
             interp_max_gap=80, y_off=54, x_off=0, person_accept_r=100,
             person_conf_min=0.35, use_kalman_smooth=False, kalman_q=1.0),
        dict(min_conf=0.15, hard_reset_conf=0.50, speed_limit=180, abc_ratio=0.6,
             interp_max_gap=100, y_off=51, x_off=0, person_accept_r=200,
             person_conf_min=0.25, use_kalman_smooth=True, kalman_q=0.3),
        dict(min_conf=0.10, hard_reset_conf=0.40, speed_limit=250, abc_ratio=0.6,
             interp_max_gap=80, y_off=58, x_off=0, person_accept_r=100,
             person_conf_min=0.30, use_kalman_smooth=True, kalman_q=2.0),
        dict(min_conf=0.30, hard_reset_conf=0.50, speed_limit=150, abc_ratio=0.5,
             interp_max_gap=100, y_off=58, x_off=0, person_accept_r=100,
             person_conf_min=0.30, use_kalman_smooth=True, kalman_q=2.0),
        dict(min_conf=0.10, hard_reset_conf=0.40, speed_limit=250, abc_ratio=0.4,
             interp_max_gap=120, y_off=51, x_off=0, person_accept_r=150,
             person_conf_min=0.30, use_kalman_smooth=True, kalman_q=1.0),
    ]

    random_combos = []
    for _ in range(72):
        random_combos.append(dict(
            min_conf        = random.choice([0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]),
            hard_reset_conf = random.choice([0.25, 0.30, 0.40, 0.50, 0.60]),
            speed_limit     = random.choice([120, 150, 200, 250, 300, 400]),
            abc_ratio       = random.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            interp_max_gap  = random.choice([20, 40, 60, 80, 100, 120]),
            y_off           = random.choice([44, 47, 51, 54, 58, 65]),
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
        pos = track_v7b(frames, seed_frame, seed_pos, cache, **params)
        tr, w30, w50, med, sc = evaluate(pos, labels)
        tag = (f"mc={params['min_conf']:.2f} sl={params['speed_limit']:.0f} "
               f"abc={params['abc_ratio']:.1f} ig={params['interp_max_gap']} "
               f"yo={params['y_off']} par={params['person_accept_r']}")
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
    print(f"  Score={sc:.4f}  W30={w30}  W50={w50}  Median={med:.1f}px")
    print(f"{'='*80}")

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

    with open('tracking_analysis_v7b.txt', 'w') as f:
        f.write(f"V7b Winner: score={sc:.4f} w30={w30} w50={w50} med={med:.1f}\n")
        f.write(f"Params: {best_params}\n\n")
        for s, w3, w5, md, t, tg, _, _ in results_log[:20]:
            f.write(f"{s:.4f}  W30={w3}  W50={w5}  Med={md:.1f}  {tg}\n")
    print("\nSaved to tracking_analysis_v7b.txt")

    if args.render:
        render_video(frames, best_pos, labels, 'tracking_v7b_winner.mp4')


if __name__ == '__main__':
    main()
