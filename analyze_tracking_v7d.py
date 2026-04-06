#!/usr/bin/env python3
"""
Ball tracking v7d — v7c + HC-boundary gap filter.

Key fix vs v7c:
  The gap consistency filter in v7c used ALL consecutive anchor pairs as
  gap boundaries. But false-positive chains form their own small consecutive
  anchors (5-15 frames apart) so the gap filter never triggers.

  Fix: use ONLY HIGH-CONFIDENCE (conf >= hc_thresh) detections as gap boundaries.
  Any detection (even high-conf) between two HC anchors that deviates >gap_margin
  from the linear interpolation is removed.

  This breaks FP chains like: [310✓] → [315 FP] → [320 FP] → ... → [380✓]
  because 310 and 380 are HC anchors and the FPs are outside the band.

Uses yolo_cache_v7.json (already computed).
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
# Pass 1: Trajectory-guided collection
# ─────────────────────────────────────────────────────────────

def collect_trajectory_guided(cache, n_frames, seed_pos,
                               min_conf=0.10,
                               hard_reset_conf=0.40,
                               speed_limit=250.0,
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


# ─────────────────────────────────────────────────────────────
# Pass 1b: ABC filter
# ─────────────────────────────────────────────────────────────

def abc_filter(dets, hard_reset_conf=0.40, speed_limit=250.0, abc_ratio=0.6):
    clean = dict(dets)
    for _ in range(3):
        frames = sorted(clean)
        to_remove = set()
        for i, idx in enumerate(frames):
            bx, by, bcf = clean[idx]
            if bcf >= hard_reset_conf:
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


# ─────────────────────────────────────────────────────────────
# Pass 1c: HC-boundary gap consistency filter
# ─────────────────────────────────────────────────────────────

def hc_gap_filter(clean_dets, seed_frame, seed_pos,
                  hc_thresh=0.40,
                  gap_margin=140.0,
                  min_gap_for_filter=15):
    """
    Use only HIGH-CONFIDENCE detections (conf >= hc_thresh) as gap boundaries.
    ANY detection between two HC anchors that deviates > gap_margin from the
    linear interpolation between those HC anchors is removed.

    This breaks slowly-drifting FP chains (each step < speed_limit from prev)
    that would never be caught by the local ABC filter.
    """
    filtered = dict(clean_dets)

    changed = True
    while changed:
        changed = False

        # Identify HC anchors
        hc_anchors = {f: v for f, v in filtered.items() if v[2] >= hc_thresh}
        hc_anchors[seed_frame] = (float(seed_pos[0]), float(seed_pos[1]), 1.0)
        hc_frames = sorted(hc_anchors)

        for i in range(len(hc_frames)-1):
            f_a = hc_frames[i]
            f_b = hc_frames[i+1]
            gap = f_b - f_a
            if gap < min_gap_for_filter:
                continue

            ax, ay, _ = hc_anchors[f_a]
            bx, by, _ = hc_anchors[f_b]

            to_remove = []
            for f in list(filtered.keys()):
                if f <= f_a or f >= f_b:
                    continue
                # Even high-conf dets can be removed here
                cx, cy, cf = filtered[f]
                t = (f - f_a) / gap
                ex = ax + t * (bx - ax)
                ey = ay + t * (by - ay)
                if dist((cx, cy), (ex, ey)) > gap_margin:
                    to_remove.append(f)

            for f in to_remove:
                del filtered[f]
                changed = True

    return filtered


# ─────────────────────────────────────────────────────────────
# Pass 2: Gap filling with running reference
# ─────────────────────────────────────────────────────────────

def fill_gaps(n_frames, clean_dets, cache, seed_frame, seed_pos,
              interp_max_gap=80,
              y_off=51.0,
              x_off=0.0,
              person_accept_r=150.0,
              person_conf_min=0.30):
    positions = {}
    anchors = dict(clean_dets)
    anchors[seed_frame] = (float(seed_pos[0]), float(seed_pos[1]), 1.0)
    anchor_frames = sorted(anchors)
    running_ref = (float(seed_pos[0]), float(seed_pos[1]))

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

        # Person estimation
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
# Pass 3: Kalman smoother
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
            kf_pos[idx] = (int(clamp(float(state[0,0]), 0, W-1)),
                           int(clamp(float(state[1,0]), 0, H-1)))
        else:
            kf.statePost[2,0] *= 0.85
            kf.statePost[3,0] *= 0.85
            kf_pos[idx] = (int(clamp(px, 0, W-1)), int(clamp(py, 0, H-1)))

    return kf_pos


# ─────────────────────────────────────────────────────────────
# Full tracker
# ─────────────────────────────────────────────────────────────

def track_v7d(frames, seed_frame, seed_pos, cache,
              min_conf=0.10,
              hard_reset_conf=0.40,
              speed_limit=250.0,
              vel_decay=0.85,
              abc_ratio=0.6,
              hc_thresh=0.40,
              gap_margin=140.0,
              min_gap_for_filter=15,
              interp_max_gap=80,
              y_off=51.0,
              x_off=0.0,
              person_accept_r=150.0,
              person_conf_min=0.30,
              use_kalman_smooth=True,
              kalman_q=1.0):
    n = len(frames)

    raw_dets = collect_trajectory_guided(
        cache, n, seed_pos,
        min_conf=min_conf,
        hard_reset_conf=hard_reset_conf,
        speed_limit=speed_limit,
        vel_decay=vel_decay)

    clean_dets = abc_filter(raw_dets, hard_reset_conf=hard_reset_conf,
                             speed_limit=speed_limit, abc_ratio=abc_ratio)

    clean_dets = hc_gap_filter(
        clean_dets, seed_frame, seed_pos,
        hc_thresh=hc_thresh,
        gap_margin=gap_margin,
        min_gap_for_filter=min_gap_for_filter)

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

    seed_pos   = (int(lmap[0]['px']), int(lmap[0]['py'])) if 0 in lmap else (W//2, H//2)
    seed_frame = 0

    print("\n--- Baseline v7d ---")
    pos = track_v7d(frames, seed_frame, seed_pos, cache)
    tr, w30, w50, med, sc = evaluate(pos, labels)
    print(f"  W30={w30}  W50={w50}  Med={med:.1f}  Score={sc:.4f}")

    print("\n--- Grid search v7d ---")

    hand_tuned = [
        # Core: HC boundary filter, moderate margin
        dict(min_conf=0.10, hard_reset_conf=0.40, speed_limit=250, vel_decay=0.85,
             abc_ratio=0.6, hc_thresh=0.40, gap_margin=140, min_gap_for_filter=15,
             interp_max_gap=80, y_off=51, x_off=0, person_accept_r=150,
             person_conf_min=0.30, use_kalman_smooth=True, kalman_q=1.0),
        # Tighter margin
        dict(min_conf=0.10, hard_reset_conf=0.40, speed_limit=200, vel_decay=0.85,
             abc_ratio=0.5, hc_thresh=0.40, gap_margin=120, min_gap_for_filter=12,
             interp_max_gap=100, y_off=51, x_off=0, person_accept_r=120,
             person_conf_min=0.30, use_kalman_smooth=True, kalman_q=0.5),
        # Lower hc_thresh (more anchors)
        dict(min_conf=0.10, hard_reset_conf=0.35, speed_limit=250, vel_decay=0.85,
             abc_ratio=0.6, hc_thresh=0.30, gap_margin=130, min_gap_for_filter=15,
             interp_max_gap=80, y_off=51, x_off=0, person_accept_r=150,
             person_conf_min=0.25, use_kalman_smooth=True, kalman_q=1.0),
        # High hc_thresh (fewer, more trusted anchors)
        dict(min_conf=0.10, hard_reset_conf=0.50, speed_limit=250, vel_decay=0.85,
             abc_ratio=0.6, hc_thresh=0.55, gap_margin=150, min_gap_for_filter=20,
             interp_max_gap=100, y_off=51, x_off=0, person_accept_r=150,
             person_conf_min=0.30, use_kalman_smooth=True, kalman_q=0.5),
        # Wide interp + person
        dict(min_conf=0.08, hard_reset_conf=0.35, speed_limit=300, vel_decay=0.80,
             abc_ratio=0.7, hc_thresh=0.40, gap_margin=150, min_gap_for_filter=20,
             interp_max_gap=60, y_off=48, x_off=0, person_accept_r=180,
             person_conf_min=0.25, use_kalman_smooth=True, kalman_q=1.5),
        # y_off=58 for person est
        dict(min_conf=0.10, hard_reset_conf=0.40, speed_limit=250, vel_decay=0.85,
             abc_ratio=0.6, hc_thresh=0.40, gap_margin=140, min_gap_for_filter=15,
             interp_max_gap=80, y_off=58, x_off=0, person_accept_r=100,
             person_conf_min=0.30, use_kalman_smooth=True, kalman_q=2.0),
        # Very wide gap_margin (accept most, rely on trajectory guidance)
        dict(min_conf=0.30, hard_reset_conf=0.50, speed_limit=150, vel_decay=0.85,
             abc_ratio=0.5, hc_thresh=0.50, gap_margin=200, min_gap_for_filter=15,
             interp_max_gap=120, y_off=58, x_off=0, person_accept_r=100,
             person_conf_min=0.30, use_kalman_smooth=True, kalman_q=2.0),
        # No Kalman smooth
        dict(min_conf=0.10, hard_reset_conf=0.40, speed_limit=250, vel_decay=0.85,
             abc_ratio=0.4, hc_thresh=0.40, gap_margin=140, min_gap_for_filter=15,
             interp_max_gap=100, y_off=51, x_off=0, person_accept_r=150,
             person_conf_min=0.30, use_kalman_smooth=False, kalman_q=1.0),
    ]

    random_combos = []
    for _ in range(72):
        random_combos.append(dict(
            min_conf        = random.choice([0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]),
            hard_reset_conf = random.choice([0.25, 0.30, 0.40, 0.50, 0.60]),
            speed_limit     = random.choice([120, 150, 200, 250, 300, 400]),
            vel_decay       = random.choice([0.75, 0.80, 0.85, 0.90, 0.95]),
            abc_ratio       = random.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            hc_thresh       = random.choice([0.25, 0.30, 0.35, 0.40, 0.50, 0.60]),
            gap_margin      = random.choice([80, 100, 120, 140, 160, 200]),
            min_gap_for_filter = random.choice([8, 10, 15, 20, 25]),
            interp_max_gap  = random.choice([40, 60, 80, 100, 120, 150]),
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
        pos = track_v7d(frames, seed_frame, seed_pos, cache, **params)
        tr, w30, w50, med, sc = evaluate(pos, labels)
        tag = (f"mc={params['min_conf']:.2f} sl={params['speed_limit']:.0f} "
               f"hc={params['hc_thresh']:.2f} gm={params['gap_margin']:.0f} "
               f"ig={params['interp_max_gap']} yo={params['y_off']}")
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

    with open('tracking_analysis_v7d.txt', 'w') as f:
        f.write(f"V7d Winner: score={sc:.4f} w30={w30} w50={w50} med={med:.1f}\n")
        f.write(f"Params: {best_params}\n\n")
        for s, w3, w5, md, t, tg, _, _ in results_log[:20]:
            f.write(f"{s:.4f}  W30={w3}  W50={w5}  Med={md:.1f}  {tg}\n")
    print("\nSaved to tracking_analysis_v7d.txt")

    if args.render:
        render_video(frames, best_pos, labels, 'tracking_v7d_winner.mp4')


if __name__ == '__main__':
    main()
