#!/usr/bin/env python3
"""
Ball tracking v9 — TrackNet detections + 15fps.

Differences from v8:
  • Ball detections come from TrackNet (tracknet_cache_15fps.json) instead of YOLO.
    TrackNet sees 3 consecutive frames at once, exploiting motion blur and
    temporal context — dramatically reducing missed detections in fast-play.
  • Person detections still come from YOLO (yolo_cache_test1_15fps.json).
  • Frames are extracted at 15fps (frames_15fps/) for tighter temporal resolution.
  • Labels are remapped from original 10fps timestamps to 15fps frame indices.
  • All world-space thresholds are unchanged (meters); speed_limit search range
    is scaled to 15fps (m/frame).

Prerequisites:
  1. .venv/bin/python3 build_cache_15fps.py
  2. .venv/bin/python3 tracknet.py train
  3. .venv/bin/python3 tracknet.py infer

Usage:
  .venv/bin/python3 analyze_tracking_v9.py \\
      --video "test 1.mp4" \\
      --labels ball_labels.json \\
      --tracknet_cache tracknet_cache_15fps.json \\
      --yolo_cache yolo_cache_test1_15fps.json \\
      --calibration calibration_test1.json \\
      [--render] [--random-seeds 40]
"""

import argparse
import json
import os
import random
import subprocess
import sys

import cv2
import numpy as np

# Import all tracker logic from v8 — only data loading and main() differ.
sys.path.insert(0, os.path.dirname(__file__))
from analyze_tracking_v8 import (
    FieldHomography, dist, clamp,
    load_frames, load_cache,
    collect_trajectory_guided, hc_override, abc_filter,
    validate_hc_chain, hc_gap_filter, fill_gaps, kalman_smooth,
    track_v8, track_v8_backward, merge_bidirectional, track_bidir,
    evaluate, render_video, find_best_seed_in_window,
)

FPS = 15.0
FRAMES_DIR = 'frames_15fps'


# ── 15fps helpers ─────────────────────────────────────────────────────────────

def remap_labels_15fps(labels_data, fps=FPS):
    """
    Remap labels from their original timestamps to 15fps frame indices.

    Original labels store both 'frame' (at 10fps) and 't' (seconds).
    We use 't' so the mapping is FPS-independent.
    """
    remapped = []
    for lb in labels_data['labels']:
        f15 = int(round(lb['t'] * fps))
        remapped.append({
            'frame': f15,
            't':     lb['t'],
            'px':    lb['px'],
            'py':    lb['py'],
        })
    # Sort by frame index and deduplicate (keep first if collision)
    seen = set()
    out  = []
    for lb in sorted(remapped, key=lambda x: x['frame']):
        if lb['frame'] not in seen:
            seen.add(lb['frame'])
            out.append(lb)
    return out


def build_merged_cache(tracknet_cache, yolo_cache):
    """
    Merge TrackNet ball detections with YOLO person detections.

    TrackNet cache has 'balls' keys only.
    YOLO cache has both 'balls' and 'persons'; we take persons from YOLO.
    """
    all_frames = set(tracknet_cache.keys()) | set(yolo_cache.keys())
    merged = {}
    for f in all_frames:
        balls   = tracknet_cache.get(f, {}).get('balls',   [])
        persons = yolo_cache.get(f, {}).get('persons', [])
        merged[f] = {'balls': balls, 'persons': persons}
    return merged


def ensure_frames(video, frames_dir):
    """Extract frames at 15fps if the directory is empty."""
    if os.path.isdir(frames_dir) and os.listdir(frames_dir):
        return
    os.makedirs(frames_dir, exist_ok=True)
    print(f"Extracting frames from '{video}' at {FPS}fps → {frames_dir}/")
    subprocess.run([
        'ffmpeg', '-i', video, '-vf', f'fps={FPS}',
        '-q:v', '2', '-loglevel', 'error',
        os.path.join(frames_dir, 'frame_%06d.jpg')
    ], check=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video',           required=True)
    ap.add_argument('--labels',          required=True)
    ap.add_argument('--tracknet_cache',  required=True,
                    help='Ball detections from TrackNet (tracknet_cache_15fps.json)')
    ap.add_argument('--yolo_cache',      required=True,
                    help='Person detections from YOLO (yolo_cache_test1_15fps.json)')
    ap.add_argument('--calibration',     required=True)
    ap.add_argument('--frames-dir',      default=FRAMES_DIR)
    ap.add_argument('--render',          action='store_true')
    ap.add_argument('--output',          default='tracking_v9_winner.mp4')
    ap.add_argument('--analysis-out',    default='tracking_analysis_v9.txt')
    ap.add_argument('--random-seeds',    type=int, default=40)
    args = ap.parse_args()

    # ── Labels (remapped to 15fps) ─────────────────────────────────────────
    with open(args.labels) as f:
        labels_data = json.load(f)
    labels = remap_labels_15fps(labels_data)
    print(f'Loaded {len(labels)} labels → remapped to {FPS}fps frame indices')
    print(f'  First: frame {labels[0]["frame"]}  (t={labels[0]["t"]}s)')
    print(f'  Last:  frame {labels[-1]["frame"]}  (t={labels[-1]["t"]}s)')

    # ── Frames ────────────────────────────────────────────────────────────
    frames_dir = args.frames_dir
    ensure_frames(args.video, frames_dir)
    frames = load_frames(frames_dir)
    print(f'Using {len(frames)} frames at {FPS}fps')

    # ── Caches ────────────────────────────────────────────────────────────
    print(f'Loading TrackNet cache: {args.tracknet_cache}')
    tracknet_cache = load_cache(args.tracknet_cache)
    print(f'  {len(tracknet_cache)} frames')

    print(f'Loading YOLO cache (persons): {args.yolo_cache}')
    yolo_cache = load_cache(args.yolo_cache)
    print(f'  {len(yolo_cache)} frames')

    cache = build_merged_cache(tracknet_cache, yolo_cache)
    print(f'Merged cache: {len(cache)} frames')

    ball_det_count = sum(len(v['balls']) for v in cache.values())
    high_conf      = sum(1 for v in cache.values()
                         for b in v['balls'] if b[2] >= 0.5)
    print(f'Ball detections: {ball_det_count} total, {high_conf} with conf>=0.5')

    # ── Calibration ───────────────────────────────────────────────────────
    print(f'Loading calibration: {args.calibration}')
    hom = FieldHomography(args.calibration)
    print(f'  Format: {hom.fmt}  Box: {hom.box}')

    # ── Seed ──────────────────────────────────────────────────────────────
    first       = labels[0]
    seed_frame  = first['frame']
    seed_pos    = (int(first['px']), int(first['py']))
    seed_world  = hom.to_world(*seed_pos)
    ppm_seed    = hom.local_ppm(*seed_pos)
    print(f'\nSeed: frame {seed_frame}  pixel={seed_pos}  '
          f'world=({seed_world[0]:.2f},{seed_world[1]:.2f})m  '
          f'scale={ppm_seed:.1f}px/m')

    # Backward seed: last labeled frame
    last_lb        = labels[-1]
    bwd_seed_frame = last_lb['frame']
    bwd_seed_pos   = (int(last_lb['px']), int(last_lb['py']))
    print(f'Backward seed: frame {bwd_seed_frame}')

    # ── Parameter search ──────────────────────────────────────────────────
    # Speed limits are in m/frame. At 15fps a 30 m/s kick = 2.0 m/frame.
    # A very fast kick (50 m/s) = 3.3 m/frame. Upper bound of 4.0 is generous.
    print('\n--- Parameter search v9 (TrackNet + 15fps) ---')

    hand_tuned = [
        # Baseline for 15fps (speed_limit scaled from v8's 3.0 @ 10fps)
        dict(min_conf=0.10, hard_reset_conf=0.50, abc_protect_conf=0.65,
             speed_limit=2.0, vel_decay=0.85, abc_ratio=0.6,
             hc_thresh=0.55, chain_margin=5.0, gap_margin=3.0,
             min_gap_for_filter=6, interp_max_gap=150,
             person_accept_r=3.0, person_conf_min=0.30,
             use_kalman_smooth=True, kalman_q_m=0.1),
        # Tighter — trust TrackNet more, fewer person fallbacks
        dict(min_conf=0.15, hard_reset_conf=0.50, abc_protect_conf=0.65,
             speed_limit=1.5, vel_decay=0.85, abc_ratio=0.6,
             hc_thresh=0.55, chain_margin=4.0, gap_margin=2.5,
             min_gap_for_filter=6, interp_max_gap=120,
             person_accept_r=2.5, person_conf_min=0.30,
             use_kalman_smooth=True, kalman_q_m=0.08),
        # Looser margins — accept more of TrackNet's softer detections
        dict(min_conf=0.07, hard_reset_conf=0.45, abc_protect_conf=0.60,
             speed_limit=3.0, vel_decay=0.85, abc_ratio=0.55,
             hc_thresh=0.50, chain_margin=7.0, gap_margin=5.0,
             min_gap_for_filter=5, interp_max_gap=200,
             person_accept_r=4.0, person_conf_min=0.25,
             use_kalman_smooth=True, kalman_q_m=0.15),
        # High hc_thresh — only very confident TrackNet peaks as anchors
        dict(min_conf=0.10, hard_reset_conf=0.50, abc_protect_conf=0.65,
             speed_limit=2.0, vel_decay=0.85, abc_ratio=0.6,
             hc_thresh=0.70, chain_margin=5.0, gap_margin=3.0,
             min_gap_for_filter=6, interp_max_gap=150,
             person_accept_r=3.0, person_conf_min=0.30,
             use_kalman_smooth=True, kalman_q_m=0.1),
        # Wide person fallback
        dict(min_conf=0.10, hard_reset_conf=0.50, abc_protect_conf=0.65,
             speed_limit=2.0, vel_decay=0.85, abc_ratio=0.6,
             hc_thresh=0.55, chain_margin=5.0, gap_margin=3.0,
             min_gap_for_filter=6, interp_max_gap=150,
             person_accept_r=6.0, person_conf_min=0.25,
             use_kalman_smooth=True, kalman_q_m=0.1),
        # No Kalman
        dict(min_conf=0.10, hard_reset_conf=0.50, abc_protect_conf=0.65,
             speed_limit=2.0, vel_decay=0.85, abc_ratio=0.6,
             hc_thresh=0.55, chain_margin=5.0, gap_margin=3.0,
             min_gap_for_filter=6, interp_max_gap=150,
             person_accept_r=3.0, person_conf_min=0.30,
             use_kalman_smooth=False, kalman_q_m=0.1),
        # Very fast kicks allowed
        dict(min_conf=0.08, hard_reset_conf=0.50, abc_protect_conf=0.60,
             speed_limit=4.0, vel_decay=0.85, abc_ratio=0.5,
             hc_thresh=0.50, chain_margin=8.0, gap_margin=6.0,
             min_gap_for_filter=4, interp_max_gap=200,
             person_accept_r=5.0, person_conf_min=0.25,
             use_kalman_smooth=True, kalman_q_m=0.2),
        # Tight Kalman
        dict(min_conf=0.10, hard_reset_conf=0.50, abc_protect_conf=0.65,
             speed_limit=2.0, vel_decay=0.85, abc_ratio=0.6,
             hc_thresh=0.55, chain_margin=5.0, gap_margin=3.0,
             min_gap_for_filter=6, interp_max_gap=150,
             person_accept_r=3.0, person_conf_min=0.30,
             use_kalman_smooth=True, kalman_q_m=0.03),
        # Large interp gap — trust interpolation over person fallback
        dict(min_conf=0.10, hard_reset_conf=0.50, abc_protect_conf=0.65,
             speed_limit=2.0, vel_decay=0.85, abc_ratio=0.6,
             hc_thresh=0.55, chain_margin=5.0, gap_margin=3.0,
             min_gap_for_filter=6, interp_max_gap=300,
             person_accept_r=3.0, person_conf_min=0.30,
             use_kalman_smooth=True, kalman_q_m=0.1),
        # v8 winner params carried forward (speed_limit scaled 10→15fps)
        dict(min_conf=0.08, hard_reset_conf=0.50, abc_protect_conf=0.65,
             speed_limit=4.449981720664334 * (10/15),
             vel_decay=0.8886399876164743, abc_ratio=0.5072620179910261,
             hc_thresh=0.50, chain_margin=6.927751904241679,
             gap_margin=3.30524154160048, min_gap_for_filter=4,
             interp_max_gap=150, person_accept_r=6.522248683020497,
             person_conf_min=0.35, use_kalman_smooth=True,
             kalman_q_m=0.26661925914762064),
    ]

    rng = random.Random(42)
    random_combos = []
    for _ in range(args.random_seeds):
        random_combos.append(dict(
            min_conf          = rng.choice([0.05, 0.07, 0.10, 0.12, 0.15]),
            hard_reset_conf   = rng.choice([0.40, 0.45, 0.50, 0.55]),
            abc_protect_conf  = rng.choice([0.55, 0.60, 0.65, 0.70]),
            speed_limit       = rng.uniform(1.0, 4.0),    # m/frame @ 15fps
            vel_decay         = rng.uniform(0.78, 0.95),
            abc_ratio         = rng.uniform(0.40, 0.70),
            hc_thresh         = rng.choice([0.45, 0.50, 0.55, 0.60, 0.65, 0.70]),
            chain_margin      = rng.uniform(2.5, 10.0),
            gap_margin        = rng.uniform(1.5, 7.0),
            min_gap_for_filter = rng.choice([4, 5, 6, 8]),
            interp_max_gap    = rng.choice([100, 120, 150, 200, 250, 300]),
            person_accept_r   = rng.uniform(1.5, 8.0),
            person_conf_min   = rng.choice([0.20, 0.25, 0.30, 0.35]),
            use_kalman_smooth = True,
            kalman_q_m        = rng.uniform(0.02, 0.40),
        ))

    all_combos  = hand_tuned + random_combos
    results_log = []
    best_score  = -1
    best_pos    = None
    best_params = None

    for i, params in enumerate(all_combos):
        tag = f'hand_{i}' if i < len(hand_tuned) else f'rand_{i - len(hand_tuned)}'
        pos = track_bidir(frames, seed_frame, seed_pos, cache, hom,
                          bwd_seed_frame=bwd_seed_frame,
                          bwd_seed_pos=bwd_seed_pos,
                          **params)
        tr, w30, w50, med, sc = evaluate(pos, labels)
        results_log.append((sc, w30, w50, med, tr, tag, params, pos))
        if sc > best_score:
            best_score  = sc
            best_params = params
            best_pos    = pos
        if (i + 1) % 10 == 0:
            top = max(results_log, key=lambda x: x[0])
            print(f'  [{i+1}/{len(all_combos)}] '
                  f'best: score={top[0]:.4f} w30={top[1]} w50={top[2]} med={top[3]:.0f}px')

    results_log.sort(key=lambda x: x[0], reverse=True)

    print(f'\nTop 5:')
    for sc, w30, w50, med, tr, tag, params, _ in results_log[:5]:
        print(f'  {tag}  score={sc:.4f}  W30={w30:3d}  W50={w50:3d}  Med={med:.0f}px')

    sc, w30, w50, med, tr, tag, best_params, best_pos = results_log[0]
    print(f'\nWINNER: score={sc:.4f}  W30={w30}/{len(labels)}  '
          f'W50={w50}/{len(labels)}  Median={med:.1f}px')
    print(f'  Params: {best_params}')

    # Per-frame misses
    miss_count = 0
    print('\nPer-frame errors > 50px (winner):')
    for lb in labels:
        f = lb['frame']
        if f not in best_pos:
            print(f'  f{f:4d}  NO_DET')
            miss_count += 1
            continue
        px, py = best_pos[f]
        err = dist((px, py), (lb['px'], lb['py']))
        if err > 50:
            print(f'  f{f:4d}  err={err:.0f}px  '
                  f'gt=({lb["px"]},{lb["py"]})  pred=({px},{py})')
            miss_count += 1
    print(f'  Total misses > 50px: {miss_count}/{len(labels)}')

    # Save analysis
    with open(args.analysis_out, 'w') as f:
        f.write(f'V9 Winner: score={sc:.4f} w30={w30} w50={w50} med={med:.1f}\n')
        f.write(f'Params: {best_params}\n\n')
        for sc2, w30_2, w50_2, md, tr2, tg, p, _ in results_log:
            f.write(f'{sc2:.4f}  W30={w30_2}  W50={w50_2}  Med={md:.1f}  {tg}\n')
    print(f'Saved {args.analysis_out}')

    if args.render:
        render_video(frames, best_pos, labels, args.output, fps=int(FPS))


if __name__ == '__main__':
    main()
