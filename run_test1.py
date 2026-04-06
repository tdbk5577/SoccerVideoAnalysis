#!/usr/bin/env python3
"""
Run v7h tracker on 'test 1.mp4' using yolo_cache_test1.json.
Evaluates against ball_labels.json and renders tracking_test1_winner.mp4.

Usage: .venv/bin/python3 run_test1.py [--render]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

# ── import tracker functions from v7h ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# Patch the cache path before importing
import analyze_tracking_v7h as v7h
v7h.YOLO_CACHE_FILE = 'yolo_cache_test1.json'

track   = v7h.track_v7h
evaluate = v7h.evaluate
render  = v7h.render_video
load_frames = v7h.load_frames
load_cache  = v7h.load_cache

FPS = 10.0
FRAMES_DIR = '/tmp/soccer_test1_frames'
VIDEO      = 'test 1.mp4'
CACHE_FILE = 'yolo_cache_test1.json'
LABELS_FILE = 'ball_labels.json'
OUT_VIDEO   = 'tracking_test1_winner.mp4'

# v7g/v7h winning params
BEST_PARAMS = dict(
    min_conf=0.12, hard_reset_conf=0.50, abc_protect_conf=0.65,
    speed_limit=120, vel_decay=0.85,
    abc_ratio=0.6, hc_thresh=0.65, chain_margin=400, gap_margin=320,
    min_gap_for_filter=6, interp_max_gap=100, y_off=47, x_off=0,
    person_accept_r=150, person_conf_min=0.30,
    use_kalman_smooth=True, kalman_q=1.0,
)


def extract_frames(video, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    pattern = os.path.join(frames_dir, 'frame_%06d.jpg')
    subprocess.run([
        'ffmpeg', '-i', video,
        '-vf', f'fps={FPS}',
        '-q:v', '2', '-loglevel', 'error', pattern
    ], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--render', action='store_true')
    args = ap.parse_args()

    # Load labels
    with open(LABELS_FILE) as f:
        data = json.load(f)
    labels = data['labels']
    lmap   = {lb['frame']: lb for lb in labels}
    print(f"Loaded {len(labels)} labels from {LABELS_FILE}")

    # Extract frames if needed
    if not os.path.isdir(FRAMES_DIR) or len(os.listdir(FRAMES_DIR)) == 0:
        print(f"Extracting frames from '{VIDEO}' to {FRAMES_DIR}...")
        extract_frames(VIDEO, FRAMES_DIR)
    frames = load_frames(FRAMES_DIR)
    print(f"Using {len(frames)} frames")

    # Load cache
    print(f"Loading YOLO cache from {CACHE_FILE}...")
    cache = load_cache(CACHE_FILE)
    print(f"  {len(cache)} frames cached")

    # Seed: use first label
    first = labels[0]
    seed_frame = first['frame']
    seed_pos   = (int(first['px']), int(first['py']))
    print(f"Seed: frame {seed_frame} → {seed_pos}")

    # Run tracker
    print("\nRunning tracker with best params...")
    pos = track(frames, seed_frame, seed_pos, cache, **BEST_PARAMS)

    # Evaluate
    tr, w30, w50, med, sc = evaluate(pos, labels)
    print(f"\nResults on {len(labels)} labeled frames:")
    print(f"  Tracked:  {tr}/{len(labels)}")
    print(f"  W30:      {w30}/{len(labels)}  ({100*w30/len(labels):.1f}%)")
    print(f"  W50:      {w50}/{len(labels)}  ({100*w50/len(labels):.1f}%)")
    print(f"  Median:   {med:.1f}px")
    print(f"  Score:    {sc:.4f}")

    # Per-frame errors
    print("\nPer-frame errors (misses > 50px):")
    misses = []
    for lb in labels:
        f = lb['frame']
        if f not in pos:
            misses.append((f, -1, lb['px'], lb['py'], 'NO_DET'))
            continue
        px, py = pos[f]
        err = v7h.dist((px, py), (lb['px'], lb['py']))
        if err > 50:
            misses.append((f, err, lb['px'], lb['py'], f'pred=({px},{py})'))
    print(f"  {len(misses)} misses > 50px:")
    for m in misses[:30]:
        print(f"    f{m[0]:4d}  err={m[1]:.0f}px  gt=({m[2]},{m[3]})  {m[4]}")
    if len(misses) > 30:
        print(f"    ... and {len(misses)-30} more")

    # Render
    if args.render:
        print(f"\nRendering {OUT_VIDEO}...")
        render(frames, pos, labels, OUT_VIDEO)


if __name__ == '__main__':
    main()
