#!/usr/bin/env python3
"""
Extract 'test 1.mp4' frames at 15fps → frames_15fps/

No YOLO needed. TrackNet replaces ball detection.
Person detections are remapped from the existing 10fps YOLO cache in v9.

Usage:
  .venv/bin/python3 build_cache_15fps.py
"""

import os
import subprocess

FPS        = 15.0
VIDEO      = 'test 1.mp4'
FRAMES_DIR = 'frames_15fps'


def main():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    existing = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg'))
    if existing:
        print(f'{FRAMES_DIR}/ already has {len(existing)} frames — nothing to do.')
        return

    print(f'Extracting "{VIDEO}" at {FPS}fps → {FRAMES_DIR}/')
    pattern = os.path.join(FRAMES_DIR, 'frame_%06d.jpg')
    subprocess.run([
        'ffmpeg', '-i', VIDEO, '-vf', f'fps={FPS}',
        '-q:v', '2', '-loglevel', 'error', pattern
    ], check=True)

    count = len([f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')])
    print(f'Done: {count} frames  ({count / FPS:.1f}s of video)')


if __name__ == '__main__':
    main()
