#!/usr/bin/env python3
"""
Build YOLO cache for 'test 1.mp4' at 15fps.

Extracts frames to frames_15fps/ (persistent directory — TrackNet reads from here).
Runs YOLOv8x on every frame and saves detections to yolo_cache_test1_15fps.json.

Usage:
  .venv/bin/python3 build_cache_15fps.py
"""

import json
import os
import subprocess
import time

import cv2

FPS          = 15.0
MODEL_NAME   = 'yolov8x.pt'
IMGSZ        = 1920
CONF         = 0.05
VIDEO        = 'test 1.mp4'
CACHE_FILE   = 'yolo_cache_test1_15fps.json'
FRAMES_DIR   = 'frames_15fps'
BALL_CLASS   = 32
PERSON_CLASS = 0


def extract_frames():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    existing = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg'))
    if existing:
        print(f'  {FRAMES_DIR}/ already has {len(existing)} frames — skipping extraction')
        return [os.path.join(FRAMES_DIR, f) for f in existing]

    pattern = os.path.join(FRAMES_DIR, 'frame_%06d.jpg')
    subprocess.run([
        'ffmpeg', '-i', VIDEO, '-vf', f'fps={FPS}',
        '-q:v', '2', '-loglevel', 'error', pattern
    ], check=True)
    files = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg'))
    return [os.path.join(FRAMES_DIR, f) for f in files]


def run_yolo(frames):
    from ultralytics import YOLO
    model = YOLO(MODEL_NAME)
    cache = {}
    n = len(frames)
    t0 = time.time()
    for i, fp in enumerate(frames):
        fr = cv2.imread(fp)
        if fr is None:
            cache[i] = {'balls': [], 'persons': []}
            continue
        r = model(fr, verbose=False, conf=CONF, imgsz=IMGSZ)[0]
        balls, persons = [], []
        for j, cls in enumerate(r.boxes.cls):
            cls_id = int(cls)
            box    = r.boxes.xyxy[j].cpu().numpy()
            cf     = float(r.boxes.conf[j])
            cx     = float((box[0] + box[2]) / 2)
            cy     = float((box[1] + box[3]) / 2)
            if cls_id == BALL_CLASS:
                balls.append([cx, cy, cf])
            elif cls_id == PERSON_CLASS:
                persons.append([cx, cy, cf,
                                float(box[0]), float(box[1]),
                                float(box[2]), float(box[3])])
        cache[i] = {'balls': balls, 'persons': persons}
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f'  {i+1}/{n}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s')
    return cache


def main():
    print(f'Extracting "{VIDEO}" at {FPS}fps → {FRAMES_DIR}/')
    frames = extract_frames()
    print(f'  {len(frames)} frames  ({len(frames)/FPS:.1f}s of video)')

    if os.path.exists(CACHE_FILE):
        print(f'\n{CACHE_FILE} already exists — delete it to re-run YOLO.')
        return

    print(f'\nRunning {MODEL_NAME}  imgsz={IMGSZ}  conf={CONF}...')
    t0 = time.time()
    cache = run_yolo(frames)
    print(f'YOLO done in {time.time()-t0:.1f}s')

    with open(CACHE_FILE, 'w') as f:
        json.dump({str(k): v for k, v in cache.items()}, f)
    print(f'Saved {CACHE_FILE}  ({len(cache)} frames)')


if __name__ == '__main__':
    main()
