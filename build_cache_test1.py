#!/usr/bin/env python3
"""
Build YOLO cache for 'test 1.mp4' → yolo_cache_test1.json
Usage: .venv/bin/python3 build_cache_test1.py
"""

import json
import os
import subprocess
import tempfile
import time

import cv2

FPS        = 10.0
MODEL_NAME = 'yolov8x.pt'
IMGSZ      = 1920
CONF       = 0.05
VIDEO      = 'test 1.mp4'
CACHE_FILE = 'yolo_cache_test1.json'
BALL_CLASS   = 32
PERSON_CLASS = 0


def extract_frames(video, tmpdir):
    pattern = os.path.join(tmpdir, 'frame_%06d.jpg')
    subprocess.run([
        'ffmpeg', '-i', video,
        '-vf', f'fps={FPS}',
        '-q:v', '2', '-loglevel', 'error', pattern
    ], check=True)
    files = sorted(f for f in os.listdir(tmpdir) if f.endswith('.jpg'))
    return [os.path.join(tmpdir, f) for f in files]


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
        if (i+1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (n - i - 1)
            print(f'  {i+1}/{n}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s')
    return cache


def main():
    print(f'Extracting frames from "{VIDEO}" at {FPS}fps...')
    with tempfile.TemporaryDirectory() as tmpdir:
        frames = extract_frames(VIDEO, tmpdir)
        print(f'Extracted {len(frames)} frames')

        print(f'Running {MODEL_NAME} imgsz={IMGSZ} conf={CONF}...')
        t0 = time.time()
        cache = run_yolo(frames)
        elapsed = time.time() - t0
        print(f'YOLO done in {elapsed:.1f}s')

    with open(CACHE_FILE, 'w') as f:
        json.dump({str(k): v for k, v in cache.items()}, f)
    print(f'Saved cache to {CACHE_FILE}  ({len(cache)} frames)')


if __name__ == '__main__':
    main()
