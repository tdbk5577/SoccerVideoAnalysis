#!/usr/bin/env python3
"""
Diagnostic tool — runs YOLO on one frame and saves an annotated image
showing every detected player (with classified color) and the ball.

Usage:
  .venv/bin/python3 diagnose.py --video Test.mp4 --second 13
"""

import argparse
import os
import subprocess
import tempfile

import cv2
import numpy as np
from ultralytics import YOLO

TEAM_COLOR_HSV = {
    "blue":   [([90,  40,  40],  [135, 255, 255])],
    "red":    [([0,   40,  40],  [15,  255, 255]),
               ([155, 40,  40],  [180, 255, 255])],
    "orange": [([8,   80,  80],  [25,  255, 255])],
    "green":  [([35,  40,  40],  [85,  255, 255])],
    "yellow": [([18,  80,  80],  [38,  255, 255])],
    "white":  [([0,   0,   180], [180, 40,  255])],
    "black":  [([0,   0,   0],   [180, 255, 60])],
    "purple": [([125, 40,  40],  [160, 255, 255])],
}

DRAW_COLORS = {
    "blue":   (255, 100, 0),
    "red":    (0,   0,   255),
    "orange": (0,   140, 255),
    "green":  (0,   200, 0),
    "yellow": (0,   220, 220),
    "white":  (220, 220, 220),
    "black":  (80,  80,  80),
    "purple": (180, 0,   180),
    "unknown":(128, 128, 128),
    "ball":   (0,   255, 255),
}

def count_color_pixels(hsv, color):
    total = 0
    for lo, hi in TEAM_COLOR_HSV[color]:
        total += int(cv2.countNonZero(cv2.inRange(hsv, np.array(lo), np.array(hi))))
    return total

def classify_jersey(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    jersey_h = max(1, (y2 - y1) * 2 // 5)
    region = frame[y1:y1 + jersey_h, x1:x2]
    if region.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    scores = {c: count_color_pixels(hsv, c) for c in TEAM_COLOR_HSV}
    best = max(scores, key=scores.get)
    return best if scores[best] > 20 else "unknown"

def detect_ball_motion(frame, prev_frame, w, h):
    gray      = cv2.cvtColor(frame,      cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, prev_gray)
    _, motion = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN,  kernel)
    motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 8 < area < 500:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"] / w
                cy = M["m01"] / M["m00"] / h
                perimeter = cv2.arcLength(cnt, True)
                circ = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0
                if circ > 0.35:
                    candidates.append({"cx": cx, "cy": cy, "area": area, "circ": circ})
    return max(candidates, key=lambda b: b["circ"]) if candidates else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  required=True)
    parser.add_argument("--second", type=float, default=13.0, help="Timestamp to extract (seconds)")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_path      = os.path.join(tmpdir, "frame.jpg")
        prev_frame_path = os.path.join(tmpdir, "prev_frame.jpg")

        subprocess.run([
            "ffmpeg", "-i", args.video, "-ss", str(args.second),
            "-vframes", "1", "-q:v", "2", "-loglevel", "error", frame_path
        ], check=True)
        prev_t = max(0, args.second - 0.5)
        subprocess.run([
            "ffmpeg", "-i", args.video, "-ss", str(prev_t),
            "-vframes", "1", "-q:v", "2", "-loglevel", "error", prev_frame_path
        ], check=True)

        frame = cv2.imread(frame_path)
        h, w = frame.shape[:2]
        print(f"Frame size: {w}x{h}")

        model = YOLO("yolov8m.pt")
        detections = model(frame, verbose=False)[0]

        player_count = 0
        ball_found   = False

        for box in detections.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, bbox)
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h

            if cls == 0 and conf > 0.25:  # person
                color = classify_jersey(frame, bbox)
                draw_col = DRAW_COLORS.get(color, (128, 128, 128))
                cv2.rectangle(frame, (x1, y1), (x2, y2), draw_col, 2)
                label = f"{color} ({cx:.2f},{cy:.2f})"
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, draw_col, 1)
                print(f"  Player: color={color:8s}  cx={cx:.2f}  cy={cy:.2f}  conf={conf:.2f}")
                player_count += 1

            elif cls == 32 and conf > 0.1:  # ball
                cx_b = (bbox[0] + bbox[2]) / 2 / w
                cy_b = (bbox[1] + bbox[3]) / 2 / h
                cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 10, DRAW_COLORS["ball"], 3)
                cv2.putText(frame, f"BALL ({cx_b:.2f},{cy_b:.2f})", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, DRAW_COLORS["ball"], 2)
                print(f"  Ball:   cx={cx_b:.2f}  cy={cy_b:.2f}  conf={conf:.2f}")
                ball_found = True

        # Ball detection via motion + proximity filter
        prev_frame_img = cv2.imread(prev_frame_path)
        players_list   = [{"cx": p["cx"], "cy": p["cy"]} for p in detections.boxes
                          if int(p.cls[0]) == 0 and float(p.conf[0]) > 0.25] if False else []
        # Re-collect players for filter
        all_players = []
        for box in detections.boxes:
            if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.25:
                bbox = box.xyxy[0].tolist()
                all_players.append({
                    "cx": (bbox[0] + bbox[2]) / 2 / w,
                    "cy": (bbox[1] + bbox[3]) / 2 / h,
                })

        ball_motion = detect_ball_motion(frame, prev_frame_img, w, h) if prev_frame_img is not None else None
        # Filter: must be near a player
        if ball_motion and all_players:
            nearest = min(
                ((p["cx"] - ball_motion["cx"]) ** 2 + (p["cy"] - ball_motion["cy"]) ** 2) ** 0.5
                for p in all_players
            )
            if nearest > 0.15:
                ball_motion = None

        if ball_motion:
            bx = int(ball_motion["cx"] * w)
            by = int(ball_motion["cy"] * h)
            cv2.circle(frame, (bx, by), 12, (0, 255, 0), 3)
            cv2.putText(frame, f"BALL ({ball_motion['cx']:.2f},{ball_motion['cy']:.2f})",
                        (bx + 5, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"  Ball (motion): cx={ball_motion['cx']:.2f}  cy={ball_motion['cy']:.2f}  circ={ball_motion['circ']:.2f}")
        else:
            print("  Ball (motion): not detected near any player")

        print(f"\nTotal players detected: {player_count}")
        print(f"YOLO ball detected: {ball_found}")

        out = f"diagnostic_second_{int(args.second)}.jpg"
        cv2.imwrite(out, frame)
        print(f"\nAnnotated frame saved to: {out}")
        print("Open it to see what YOLO detected and what colors were assigned.")

if __name__ == "__main__":
    main()
