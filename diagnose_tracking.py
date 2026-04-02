#!/usr/bin/env python3
"""
Ball tracking diagnostic.
Seeds by finding the stationary ball at center before kickoff (temporal average
of first N frames makes moving objects blur out, stationary ball stays sharp).
Tracks with template matching + linear velocity prediction.
Max search radius 50px per frame (calibrated from ground-truth labels).

Usage:
  .venv/bin/python3 diagnose_tracking.py --video Test.mp4
  .venv/bin/python3 diagnose_tracking.py --video Test.mp4 --click
  .venv/bin/python3 diagnose_tracking.py --video Test.mp4 --eval ball_labels.json
"""

import argparse
import json
import os
import subprocess
import tempfile

import cv2
import numpy as np

TRACKING_FPS  = 10
TEMPLATE_SIZE = 22    # px patch around ball
MAX_SPEED_PX  = 50    # max ball displacement per frame (calibrated from labels)
SCORE_THRESH  = 0.28  # min template match score to accept


def find_ball_stationary(frames, seed_frame_idx, w, h, avg_frames=20):
    """
    Strategy 1: Before kickoff the ball sits motionless at center.
    Averaging N frames blurs moving players; the stationary ball stays sharp.
    Find it with Hough circles on the temporal average.
    """
    x1_r = int(0.35 * w); x2_r = int(0.65 * w)
    y1_r = int(0.33 * h); y2_r = int(0.67 * h)

    stack = []
    for i in range(seed_frame_idx, min(seed_frame_idx + avg_frames, len(frames))):
        f = cv2.imread(frames[i])
        if f is not None:
            stack.append(f[y1_r:y2_r, x1_r:x2_r].astype(np.float32))
    if not stack:
        return None

    avg = np.mean(stack, axis=0).astype(np.uint8)
    gray = cv2.cvtColor(avg, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("debug_avg.jpg", avg)

    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1, minDist=10,
        param1=30, param2=8,
        minRadius=3, maxRadius=13
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)
    roi_cx = (x2_r - x1_r) / 2
    roi_cy = (y2_r - y1_r) / 2

    best = None
    best_score = -1
    for cx, cy, r in circles:
        px1 = max(0, cx - r); px2 = min(gray.shape[1], cx + r)
        py1 = max(0, cy - r); py2 = min(gray.shape[0], cy + r)
        patch = gray[py1:py2, px1:px2]
        brightness = float(np.mean(patch)) if patch.size > 0 else 0
        dist = ((cx - roi_cx)**2 + (cy - roi_cy)**2)**0.5
        score = brightness / (dist + 10)
        if score > best_score:
            best_score = score
            best = (cx + x1_r, cy + y1_r)

    if best:
        print(f"  Strategy 1 (stationary): px=({best[0]},{best[1]}) "
              f"cx={best[0]/w:.3f} score={best_score:.2f}")
        return seed_frame_idx, best[0], best[1]
    return None


def find_ball_in_motion(frames, seed_frame_idx, w, h, search_seconds=6):
    """
    Strategy 2: Ball not visible at rest — scan for when it gets kicked.
    Key physics: after a kick the ball moves 25-50px/frame.
    Players move 3-12px/frame (walking/jogging).
    Ball is also much smaller than any player bounding box.

    Look for a small, fast, roughly-circular moving blob that originated
    near center (cx 0.28-0.72). Take the fastest small blob each frame pair;
    if consistent over 3 consecutive pairs, that's the ball.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Wider search region — ball may have already moved away from dead center
    x1_r = int(0.25 * w); x2_r = int(0.75 * w)
    y1_r = int(0.22 * h); y2_r = int(0.78 * h)

    max_idx = min(seed_frame_idx + int(search_seconds * TRACKING_FPS), len(frames) - 3)

    def blobs_between(ia, ib):
        """Return moving blobs (cx, cy, speed, circ) between frames ia and ib."""
        fa = cv2.imread(frames[ia])
        fb = cv2.imread(frames[ib])
        if fa is None or fb is None:
            return []
        ga = cv2.cvtColor(fa[y1_r:y2_r, x1_r:x2_r], cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(fb[y1_r:y2_r, x1_r:x2_r], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(ga, gb)
        _, motion = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Ball blob: small (not a player) — players are 10x bigger
            if 10 < area < 500:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"]) + x1_r
                    cy = int(M["m01"] / M["m00"]) + y1_r
                    perimeter = cv2.arcLength(cnt, True)
                    circ = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
                    result.append((cx, cy, area, circ))
        return result

    # For each frame pair, find the fastest small blob in center region
    # Then check consistency over 3 pairs (ball keeps moving, noise doesn't)
    for i in range(seed_frame_idx, max_idx - 2):
        b01 = blobs_between(i,   i+1)
        b12 = blobs_between(i+1, i+2)
        b23 = blobs_between(i+2, i+3)

        if not b01:
            continue

        # Only consider blobs that started near center
        center_blobs = [b for b in b01
                        if int(0.28 * w) < b[0] < int(0.72 * w)
                        and int(0.25 * h) < b[1] < int(0.75 * h)]
        if not center_blobs:
            continue

        def nearest(blobs, cx, cy, max_dist=80):
            best, bd = None, max_dist
            for b in blobs:
                d = ((b[0]-cx)**2 + (b[1]-cy)**2)**0.5
                if d < bd:
                    bd, best = d, b
            return best, bd

        best_ball   = None
        best_score  = 0

        for b0 in center_blobs:
            cx0, cy0 = b0[0], b0[1]

            b1, d01 = nearest(b12, cx0, cy0, max_dist=80)
            if b1 is None or d01 < 15:   # must have moved at least 15px
                continue

            b2, d12 = nearest(b23, b1[0], b1[1], max_dist=100)

            # Speed score: ball moves fast, players move slow
            speed_score = min(d01 / 20.0, 2.0)

            # Consistency: ball continues in same direction
            consistency = 1.0
            if b2 is not None and d12 > 10:
                dx01 = b1[0] - cx0; dy01 = b1[1] - cy0
                dx12 = b2[0] - b1[0]; dy12 = b2[1] - b1[1]
                dot = dx01*dx12 + dy01*dy12
                consistency = 1.4 if dot > 0 else 0.6

            score = speed_score * consistency * (b0[3] + 0.1)  # circ bonus

            if score > best_score:
                best_score = score
                # Return position at frame i+1 (after first kick)
                best_ball = (i + 1, b1[0], b1[1])

        if best_ball and best_score > 0.5:
            idx, px, py = best_ball
            print(f"  Strategy 2 (motion): frame {idx} ({idx/TRACKING_FPS:.1f}s) "
                  f"px=({px},{py}) cx={px/w:.3f} score={best_score:.2f}")
            return best_ball

    return None


def find_ball_seed(frames, seed_frame_idx, w, h):
    """Try stationary detection first, fall back to motion detection."""
    print("  Trying strategy 1: stationary ball at center (temporal average)...")
    result = find_ball_stationary(frames, seed_frame_idx, w, h, avg_frames=20)
    if result:
        return result

    print("  Strategy 1 failed. Trying strategy 2: first ball motion near center...")
    result = find_ball_in_motion(frames, seed_frame_idx, w, h, search_seconds=6)
    if result:
        return result

    return None


def click_to_seed(frames, seed_frame_idx, w, h):
    """Show a frame and let the user click on the ball."""
    clicked = []
    DISPLAY_W, DISPLAY_H = 1600, 900
    scale = min(DISPLAY_W / w, DISPLAY_H / h)
    dw, dh = int(w * scale), int(h * scale)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((int(x / scale), int(y / scale)))

    img = cv2.imread(frames[seed_frame_idx])
    disp = cv2.resize(img, (dw, dh))
    cv2.putText(disp, "Click on the ball, then press any key", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
    cv2.namedWindow("Click on ball", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Click on ball", dw, dh)
    cv2.setMouseCallback("Click on ball", on_click)

    while True:
        show = disp.copy()
        if clicked:
            px, py = int(clicked[-1][0] * scale), int(clicked[-1][1] * scale)
            cv2.circle(show, (px, py), 14, (0, 255, 0), 3)
        cv2.imshow("Click on ball", show)
        key = cv2.waitKey(50)
        if key != -1 and clicked:
            break
        if key == 27:
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    px, py = clicked[-1]
    print(f"  Clicked: px=({px},{py})  cx={px/w:.3f}  cy={py/h:.3f}")
    return seed_frame_idx, px, py


def track_with_template(frames, seed_frame_idx, seed_px, seed_py, w, h):
    """
    Template matching with linear velocity prediction.
    Max search: MAX_SPEED_PX around predicted position.
    """
    positions = [None] * len(frames)
    positions[seed_frame_idx] = (seed_px, seed_py)

    half = TEMPLATE_SIZE // 2

    def get_template(frame, px, py):
        x1, y1 = max(0, px - half), max(0, py - half)
        x2, y2 = min(w, px + half), min(h, py + half)
        patch = frame[y1:y2, x1:x2]
        return patch.copy() if patch.size > 0 else None

    def predict(history):
        if len(history) >= 2:
            vx = history[-1][0] - history[-2][0]
            vy = history[-1][1] - history[-2][1]
            return history[-1][0] + vx, history[-1][1] + vy
        return history[-1]

    def search(frame, pred_px, pred_py, tmpl, radius):
        pred_px, pred_py = int(pred_px), int(pred_py)
        x1 = max(0, pred_px - radius)
        y1 = max(0, pred_py - radius)
        x2 = min(w, pred_px + radius)
        y2 = min(h, pred_py + radius)
        roi = frame[y1:y2, x1:x2]
        th, tw = tmpl.shape[:2]
        if roi.shape[0] < th or roi.shape[1] < tw:
            return None, 0.0
        result = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        found_px = x1 + max_loc[0] + tw // 2
        found_py = y1 + max_loc[1] + th // 2
        return (found_px, found_py), float(max_val)

    seed_frame = cv2.imread(frames[seed_frame_idx])
    base_tmpl  = get_template(seed_frame, seed_px, seed_py)
    if base_tmpl is None:
        return positions

    def run_direction(start_idx, start_pos, step):
        history = [start_pos]
        tmpl    = base_tmpl.copy()
        idx     = start_idx + step
        consecutive_lost = 0

        while 0 <= idx < len(frames):
            frame = cv2.imread(frames[idx])
            if frame is None:
                break

            pred_px, pred_py = predict(history)

            # Tight window first (trust velocity prediction)
            pos, score = search(frame, pred_px, pred_py, tmpl, radius=MAX_SPEED_PX)

            # Wider fallback only if tight failed
            if score < SCORE_THRESH:
                pos, score = search(frame, pred_px, pred_py, tmpl, radius=MAX_SPEED_PX * 2)

            if score >= SCORE_THRESH:
                positions[idx] = pos
                history.append(pos)
                consecutive_lost = 0
                if len(history) > 5:
                    history.pop(0)
                # Refresh template if ball moved a reasonable distance (good lock)
                if len(history) >= 2:
                    move = ((pos[0]-history[-2][0])**2 + (pos[1]-history[-2][1])**2)**0.5
                    if move < MAX_SPEED_PX:
                        new_t = get_template(frame, pos[0], pos[1])
                        if new_t is not None and new_t.shape == base_tmpl.shape:
                            tmpl = new_t
            else:
                consecutive_lost += 1
                # Hold predicted position for up to 10 frames; then stop
                if consecutive_lost <= 10:
                    ipos = (int(pred_px), int(pred_py))
                    positions[idx] = ipos
                    history.append(ipos)
                    if len(history) > 5:
                        history.pop(0)
                else:
                    # Lost the ball — stop tracking in this direction
                    break

            idx += step

    run_direction(seed_frame_idx, (seed_px, seed_py), step=+1)
    run_direction(seed_frame_idx, (seed_px, seed_py), step=-1)
    return positions


def evaluate(positions, labels_path, fps, w, h):
    """Compare tracked positions against ground-truth labels."""
    with open(labels_path) as f:
        data = json.load(f)

    labels = {e["frame"]: (e["px"], e["py"]) for e in data["labels"]}
    errors = []
    for fidx, gt in sorted(labels.items()):
        pred = positions[fidx] if fidx < len(positions) else None
        if pred is None:
            print(f"  frame {fidx:3d} ({fidx/fps:5.1f}s): NOT TRACKED  gt=({gt[0]},{gt[1]})")
            errors.append(None)
        else:
            err = ((pred[0]-gt[0])**2 + (pred[1]-gt[1])**2)**0.5
            flag = " <-- MISS" if err > 50 else ""
            print(f"  frame {fidx:3d} ({fidx/fps:5.1f}s): err={err:5.1f}px  "
                  f"pred=({pred[0]},{pred[1]})  gt=({gt[0]},{gt[1]}){flag}")
            errors.append(err)

    tracked_errors = [e for e in errors if e is not None]
    if tracked_errors:
        print(f"\n  Tracked: {len(tracked_errors)}/{len(errors)} labeled frames")
        print(f"  Median error: {np.median(tracked_errors):.1f}px")
        print(f"  Mean error:   {np.mean(tracked_errors):.1f}px")
        print(f"  Max error:    {max(tracked_errors):.1f}px")
        print(f"  Within 30px:  {sum(1 for e in tracked_errors if e <= 30)}/{len(tracked_errors)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",       required=True)
    parser.add_argument("--seed-second", type=float, default=0.0)
    parser.add_argument("--click",       action="store_true",
                        help="Click on the ball to set seed")
    parser.add_argument("--eval",        metavar="LABELS_JSON",
                        help="Evaluate against ground-truth labels file")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Extracting {TRACKING_FPS}fps frames...")
        pattern = os.path.join(tmpdir, "track_%04d.jpg")
        subprocess.run([
            "ffmpeg", "-i", args.video,
            "-vf", f"fps={TRACKING_FPS}",
            "-q:v", "2", "-loglevel", "error", pattern
        ], check=True)

        frames = sorted([os.path.join(tmpdir, f) for f in os.listdir(tmpdir)
                         if f.startswith("track") and f.endswith(".jpg")])
        print(f"Extracted {len(frames)} frames")

        ref = cv2.imread(frames[0])
        h, w = ref.shape[:2]
        print(f"Frame size: {w}x{h}")

        seed_frame_idx = int(args.seed_second * TRACKING_FPS)
        seed_frame_idx = min(seed_frame_idx, len(frames) - 2)

        if args.click:
            print(f"Showing frame at {args.seed_second:.1f}s — click on the ball...")
            result = click_to_seed(frames, seed_frame_idx, w, h)
        else:
            print("Searching for ball...")
            result = find_ball_seed(frames, seed_frame_idx, w, h)

        if result is None:
            print("Could not find ball. Use --click to click on it manually.")
            return

        actual_seed_idx, seed_px, seed_py = result
        print(f"Seed: frame {actual_seed_idx} ({actual_seed_idx/TRACKING_FPS:.1f}s) "
              f"px=({seed_px},{seed_py})")

        print("Tracking...")
        positions = track_with_template(frames, actual_seed_idx, seed_px, seed_py, w, h)

        tracked = sum(1 for p in positions if p is not None)
        print(f"Tracked in {tracked}/{len(frames)} frames")

        if args.eval:
            print(f"\nEvaluating against {args.eval}:")
            evaluate(positions, args.eval, TRACKING_FPS, w, h)

        # Write annotated video
        out_path = "tracking_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, TRACKING_FPS, (w, h))

        for i, fp in enumerate(frames):
            img = cv2.imread(fp)
            if img is None:
                continue
            t = i / TRACKING_FPS
            cv2.putText(img, f"{t:.1f}s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            if positions[i]:
                px, py = positions[i]
                cv2.circle(img, (px, py), 14, (0, 255, 0), 3)
                cv2.circle(img, (px, py), 3,  (0, 255, 0), -1)
            writer.write(img)

        writer.release()
        print(f"\nAnnotated video saved to: {out_path}")


if __name__ == "__main__":
    main()
