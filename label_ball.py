#!/usr/bin/env python3
"""
Ball labeling tool - step through frames and click on the ball.
Saves ground-truth positions to ball_labels.json for algorithm development.

Controls:
  Left-click  = mark ball position in this frame
  N / Space   = next frame (skip if ball not visible)
  B           = go back one frame
  U           = undo click in current frame
  S           = save and quit early
  ESC         = quit without saving

Usage:
  .venv/bin/python3 label_ball.py --video Test.mp4
  .venv/bin/python3 label_ball.py --video Test.mp4 --interval 0.5
"""

import argparse
import json
import os
import subprocess
import tempfile

import cv2
import numpy as np

DISPLAY_W = 1600
DISPLAY_H = 900


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Seconds between frames shown (default: 0.5)")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Extraction FPS (default: 10)")
    parser.add_argument("--output", default="ball_labels.json",
                        help="Output file (default: ball_labels.json)")
    parser.add_argument("--load", default=None,
                        help="Pre-load existing labels from this file (merges on save)")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Extracting frames at {args.fps}fps...")
        pattern = os.path.join(tmpdir, "frame_%04d.jpg")
        subprocess.run([
            "ffmpeg", "-i", args.video,
            "-vf", f"fps={args.fps}",
            "-q:v", "2", "-loglevel", "error", pattern
        ], check=True)

        frames = sorted([
            os.path.join(tmpdir, f)
            for f in os.listdir(tmpdir)
            if f.startswith("frame_") and f.endswith(".jpg")
        ])
        total = len(frames)
        print(f"Extracted {total} frames ({total/args.fps:.1f}s)")

        ref = cv2.imread(frames[0])
        h, w = ref.shape[:2]
        scale = min(DISPLAY_W / w, DISPLAY_H / h)
        dw, dh = int(w * scale), int(h * scale)

        # Step size in frames
        step = max(1, int(args.interval * args.fps))

        # Which frames to show
        indices = list(range(0, total, step))
        labels  = {}   # frame_idx -> {"px": x, "py": y, "t": seconds}
        clicked = {}   # current display-space click per frame

        # Pre-load existing labels if requested
        if args.load and os.path.exists(args.load):
            with open(args.load) as lf:
                existing = json.load(lf)
            loaded_fps = existing.get("fps", args.fps)
            for lb in existing.get("labels", []):
                # Remap frame index to current fps
                t = lb["t"]
                new_idx = int(round(t * args.fps))
                labels[new_idx] = {"px": lb["px"], "py": lb["py"], "t": round(t, 3)}
                clicked[new_idx] = (int(lb["px"] * scale), int(lb["py"] * scale))
            print(f"Pre-loaded {len(labels)} labels from {args.load}")

        cv2.namedWindow("Label Ball", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Label Ball", dw, dh)

        current = 0   # index into `indices`

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Convert display coords back to full-res
                real_x = int(x / scale)
                real_y = int(y / scale)
                idx = indices[current]
                labels[idx] = {"px": real_x, "py": real_y,
                               "t": round(idx / args.fps, 3)}
                clicked[idx] = (x, y)   # store display coords for drawing

        cv2.setMouseCallback("Label Ball", on_click)

        print("\nControls: Click=mark ball | Space/N=next | B=back | U=undo | S=save&quit | ESC=quit")
        print(f"Showing every {args.interval}s ({step} frames). {len(indices)} frames total.\n")

        while 0 <= current < len(indices):
            idx = indices[current]
            t   = idx / args.fps

            img = cv2.imread(frames[idx])
            disp = cv2.resize(img, (dw, dh))

            # Draw existing click for this frame
            if idx in clicked:
                cx, cy = clicked[idx]
                cv2.circle(disp, (cx, cy), int(14 * scale), (0, 255, 0), 2)
                cv2.circle(disp, (cx, cy), int(3  * scale), (0, 255, 0), -1)

            # Draw trail of recent labels
            recent = sorted([k for k in labels if abs(k - idx) <= step * 5])
            for k in recent:
                if k == idx:
                    continue
                lx = int(labels[k]["px"] * scale)
                ly = int(labels[k]["py"] * scale)
                alpha = 1.0 - abs(k - idx) / (step * 5 + 1)
                col = (0, int(180 * alpha), int(255 * alpha))
                cv2.circle(disp, (lx, ly), int(6 * scale), col, 1)

            # Status text
            labeled_count = len(labels)
            status = "LABELED" if idx in labels else "not labeled"
            cv2.putText(disp, f"{t:.1f}s  frame {idx}/{total-1}  [{current+1}/{len(indices)}]  {status}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(disp, f"Labeled: {labeled_count}  |  Space=next  B=back  U=undo  S=save  ESC=quit",
                        (10, dh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Label Ball", disp)
            key = cv2.waitKey(50) & 0xFF

            if key in (ord('n'), ord(' ')):
                current = min(current + 1, len(indices) - 1)
            elif key == ord('b'):
                current = max(current - 1, 0)
            elif key == ord('u'):
                labels.pop(idx, None)
                clicked.pop(idx, None)
            elif key == ord('s'):
                break
            elif key == 27:  # ESC
                print("Quit without saving.")
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()

        if not labels:
            print("No labels recorded.")
            return

        # Save
        output = {
            "video": args.video,
            "fps": args.fps,
            "frame_w": w,
            "frame_h": h,
            "labels": [
                {"frame": k, "t": v["t"], "px": v["px"], "py": v["py"],
                 "cx": round(v["px"] / w, 4), "cy": round(v["py"] / h, 4)}
                for k, v in sorted(labels.items())
            ]
        }

        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nSaved {len(labels)} labels to {args.output}")

        # Print summary
        pts = output["labels"]
        print(f"\n{'Time':>6}  {'px':>5}  {'py':>5}  {'cx':>6}  {'cy':>6}")
        print("-" * 40)
        for p in pts:
            print(f"{p['t']:>6.1f}  {p['px']:>5}  {p['py']:>5}  {p['cx']:>6.3f}  {p['cy']:>6.3f}")

        # Inter-frame speed analysis
        if len(pts) >= 2:
            print("\nBall speed between labeled frames:")
            for i in range(1, len(pts)):
                a, b = pts[i-1], pts[i]
                dt = b["t"] - a["t"]
                dist = ((b["px"]-a["px"])**2 + (b["py"]-a["py"])**2)**0.5
                pxps = dist / dt if dt > 0 else 0
                print(f"  {a['t']:.1f}s -> {b['t']:.1f}s : {dist:.0f}px in {dt:.1f}s = {pxps:.0f}px/s")


if __name__ == "__main__":
    main()
