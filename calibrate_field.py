#!/usr/bin/env python3
"""
Field calibration tool — click on the 4 corners of any visible reference box
(penalty area OR 6-yard goal area) to compute a pixel→world homography.

4 corners = minimum for homography. Use whichever box is fully visible.

Usage:
  # Using the 18-yard penalty area (default):
  .venv/bin/python3 calibrate_field.py --video "test 1.mp4" --format 11v11 \\
      --output calibration_test1.json

  # Using the 6-yard goal area (if penalty area is not fully visible):
  .venv/bin/python3 calibrate_field.py --video "test 1.mp4" --format 11v11 \\
      --box goal --output calibration_test1.json

  # Specify timestamp if default (5s) doesn't show clear field markings:
  .venv/bin/python3 calibrate_field.py --video "test 1.mp4" --format 11v11 \\
      --box goal --frame-sec 10 --output calibration_test1.json

Controls:
  Left-click  = place next point
  U           = undo last point
  S / Enter   = save (once ≥ 4 points placed)
  ESC         = quit without saving

Click order (4 required points, same for either box):
  1. Top-left  — field side (away from goal), left post side
  2. Top-right — field side (away from goal), right post side
  3. Bottom-left  — goal line, left post side
  4. Bottom-right — goal line, right post side

  "Left/Right" = from perspective of a player ATTACKING that goal.
  "Top"        = line further into the field (away from goal).
  "Bottom"     = the goal line itself.

World coordinate system (same origin regardless of which box you click):
  Origin = center of the goal line (end line)
  X = across the width  (+X = right post side)
  Y = into the field    (+Y = away from goal)
  Units = meters
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

FIELD_CONFIGS = {
    '11v11': {
        'center_circle_r': 9.15,
        # 18-yard penalty area
        'penalty_area_w':  40.32,   # total width (m)
        'penalty_area_d':  16.5,    # depth from goal line (m)
        # 6-yard goal area
        'goal_area_w':     18.32,   # total width: goal(7.32) + 2×5.5
        'goal_area_d':      5.5,    # depth from goal line (m)
        'goal_w':           7.32,
        'field_len':       105.0,
        'field_w':          68.0,
    },
    '9v9': {
        'center_circle_r': 7.0,
        # 18-yard penalty area (scaled for 9v9)
        'penalty_area_w':  28.0,
        'penalty_area_d':  12.0,
        # 6-yard goal area (scaled for 9v9)
        'goal_area_w':     13.0,    # goal(5m) + 2×4m
        'goal_area_d':      4.0,
        'goal_w':           5.0,
        'field_len':        73.0,
        'field_w':          46.0,
    },
}

# Which box dimensions to use per --box argument
BOX_KEYS = {
    'penalty': ('penalty_area_w', 'penalty_area_d'),
    'goal':    ('goal_area_w',    'goal_area_d'),
}
BOX_NAMES = {
    'penalty': '18-yard penalty area',
    'goal':    '6-yard goal area',
}

def point_labels(box_name):
    return [
        f"1 [REQUIRED] Top-left  of {box_name}  (field side, left post side)",
        f"2 [REQUIRED] Top-right of {box_name}  (field side, right post side)",
        f"3 [REQUIRED] Bottom-left  of {box_name} (goal line, left post side)",
        f"4 [REQUIRED] Bottom-right of {box_name} (goal line, right post side)",
        f"5 [OPTIONAL] Center of center circle  (skip with S if not visible)",
    ]

COLORS = [
    (0,   200, 255),   # teal    — top-left
    (255, 200,   0),   # cyan    — top-right
    (0,   100, 255),   # orange  — bottom-left
    (50,  255,  50),   # green   — bottom-right
    (255,  50, 255),   # magenta — center circle
]


def world_points_for(fmt, box, n_clicks):
    """
    Return world coordinates for the first n_clicks points.

    Origin = center of the goal line (end line).
    The reference box is centered on the goal (goal is centered on end line).

      TL (field side, left)  = (-W/2,  D)
      TR (field side, right) = ( W/2,  D)
      BL (goal line,  left)  = (-W/2,  0)
      BR (goal line,  right) = ( W/2,  0)
      Center circle center   = (0,     field_len/2)

    This keeps the same origin regardless of whether the user clicks the
    penalty area or the goal area, so calibrations are comparable.
    """
    cfg = FIELD_CONFIGS[fmt]
    wk, dk = BOX_KEYS[box]
    W  = cfg[wk]
    D  = cfg[dk]
    L  = cfg['field_len']

    all_pts = [
        [-W/2,  D  ],   # 1: top-left
        [ W/2,  D  ],   # 2: top-right
        [-W/2,  0.0],   # 3: bottom-left
        [ W/2,  0.0],   # 4: bottom-right
        [ 0.0,  L/2],   # 5: center circle center (from goal line)
    ]
    return np.array(all_pts[:n_clicks], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video',     required=True)
    ap.add_argument('--format',    choices=['11v11', '9v9'], required=True)
    ap.add_argument('--box',       choices=['penalty', 'goal'], default='penalty',
                    help='Which box to click: "penalty" (18-yard) or "goal" (6-yard). '
                         'Use whichever is fully visible. Default: penalty')
    ap.add_argument('--output',    default='field_calibration.json')
    ap.add_argument('--frame-sec', type=float, default=5.0,
                    help='Timestamp (seconds) to grab calibration frame from')
    args = ap.parse_args()

    cfg      = FIELD_CONFIGS[args.format]
    box_name = BOX_NAMES[args.box]
    wk, dk   = BOX_KEYS[args.box]
    POINT_LABELS = point_labels(box_name)

    print(f"\nField format  : {args.format}")
    print(f"Reference box : {box_name}  "
          f"({cfg[wk]:.2f}m wide × {cfg[dk]:.2f}m deep)")
    print(f"Center circle : {cfg['center_circle_r']}m radius")
    print(f"\nClick the 4 REQUIRED corners in order (5th optional):")
    for lbl in POINT_LABELS:
        print(f"  {lbl}")
    print()

    # Extract calibration frame
    print(f"Extracting frame at {args.frame_sec}s from '{args.video}'...")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'cal.jpg')
        subprocess.run([
            'ffmpeg', '-i', args.video,
            '-ss', str(args.frame_sec), '-vframes', '1',
            '-q:v', '2', '-loglevel', 'error', out
        ], check=True)
        frame = cv2.imread(out)

    if frame is None:
        print("ERROR: Could not extract frame.")
        return

    h, w = frame.shape[:2]
    scale = min(DISPLAY_W / w, DISPLAY_H / h)
    dw, dh = int(w * scale), int(h * scale)

    clicks = []   # full-res (x, y)

    cv2.namedWindow('Calibrate Field', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Calibrate Field', dw, dh)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(POINT_LABELS):
            rx, ry = int(x / scale), int(y / scale)
            clicks.append((rx, ry))
            wpt = world_points_for(args.format, args.box, len(clicks))[-1]
            print(f"  Point {len(clicks)}: pixel=({rx},{ry})  "
                  f"world=({wpt[0]:.2f},{wpt[1]:.2f})m")

    cv2.setMouseCallback('Calibrate Field', on_click)

    while True:
        disp = cv2.resize(frame.copy(), (dw, dh))

        # Draw placed points
        for i, (px, py) in enumerate(clicks):
            dx, dy = int(px * scale), int(py * scale)
            col = COLORS[i]
            cv2.circle(disp, (dx, dy), 9, col, -1)
            cv2.circle(disp, (dx, dy), 11, (255,255,255), 1)
            cv2.putText(disp, str(i+1), (dx+13, dy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

        # Draw lines between placed points to visualize the box
        order = [0, 1, 3, 2, 0]   # TL→TR→BR→BL→TL
        for j in range(len(order)-1):
            a, b = order[j], order[j+1]
            if a < len(clicks) and b < len(clicks):
                pa = (int(clicks[a][0]*scale), int(clicks[a][1]*scale))
                pb = (int(clicks[b][0]*scale), int(clicks[b][1]*scale))
                cv2.line(disp, pa, pb, (255,255,255), 1)

        # Instruction for next point
        n = len(clicks)
        if n < len(POINT_LABELS):
            lbl = POINT_LABELS[n]
            col = COLORS[n]
            cv2.putText(disp, f"Click: {lbl}",
                        (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
        else:
            cv2.putText(disp, "All points placed. Press S or Enter to save.",
                        (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)

        status = (f"Points: {n}/4 required  |  "
                  f"U=undo  S/Enter=save (≥4)  ESC=quit")
        cv2.putText(disp, status, (10, dh-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        cv2.imshow('Calibrate Field', disp)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('u') and clicks:
            clicks.pop()
            print("  Undo.")
        elif key in (ord('s'), 13) and len(clicks) >= 4:
            break
        elif key == 27:
            print("Cancelled — no file saved.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    if len(clicks) < 4:
        print("Need at least 4 points. Cancelled.")
        return

    # Compute homography
    pixel_pts = np.array(clicks, dtype=np.float32)
    world_pts = world_points_for(args.format, args.box, len(clicks))

    H, mask = cv2.findHomography(pixel_pts, world_pts, cv2.RANSAC, 5.0)
    H_inv, _ = cv2.findHomography(world_pts, pixel_pts, cv2.RANSAC, 5.0)

    if H is None:
        print("ERROR: Homography computation failed. Try again with clearer points.")
        return

    # Validate reprojection
    print("\nReprojection errors (pixel → world → back to pixel):")
    for i, (px, py) in enumerate(clicks):
        # pixel → world
        pt_px = np.array([[[float(px), float(py)]]], dtype=np.float64)
        wpt   = cv2.perspectiveTransform(pt_px, H.astype(np.float64))[0][0]
        # world → pixel
        pt_wp = np.array([[[float(wpt[0]), float(wpt[1])]]], dtype=np.float64)
        rpt   = cv2.perspectiveTransform(pt_wp, H_inv.astype(np.float64))[0][0]
        px_err = np.linalg.norm(rpt - np.array([px, py]))
        exp    = world_pts[i]
        w_err  = np.linalg.norm(wpt - exp)
        print(f"  Pt {i+1}: world=({wpt[0]:.2f},{wpt[1]:.2f})m  "
              f"expected=({exp[0]:.2f},{exp[1]:.2f})m  "
              f"world_err={w_err:.3f}m  pixel_err={px_err:.1f}px")

    # Save
    cal = {
        'video':        args.video,
        'format':       args.format,
        'box':          args.box,
        'box_name':     box_name,
        'field_config': cfg,
        'frame_w':      w,
        'frame_h':      h,
        'frame_sec':    args.frame_sec,
        'n_points':     len(clicks),
        'points': [
            {
                'label':      POINT_LABELS[i],
                'pixel':      list(clicks[i]),
                'world':      [float(world_pts[i][0]), float(world_pts[i][1])],
            }
            for i in range(len(clicks))
        ],
        'H':     H.tolist(),
        'H_inv': H_inv.tolist(),
    }
    with open(args.output, 'w') as f:
        json.dump(cal, f, indent=2)

    print(f"\nSaved calibration → {args.output}")
    print(f"  Format : {args.format}")
    print(f"  Box    : {box_name}")
    print(f"  Points : {len(clicks)} ({'including optional center circle' if len(clicks)==5 else '4 box corners only'})")
    print(f"\nNext step:")
    print(f"  .venv/bin/python3 analyze_tracking_v8.py \\")
    print(f"      --video \"{args.video}\" --labels ball_labels.json \\")
    print(f"      --cache yolo_cache_test1.json \\")
    print(f"      --calibration {args.output} --render")


if __name__ == '__main__':
    main()
