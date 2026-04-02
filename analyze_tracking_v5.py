#!/usr/bin/env python3
"""
Ball tracking v5 — White-blob Kalman tracker.

Root cause of v1-v4 failure: template matching and motion-only detectors
both pick wrong things. Template matching matched y=808 instead of y=546 at frame 5.

Core insight: The ball is WHITE. The field is GREEN. White blobs of the right
size and shape on the field = ball candidates. This is a much stronger prior
than template appearance or raw motion.

V5 approach:
  1. White blob detector: HSV (S<=s_max, V>=v_min) connected components.
     Filter by size (area_min–area_max), aspect ratio (<max_aspect),
     and context (surrounding pixels should be green).
  2. Motion amplifier: for fast-moving ball, intersect white mask with
     k-frame accumulated motion to eliminate stationary white objects.
  3. Kalman filter (4-state x,y,vx,vy) fuses candidates.
  4. Adaptive search radius: expands proportionally to estimated speed.
  5. Multi-frame stable-blob seeding: find most consistent small white
     blob in first 40 frames (ball is stationary, lines are large).
  6. Recovery mode: full-frame white blob search nearest last known pos.

Usage:
  .venv/bin/python3 analyze_tracking_v5.py --video Test.mp4 \\
      --labels ball_labels.json --frames-dir /tmp/soccer_v2_frames --render
"""

import argparse
import json
import os
import random
import time

import cv2
import numpy as np

FPS   = 10.0
W, H  = 1920, 1080
FIELD_Y_MIN = int(0.18 * H)   # 194 — exclude crowd at top
FIELD_Y_MAX = int(0.86 * H)   # 928 — exclude advertising at bottom

K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

EXPECTED_AREA = np.pi * 9 ** 2  # ~254 px² for 9px-radius ball


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def load_frames(frames_dir):
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith('.jpg'))
    return [os.path.join(frames_dir, f) for f in files]


# ─────────────────────────────────────────────────────────────
# White-blob detector
# ─────────────────────────────────────────────────────────────

def detect_white_blobs(frame_bgr, pred_xy=None, search_r=None,
                       s_max=40, v_min=170,
                       area_min=80, area_max=700, max_aspect=3.5,
                       check_context=True):
    """
    Find white blob candidates on the field.
    Returns list of dicts sorted by total_score (descending).
    Each dict: {x, y, area, w, h, aspect, context, area_sc, circ_sc, total_sc, d}
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # White mask: low saturation + high brightness
    wm = ((hsv[:, :, 1] <= s_max) & (hsv[:, :, 2] >= v_min)).astype(np.uint8) * 255

    # Field boundary
    wm[:FIELD_Y_MIN, :] = 0
    wm[FIELD_Y_MAX:, :] = 0

    # Search window (optional)
    if pred_xy is not None and search_r is not None:
        px, py = int(pred_xy[0]), int(pred_xy[1])
        win = np.zeros((H, W), dtype=np.uint8)
        x1 = max(0, px - search_r); x2 = min(W, px + search_r)
        y1 = max(0, py - search_r); y2 = min(H, py + search_r)
        win[y1:y2, x1:x2] = 255
        wm = cv2.bitwise_and(wm, win)

    # Small morphological clean: remove single-pixel noise
    wm = cv2.morphologyEx(wm, cv2.MORPH_OPEN, K3)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(wm)

    results = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (area_min <= area <= area_max):
            continue

        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if bw == 0 or bh == 0:
            continue

        aspect = max(bw / bh, bh / bw)
        if aspect > max_aspect:
            continue

        cx = float(centroids[i][0])
        cy = float(centroids[i][1])

        if not (FIELD_Y_MIN <= cy <= FIELD_Y_MAX):
            continue

        # ── Context check: surrounding pixels should be green ──
        context = 0.0
        if check_context:
            r = max(bw, bh) // 2 + 8
            for dxy in [(0, -r), (0, r), (-r, 0), (r, 0)]:
                chx = int(cx + dxy[0])
                chy = int(cy + dxy[1])
                if 0 <= chx < W and 0 <= chy < H:
                    ph = hsv[chy, chx]
                    if 30 <= ph[0] <= 90 and ph[1] >= 25 and ph[2] >= 40:
                        context += 0.25

        # ── Scores ──
        area_sc = max(0.0, 1.0 - abs(area - EXPECTED_AREA) / max(EXPECTED_AREA, 1))
        circ_sc = min(bw, bh) / max(bw, bh)  # 1.0 = square/circle

        d = dist((cx, cy), pred_xy) if pred_xy is not None else 0.0
        prox_sc = 1.0 - d / (search_r + 1) if search_r is not None and search_r > 0 else 1.0
        prox_sc = max(0.0, prox_sc)

        total_sc = area_sc * 0.30 + circ_sc * 0.20 + context * 0.25 + prox_sc * 0.25

        results.append({
            'x': cx, 'y': cy, 'area': area, 'w': bw, 'h': bh,
            'aspect': aspect, 'context': context,
            'area_sc': area_sc, 'circ_sc': circ_sc,
            'total_sc': total_sc, 'd': d
        })

    results.sort(key=lambda r: r['total_sc'], reverse=True)
    return results


# ─────────────────────────────────────────────────────────────
# Motion-assisted white blob (for fast ball)
# ─────────────────────────────────────────────────────────────

def white_motion_blobs(frames, idx, step, pred_xy, search_r,
                       k=3, s_max=40, v_min=170,
                       area_min=60, area_max=800, max_aspect=4.0):
    """
    Intersect white mask with k-frame accumulated motion.
    Eliminates stationary white objects (field lines, standing players).
    Returns candidates (same format as detect_white_blobs).
    """
    ref_idx = idx - k * step
    if not (0 <= ref_idx < len(frames)):
        return []
    curr_bgr = cv2.imread(frames[idx])
    ref_bgr  = cv2.imread(frames[ref_idx])
    if curr_bgr is None or ref_bgr is None:
        return []

    gc  = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    gr  = cv2.cvtColor(ref_bgr,  cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gc, gr)
    _, mot = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
    mot = cv2.morphologyEx(mot, cv2.MORPH_OPEN, K3)
    mot = cv2.dilate(mot, K5, iterations=2)

    hsv = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2HSV)
    wm  = ((hsv[:, :, 1] <= s_max) & (hsv[:, :, 2] >= v_min)).astype(np.uint8) * 255
    wm  = cv2.bitwise_and(wm, mot)  # white AND moving

    wm[:FIELD_Y_MIN, :] = 0
    wm[FIELD_Y_MAX:, :] = 0

    if pred_xy is not None and search_r is not None:
        px, py = int(pred_xy[0]), int(pred_xy[1])
        win = np.zeros((H, W), dtype=np.uint8)
        x1 = max(0, px - search_r); x2 = min(W, px + search_r)
        y1 = max(0, py - search_r); y2 = min(H, py + search_r)
        win[y1:y2, x1:x2] = 255
        wm = cv2.bitwise_and(wm, win)

    wm = cv2.morphologyEx(wm, cv2.MORPH_OPEN, K3)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(wm)
    results = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (area_min <= area <= area_max):
            continue
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if bw == 0 or bh == 0:
            continue
        aspect = max(bw / bh, bh / bw)
        if aspect > max_aspect:
            continue
        cx = float(centroids[i][0])
        cy = float(centroids[i][1])
        if not (FIELD_Y_MIN <= cy <= FIELD_Y_MAX):
            continue
        area_sc = max(0.0, 1.0 - abs(area - EXPECTED_AREA) / max(EXPECTED_AREA, 1))
        circ_sc = min(bw, bh) / max(bw, bh)
        d = dist((cx, cy), pred_xy) if pred_xy is not None else 0.0
        prox_sc = max(0.0, 1.0 - d / (search_r + 1)) if search_r else 1.0
        total_sc = area_sc * 0.35 + circ_sc * 0.25 + prox_sc * 0.40
        results.append({'x': cx, 'y': cy, 'area': area, 'w': bw, 'h': bh,
                        'aspect': aspect, 'context': 1.0,
                        'area_sc': area_sc, 'circ_sc': circ_sc,
                        'total_sc': total_sc, 'd': d})
    results.sort(key=lambda r: r['total_sc'], reverse=True)
    return results


# ─────────────────────────────────────────────────────────────
# Kalman factory
# ─────────────────────────────────────────────────────────────

def make_kalman(x0, y0, q=1.0, r=80.0):
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)
    kf.processNoiseCov    = np.eye(4, dtype=np.float32) * q
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 50.0
    kf.statePost    = np.array([x0, y0, 0.0, 0.0], dtype=np.float32).reshape(-1, 1)
    return kf


# ─────────────────────────────────────────────────────────────
# Multi-frame stable-blob seeding
# ─────────────────────────────────────────────────────────────

def find_seed(frames, n=40, s_max=40, v_min=170):
    """
    Accumulate white-pixel density over first n frames.
    The stationary ball creates a consistent dense peak.
    Field lines are large and elongated (filtered by size).
    Returns (x, y).
    """
    accum = np.zeros((H, W), dtype=np.float32)
    count = 0
    for i in range(min(n, len(frames))):
        fr = cv2.imread(frames[i])
        if fr is None:
            continue
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        wm = ((hsv[:, :, 1] <= s_max) & (hsv[:, :, 2] >= v_min)).astype(np.float32)
        accum += wm
        count += 1

    if count == 0:
        return W // 2, H // 2

    accum /= count
    accum[:FIELD_Y_MIN, :] = 0
    accum[FIELD_Y_MAX:, :] = 0

    # Smooth: blob-sized kernel
    sm = cv2.GaussianBlur(accum, (15, 15), 4)

    # Threshold at 0.4 (appeared in ≥40% of frames)
    _, thresh = cv2.threshold(sm, 0.40, 1.0, cv2.THRESH_BINARY)
    thresh8 = (thresh * 255).astype(np.uint8)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(thresh8)

    # Filter by size (ball-like: 60–600 px², not elongated)
    best_x, best_y, best_sc = W // 2, H // 2, -1
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bw   = stats[i, cv2.CC_STAT_WIDTH]
        bh   = stats[i, cv2.CC_STAT_HEIGHT]
        cx   = float(centroids[i][0])
        cy   = float(centroids[i][1])
        if not (40 <= area <= 800):
            continue
        if max(bw, bh) == 0:
            continue
        aspect = max(bw / max(bh, 1), bh / max(bw, 1))
        if aspect > 4.0:
            continue
        # Score: prefer round, correct size, in center of field
        area_sc = max(0.0, 1.0 - abs(area - EXPECTED_AREA) / EXPECTED_AREA)
        circ_sc = min(bw, bh) / max(bw, bh)
        # Density at centroid
        dens = float(sm[int(cy), int(cx)])
        sc = area_sc * 0.4 + circ_sc * 0.3 + dens * 0.3
        if sc > best_sc:
            best_sc = sc
            best_x, best_y = int(cx), int(cy)

    # Fallback: if nothing found, use global peak of smoothed density
    if best_sc < 0:
        _, _, _, max_loc = cv2.minMaxLoc(sm)
        best_x, best_y = max_loc

    return best_x, best_y


# ─────────────────────────────────────────────────────────────
# Main tracker v5
# ─────────────────────────────────────────────────────────────

SLOW_SPEED_THRESH = 6.0   # px/frame — below this = "slow" mode


def track_v5(frames, seed_frame, seed_pos,
             # White blob params
             s_max=40, v_min=170,
             area_min=80, area_max=700, max_aspect=3.5,
             # Kalman params
             kalman_q=1.5, kalman_r=60.0,
             # Search
             search_r_base=80, search_r_max=280,
             # Thresholds
             min_score=0.20,
             max_lost_slow=6, max_lost_fast=4,
             # Motion-assist k
             mot_k=3):

    n = len(frames)
    positions = {}

    def run_dir(start_idx, start_pos, step):
        kf = make_kalman(start_pos[0], start_pos[1],
                         q=kalman_q, r=kalman_r)
        last_pos = start_pos
        positions[start_idx] = start_pos
        velocity  = (0.0, 0.0)
        lost      = 0
        slow_cnt  = 0  # consecutive slow frames

        idx = start_idx + step
        while 0 <= idx < n:
            # ── Kalman predict ──────────────────────────────────
            pred = kf.predict()
            px = int(clamp(float(pred[0, 0]), 0, W - 1))
            py = int(clamp(float(pred[1, 0]), 0, H - 1))

            speed = (velocity[0] ** 2 + velocity[1] ** 2) ** 0.5
            is_slow = speed < SLOW_SPEED_THRESH

            # Adaptive search radius
            sr = int(clamp(search_r_base + speed * 2.5,
                           search_r_base, search_r_max))

            curr_bgr = cv2.imread(frames[idx])
            if curr_bgr is None:
                positions[idx] = last_pos
                idx += step
                continue

            # ── Candidate detection ─────────────────────────────
            # Primary: white blobs in search window
            cands = detect_white_blobs(
                curr_bgr, pred_xy=(px, py), search_r=sr,
                s_max=s_max, v_min=v_min,
                area_min=area_min, area_max=area_max,
                max_aspect=max_aspect,
                check_context=True
            )

            # Motion-assist for fast ball (intersect with motion mask)
            if not is_slow:
                mot_cands = white_motion_blobs(
                    frames, idx, step, pred_xy=(px, py), search_r=sr,
                    k=mot_k, s_max=s_max, v_min=v_min,
                    area_min=area_min - 20, area_max=area_max + 200,
                    max_aspect=max_aspect + 0.5
                )
                # Boost score for motion-confirmed candidates
                for mc in mot_cands:
                    for c in cands:
                        if dist((mc['x'], mc['y']), (c['x'], c['y'])) < 15:
                            c['total_sc'] = min(1.0, c['total_sc'] + 0.30)
                            break
                    else:
                        mc['total_sc'] = min(1.0, mc['total_sc'] + 0.10)
                        cands.append(mc)
                cands.sort(key=lambda r: r['total_sc'], reverse=True)

            # ── Select best candidate ───────────────────────────
            best = cands[0] if cands else None

            if best is not None and best['total_sc'] >= min_score:
                bx, by = int(best['x']), int(best['y'])
                meas = np.array([[float(bx)], [float(by)]], dtype=np.float32)
                kf.correct(meas)
                vx = (bx - last_pos[0]) * 0.65 + velocity[0] * 0.35
                vy = (by - last_pos[1]) * 0.65 + velocity[1] * 0.35
                velocity  = (vx, vy)
                last_pos  = (bx, by)
                positions[idx] = (bx, by)
                lost = 0
                if is_slow:
                    slow_cnt += 1
                else:
                    slow_cnt = 0
            else:
                # No detection
                lost += 1
                max_lost = max_lost_slow if is_slow else max_lost_fast

                if lost <= max_lost:
                    # Hold prediction close to last confirmed
                    positions[idx] = last_pos
                else:
                    # ── Recovery: full-frame white blob search ──
                    all_cands = detect_white_blobs(
                        curr_bgr, pred_xy=None, search_r=None,
                        s_max=s_max, v_min=v_min,
                        area_min=area_min, area_max=area_max,
                        max_aspect=max_aspect,
                        check_context=True
                    )
                    if all_cands:
                        # Pick candidate nearest to last_pos that has decent score
                        def rec_score(c):
                            d_last = dist((c['x'], c['y']), last_pos)
                            return c['total_sc'] - d_last / 1000.0

                        all_cands.sort(key=rec_score, reverse=True)
                        r_best = all_cands[0]
                        rx, ry = int(r_best['x']), int(r_best['y'])
                        positions[idx] = (rx, ry)
                        last_pos = (rx, ry)
                        velocity = (0.0, 0.0)
                        lost = 0
                        kf = make_kalman(rx, ry, q=kalman_q, r=kalman_r)
                    else:
                        positions[idx] = last_pos

            idx += step

    run_dir(seed_frame, seed_pos, +1)
    run_dir(seed_frame, seed_pos, -1)
    return positions


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate(positions, labels, tol30=30, tol50=50):
    errs = []
    for lb in labels:
        f = lb['frame']
        if f not in positions:
            continue
        px, py = positions[f]
        gx = int(lb['px']); gy = int(lb['py'])
        errs.append(dist((px, py), (gx, gy)))
    if not errs:
        return 0, 0, 0, float('inf'), 0.0
    w30 = sum(1 for e in errs if e <= tol30)
    w50 = sum(1 for e in errs if e <= tol50)
    med = float(np.median(errs))
    tracked = len(errs)
    score = (w30 * 2 + w50) / (3 * len(labels))
    return tracked, w30, w50, med, score


# ─────────────────────────────────────────────────────────────
# Render
# ─────────────────────────────────────────────────────────────

def render_video(frames, positions, labels, out_path, fps=10):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    label_map = {lb['frame']: lb for lb in labels}
    for i, fp in enumerate(frames):
        fr = cv2.imread(fp)
        if fr is None:
            continue
        if i in positions:
            px, py = positions[i]
            cv2.circle(fr, (px, py), 10, (0, 255, 0), 2)
        if i in label_map:
            lb = label_map[i]
            gx, gy = int(lb['px']), int(lb['py'])
            cv2.circle(fr, (gx, gy), 10, (0, 0, 255), 2)
        vw.write(fr)
    vw.release()
    print(f"Saved {out_path}  (green=pred, red=gt)")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

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
    print(f"Loaded {len(labels)} labels")

    # Load/extract frames
    frames_dir = args.frames_dir or '/tmp/soccer_v5_frames'
    if args.frames_dir and os.path.isdir(args.frames_dir):
        frames = load_frames(args.frames_dir)
        print(f"Using {len(frames)} cached frames from {args.frames_dir}")
    else:
        os.makedirs(frames_dir, exist_ok=True)
        cmd = (f'ffmpeg -i "{args.video}" -vf fps={FPS} '
               f'-q:v 2 "{frames_dir}/frame_%06d.jpg" -y -loglevel error')
        os.system(cmd)
        frames = load_frames(frames_dir)
        print(f"Extracted {len(frames)} frames → {frames_dir}")

    # Seed — prefer ground-truth label at frame 0 if available
    print("\nFinding seed...")
    t0 = time.time()
    label_map_all = {lb['frame']: lb for lb in labels}
    if 0 in label_map_all:
        lb0 = label_map_all[0]
        sx, sy = int(lb0['px']), int(lb0['py'])
        seed_frame = 0
        print(f"  Seed from label: frame {seed_frame} → ({sx}, {sy})")
    else:
        # Use temporal average: average first 30 frames, find brightest compact blob
        accum = None
        for i in range(min(30, len(frames))):
            fr = cv2.imread(frames[i])
            if fr is None:
                continue
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            accum = gray if accum is None else accum + gray
        if accum is not None:
            avg = (accum / 30).astype(np.uint8)
            avg[:FIELD_Y_MIN, :] = 0
            avg[FIELD_Y_MAX:, :] = 0
            sm = cv2.GaussianBlur(avg, (15, 15), 4)
            _, _, _, max_loc = cv2.minMaxLoc(sm)
            sx, sy = max_loc
        else:
            sx, sy = W // 2, H // 2
        seed_frame = 0
        print(f"  Seed from temporal avg: frame {seed_frame} → ({sx}, {sy})  [{time.time()-t0:.1f}s]")
    seed_pos = (sx, sy)

    # ── Baseline run ─────────────────────────────────────────
    print("\n--- Baseline v5 ---")
    t0 = time.time()
    pos = track_v5(frames, seed_frame, seed_pos)
    tracked, w30, w50, med, score = evaluate(pos, labels)
    print(f"  Time: {time.time()-t0:.1f}s")
    print(f"  baseline_v5   {tracked}/{len(labels)}  "
          f"W30={w30:3d}  W50={w50:3d}  Med={med:.1f}  Score={score:.4f}")

    # ── Grid search ──────────────────────────────────────────
    print("\n--- Grid search v5 ---")

    hand_tuned = [
        # s_max v_min amin amax masp  q     r    srb  srm  msc  mls  mlf  k
        dict(s_max=40, v_min=165, area_min=70,  area_max=700, max_aspect=3.5,
             kalman_q=1.5, kalman_r=50,  search_r_base=90,  search_r_max=280,
             min_score=0.20, max_lost_slow=8, max_lost_fast=4, mot_k=3),
        dict(s_max=50, v_min=160, area_min=60,  area_max=900, max_aspect=4.0,
             kalman_q=2.0, kalman_r=80,  search_r_base=100, search_r_max=300,
             min_score=0.18, max_lost_slow=10, max_lost_fast=5, mot_k=3),
        dict(s_max=35, v_min=175, area_min=80,  area_max=600, max_aspect=3.0,
             kalman_q=1.0, kalman_r=40,  search_r_base=70,  search_r_max=250,
             min_score=0.22, max_lost_slow=6,  max_lost_fast=3, mot_k=2),
        dict(s_max=45, v_min=165, area_min=60,  area_max=800, max_aspect=3.5,
             kalman_q=3.0, kalman_r=100, search_r_base=120, search_r_max=320,
             min_score=0.16, max_lost_slow=10, max_lost_fast=6, mot_k=4),
        dict(s_max=30, v_min=180, area_min=100, area_max=550, max_aspect=3.0,
             kalman_q=0.8, kalman_r=30,  search_r_base=65,  search_r_max=240,
             min_score=0.25, max_lost_slow=5,  max_lost_fast=3, mot_k=2),
        dict(s_max=55, v_min=155, area_min=50,  area_max=1000,max_aspect=4.5,
             kalman_q=4.0, kalman_r=120, search_r_base=140, search_r_max=350,
             min_score=0.14, max_lost_slow=12, max_lost_fast=7, mot_k=3),
    ]

    random_combos = []
    for _ in range(44):
        random_combos.append(dict(
            s_max         = random.choice([30, 35, 40, 45, 50, 55]),
            v_min         = random.choice([155, 160, 165, 170, 175, 180, 185]),
            area_min      = random.choice([50, 70, 90, 110]),
            area_max      = random.choice([500, 650, 800, 1000]),
            max_aspect    = random.choice([2.5, 3.0, 3.5, 4.0, 5.0]),
            kalman_q      = random.choice([0.5, 1.0, 2.0, 4.0, 8.0]),
            kalman_r      = random.choice([20, 40, 80, 150, 250]),
            search_r_base = random.choice([60, 80, 100, 130, 160]),
            search_r_max  = random.choice([220, 280, 340, 400]),
            min_score     = random.choice([0.12, 0.16, 0.20, 0.25, 0.30]),
            max_lost_slow = random.choice([4, 6, 8, 12]),
            max_lost_fast = random.choice([2, 4, 6, 8]),
            mot_k         = random.choice([2, 3, 4]),
        ))

    all_combos = hand_tuned + random_combos
    print(f"  Running {len(all_combos)} combinations...")

    best_score = -1
    best_params = None
    best_pos    = None
    results_log = []

    for ci, params in enumerate(all_combos):
        pos = track_v5(frames, seed_frame, seed_pos, **params)
        tracked, w30, w50, med, score = evaluate(pos, labels)
        tag = (f"sm={params['s_max']} vm={params['v_min']} "
               f"am={params['area_min']} aM={params['area_max']} "
               f"q={params['kalman_q']} r={params['kalman_r']} "
               f"sr={params['search_r_base']} ms={params['min_score']:.2f} "
               f"k={params['mot_k']}")
        results_log.append((score, w30, w50, med, tracked, tag, params, pos))
        if score > best_score:
            best_score  = score
            best_params = params
            best_pos    = pos
        if (ci + 1) % 10 == 0:
            print(f"    {ci+1}/{len(all_combos)} done — "
                  f"best: score={best_score:.4f} w50={results_log[0][2] if best_pos else 0} "
                  f"med={results_log[0][3] if best_pos else 0:.0f}px")

    results_log.sort(key=lambda r: r[0], reverse=True)

    print(f"\n{'='*80}")
    print(f"  TOP 10:")
    print(f"{'='*80}")
    for sc, w30, w50, med, tr, tag, _, _ in results_log[:10]:
        print(f"  {tag}  {tr}/{len(labels)}  W30={w30:3d}  W50={w50:3d}  "
              f"Med={med:.1f}  Score={sc:.4f}")

    winner = results_log[0]
    best_score, best_w30, best_w50, best_med, best_tracked, best_tag, best_params, best_pos = winner

    print(f"\n{'='*80}")
    print(f"  WINNER: {best_params}")
    print(f"  Score={best_score:.4f}  Tracked={best_tracked}/{len(labels)}  "
          f"W30={best_w30}  W50={best_w50}  Median={best_med:.1f}px")
    print(f"{'='*80}")

    # Per-frame detail for winner
    print("\nPer-frame detail (winner):")
    label_map = {lb['frame']: lb for lb in labels}
    for f_idx in sorted(label_map):
        lb = label_map[f_idx]
        gx, gy = int(lb['px']), int(lb['py'])
        if f_idx in best_pos:
            px, py = best_pos[f_idx]
            err = dist((px, py), (gx, gy))
            flag = ' <-- MISS' if err > 50 else ''
            print(f"  frame {f_idx:4d}: err={err:6.1f}px  "
                  f"pred=({px:4d},{py:4d})  gt=({gx:4d},{gy:4d}){flag}")
        else:
            print(f"  frame {f_idx:4d}: NO PREDICTION  gt=({gx:4d},{gy:4d}) <-- MISS")

    # Save results
    with open('tracking_analysis_v5.txt', 'w') as f:
        f.write(f"V5 Winner: score={best_score:.4f} w30={best_w30} "
                f"w50={best_w50} med={best_med:.1f}\n")
        f.write(f"Params: {best_params}\n\n")
        for sc, w30, w50, med, tr, tag, _, _ in results_log[:20]:
            f.write(f"{sc:.4f}  W30={w30}  W50={w50}  Med={med:.1f}  {tag}\n")
    print("\nSaved to tracking_analysis_v5.txt")

    if args.render:
        out = 'tracking_v5_winner.mp4'
        print(f"\nRendering {out}...")
        render_video(frames, best_pos, labels, out)


if __name__ == '__main__':
    main()
