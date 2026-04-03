#!/usr/bin/env python3
"""
Ball tracking v5b — White-blob Kalman with TIGHT stationary window + kick detection.

Root cause diagnosis from v5 run 1:
  - Ball at (966,546), center circle arc at (963,639), both detected as white blobs.
  - Linear proximity weight wasn't enough to prefer the ball (33px) over arc (90px).
  - The arc is ALWAYS present (field marking), ball drifts slowly during frames 0-175.

Fixes:
  1. TIGHT search window in stationary mode (sr_tight ≈ 35-50px).
     Arc is 90px away → outside window → eliminated.
  2. Parallel kick detector: k-frame accumulated motion, wide radius (150-250px).
     Runs every frame during stationary mode; when kick found → switch to FAST mode.
  3. Gaussian proximity weight: exp(-d²/2σ²) instead of linear (1-d/r).
     Makes far candidates score nearly zero, not just slightly lower.
  4. Adaptive Kalman Q/R: low Q in slow mode (trust prior), high Q in fast mode.
  5. Recovery: search centered on last_confirmed_pos, not drifted Kalman prediction.

Architecture:
  SLOW  — stationary ball. sr=sr_tight, white blob only. Kick detector runs parallel.
  FAST  — kicked ball. sr = base + speed*k. White+motion combo.
  LOST  — no detection for N frames. Full-frame recovery search.

Usage:
  .venv/bin/python3 analyze_tracking_v5.py --video Test.mp4 \
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
FIELD_Y_MIN = int(0.18 * H)   # 194
FIELD_Y_MAX = int(0.86 * H)   # 928

K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

EXPECTED_AREA = np.pi * 9 ** 2   # ~254 px² for 9px radius ball


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def load_frames(frames_dir):
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith('.jpg'))
    return [os.path.join(frames_dir, f) for f in files]


# ─────────────────────────────────────────────────────────────
# White-blob detector (with Gaussian proximity)
# ─────────────────────────────────────────────────────────────

def detect_white_blobs(frame_bgr, pred_xy=None, search_r=None,
                       s_max=42, v_min=168,
                       area_min=80, area_max=700, max_aspect=3.5,
                       check_context=True):
    """
    Find white blob candidates on the field.
    Returns list of dicts sorted by total_sc (descending).
    Proximity scored with Gaussian: exp(-d²/(2σ²)), σ = sr/3.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    wm  = ((hsv[:,:,1] <= s_max) & (hsv[:,:,2] >= v_min)).astype(np.uint8) * 255
    wm[:FIELD_Y_MIN, :] = 0
    wm[FIELD_Y_MAX:, :]  = 0

    if pred_xy is not None and search_r is not None:
        px, py = int(pred_xy[0]), int(pred_xy[1])
        win = np.zeros((H, W), dtype=np.uint8)
        x1 = max(0,px-search_r); x2 = min(W,px+search_r)
        y1 = max(0,py-search_r); y2 = min(H,py+search_r)
        win[y1:y2,x1:x2] = 255
        wm = cv2.bitwise_and(wm, win)

    wm = cv2.morphologyEx(wm, cv2.MORPH_OPEN, K3)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(wm)

    sigma = (search_r / 3.0) if search_r is not None else 80.0

    results = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (area_min <= area <= area_max):
            continue
        bw  = stats[i, cv2.CC_STAT_WIDTH]
        bh  = stats[i, cv2.CC_STAT_HEIGHT]
        if bw == 0 or bh == 0:
            continue
        aspect = max(bw/bh, bh/bw)
        if aspect > max_aspect:
            continue
        cx = float(centroids[i][0])
        cy = float(centroids[i][1])
        if not (FIELD_Y_MIN <= cy <= FIELD_Y_MAX):
            continue

        # Context: surrounding pixels should be green
        context = 0.0
        if check_context:
            r = max(bw, bh)//2 + 8
            for dxy in [(0,-r),(0,r),(-r,0),(r,0)]:
                chx = int(cx+dxy[0]); chy = int(cy+dxy[1])
                if 0<=chx<W and 0<=chy<H:
                    ph = hsv[chy, chx]
                    if 28<=ph[0]<=92 and ph[1]>=22 and ph[2]>=40:
                        context += 0.25

        area_sc = max(0.0, 1.0 - abs(area-EXPECTED_AREA)/max(EXPECTED_AREA,1))
        circ_sc = min(bw,bh)/max(bw,bh)

        # Gaussian proximity — exp(-d²/(2σ²))
        if pred_xy is not None:
            d = dist((cx,cy), pred_xy)
            prox_sc = float(np.exp(-d**2 / (2*sigma**2)))
        else:
            d = 0.0
            prox_sc = 1.0

        total_sc = area_sc*0.25 + circ_sc*0.15 + context*0.20 + prox_sc*0.40

        results.append({'x':cx,'y':cy,'area':area,'w':bw,'h':bh,
                        'aspect':aspect,'context':context,
                        'area_sc':area_sc,'circ_sc':circ_sc,
                        'prox_sc':prox_sc,'total_sc':total_sc,'d':d})

    results.sort(key=lambda r: r['total_sc'], reverse=True)
    return results


# ─────────────────────────────────────────────────────────────
# Motion + white blobs (for fast ball)
# ─────────────────────────────────────────────────────────────

def white_motion_blobs(frames, idx, step, pred_xy, search_r,
                       k=3, s_max=42, v_min=168,
                       area_min=50, area_max=900, max_aspect=4.0):
    """Intersect white mask with k-frame accumulated motion."""
    ref_idx = idx - k*step
    if not (0 <= ref_idx < len(frames)):
        return []
    curr = cv2.imread(frames[idx])
    ref  = cv2.imread(frames[ref_idx])
    if curr is None or ref is None:
        return []

    gc = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(ref,  cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gc, gr)
    _, mot = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    mot = cv2.morphologyEx(mot, cv2.MORPH_OPEN, K3)
    mot = cv2.dilate(mot, K5, iterations=2)

    hsv = cv2.cvtColor(curr, cv2.COLOR_BGR2HSV)
    wm  = ((hsv[:,:,1] <= s_max) & (hsv[:,:,2] >= v_min)).astype(np.uint8)*255
    wm  = cv2.bitwise_and(wm, mot)
    wm[:FIELD_Y_MIN,:] = 0
    wm[FIELD_Y_MAX:,:] = 0

    if pred_xy is not None and search_r is not None:
        px,py = int(pred_xy[0]),int(pred_xy[1])
        win = np.zeros((H,W),dtype=np.uint8)
        x1=max(0,px-search_r); x2=min(W,px+search_r)
        y1=max(0,py-search_r); y2=min(H,py+search_r)
        win[y1:y2,x1:x2]=255
        wm=cv2.bitwise_and(wm,win)

    wm = cv2.morphologyEx(wm, cv2.MORPH_OPEN, K3)
    num_labels,_,stats,centroids = cv2.connectedComponentsWithStats(wm)
    sigma = (search_r/3.0) if search_r else 80.0

    results = []
    for i in range(1,num_labels):
        area = stats[i,cv2.CC_STAT_AREA]
        if not (area_min<=area<=area_max): continue
        bw=stats[i,cv2.CC_STAT_WIDTH]; bh=stats[i,cv2.CC_STAT_HEIGHT]
        if bw==0 or bh==0: continue
        aspect=max(bw/bh,bh/bw)
        if aspect>max_aspect: continue
        cx=float(centroids[i][0]); cy=float(centroids[i][1])
        if not (FIELD_Y_MIN<=cy<=FIELD_Y_MAX): continue
        area_sc=max(0.0,1.0-abs(area-EXPECTED_AREA)/max(EXPECTED_AREA,1))
        circ_sc=min(bw,bh)/max(bw,bh)
        d=dist((cx,cy),pred_xy) if pred_xy else 0.0
        prox_sc=float(np.exp(-d**2/(2*sigma**2))) if pred_xy else 1.0
        total_sc=area_sc*0.30+circ_sc*0.20+prox_sc*0.50
        results.append({'x':cx,'y':cy,'area':area,'w':bw,'h':bh,
                        'aspect':aspect,'context':1.0,
                        'area_sc':area_sc,'circ_sc':circ_sc,
                        'prox_sc':prox_sc,'total_sc':total_sc,'d':d})
    results.sort(key=lambda r: r['total_sc'], reverse=True)
    return results


# ─────────────────────────────────────────────────────────────
# Kick detector (motion blob far from last position)
# ─────────────────────────────────────────────────────────────

def detect_kick(frames, idx, step, last_pos, search_r=220, min_disp=30,
                k=3, s_max=42, v_min=168):
    """
    Scan wide radius for white+motion blob that has moved far from last_pos.
    Returns (x, y, score) or None.
    """
    ref_idx = idx - k*step
    if not (0 <= ref_idx < len(frames)):
        return None
    curr = cv2.imread(frames[idx])
    ref  = cv2.imread(frames[ref_idx])
    if curr is None or ref is None:
        return None

    gc = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(ref,  cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gc, gr)
    _, mot = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    mot = cv2.morphologyEx(mot, cv2.MORPH_OPEN, K3)
    mot = cv2.dilate(mot, K5, iterations=2)

    px,py = int(last_pos[0]),int(last_pos[1])
    x1=max(0,px-search_r); x2=min(W,px+search_r)
    y1=max(0,py-search_r); y2=min(H,py+search_r)

    hsv = cv2.cvtColor(curr, cv2.COLOR_BGR2HSV)
    wm  = ((hsv[:,:,1] <= s_max) & (hsv[:,:,2] >= v_min)).astype(np.uint8)*255
    wm  = cv2.bitwise_and(wm, mot)
    wm[:FIELD_Y_MIN,:]=0; wm[FIELD_Y_MAX:,:]=0

    roi_wm = wm[y1:y2,x1:x2]
    roi_wm = cv2.morphologyEx(roi_wm, cv2.MORPH_OPEN, K3)
    num_labels,_,stats,centroids = cv2.connectedComponentsWithStats(roi_wm)

    best=None; best_sc=0
    for i in range(1,num_labels):
        area=stats[i,cv2.CC_STAT_AREA]
        if not (40<=area<=1000): continue
        bw=stats[i,cv2.CC_STAT_WIDTH]; bh=stats[i,cv2.CC_STAT_HEIGHT]
        if bw==0 or bh==0: continue
        if max(bw/bh,bh/bw)>5.0: continue
        cx=float(centroids[i][0])+x1; cy=float(centroids[i][1])+y1
        if not (FIELD_Y_MIN<=cy<=FIELD_Y_MAX): continue
        disp=dist((cx,cy),last_pos)
        if disp<min_disp: continue
        area_sc=max(0.0,1.0-abs(area-EXPECTED_AREA)/max(EXPECTED_AREA,1))
        circ_sc=min(bw,bh)/max(bw,bh)
        sc=area_sc*0.4+circ_sc*0.3+min(disp/100.0,1.0)*0.3
        if sc>best_sc:
            best_sc=sc; best=(cx,cy,sc)
    return best if best_sc>0.10 else None


# ─────────────────────────────────────────────────────────────
# Kalman factory
# ─────────────────────────────────────────────────────────────

def make_kalman(x0, y0, q=1.0, r=60.0):
    kf = cv2.KalmanFilter(4,2)
    kf.transitionMatrix  = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],dtype=np.float32)
    kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],dtype=np.float32)
    kf.processNoiseCov    = np.eye(4,dtype=np.float32)*q
    kf.measurementNoiseCov= np.eye(2,dtype=np.float32)*r
    kf.errorCovPost = np.eye(4,dtype=np.float32)*50.0
    kf.statePost    = np.array([x0,y0,0.0,0.0],dtype=np.float32).reshape(-1,1)
    return kf


# ─────────────────────────────────────────────────────────────
# Main tracker v5b
# ─────────────────────────────────────────────────────────────

SLOW = "SLOW"
FAST = "FAST"
LOST = "LOST"


def track_v5(frames, seed_frame, seed_pos,
             # White blob
             s_max=42, v_min=168,
             area_min=80, area_max=700, max_aspect=3.5,
             # Kalman
             q_slow=0.3, r_slow=30.0,
             q_fast=4.0,  r_fast=80.0,
             # Search radii
             sr_tight=40,        # stationary mode — keeps field markings out
             sr_base=80,         # fast mode base
             sr_max=300,
             # Speed threshold for state switch
             slow_speed=6.0,
             # Loss limits
             max_lost_slow=5, max_lost_fast=4, max_lost_lost=20,
             # Min detection score
             min_score=0.20,
             # Motion / kick params
             mot_k=3,
             kick_search_r=220, kick_min_disp=28):

    n = len(frames)
    positions = {}

    def run_dir(start_idx, start_pos, step):
        kf = make_kalman(start_pos[0], start_pos[1], q=q_slow, r=r_slow)
        positions[start_idx] = start_pos
        last_pos   = start_pos
        last_conf  = start_pos   # last position from actual detection (not hold)
        velocity   = (0.0, 0.0)
        state      = SLOW
        lost       = 0
        slow_cnt   = 0

        idx = start_idx + step
        while 0 <= idx < n:
            pred = kf.predict()
            px = int(clamp(float(pred[0,0]), 0, W-1))
            py = int(clamp(float(pred[1,0]), 0, H-1))

            speed = (velocity[0]**2 + velocity[1]**2)**0.5

            # ── Choose search radius based on state ──────────────
            if state == SLOW:
                sr = sr_tight
            elif state == FAST:
                sr = int(clamp(sr_base + speed*2.5, sr_base, sr_max))
            else:  # LOST
                sr = sr_base

            curr_bgr = cv2.imread(frames[idx])
            if curr_bgr is None:
                positions[idx] = last_pos
                idx += step; continue

            # ── Kick detection (always in SLOW mode, also in FAST) ─
            kick = None
            if state in (SLOW, FAST):
                kick = detect_kick(frames, idx, step, last_pos,
                                   search_r=kick_search_r,
                                   min_disp=kick_min_disp,
                                   k=mot_k, s_max=s_max, v_min=v_min)

            # ── Primary detection ─────────────────────────────────
            if state == SLOW:
                cands = detect_white_blobs(
                    curr_bgr, pred_xy=(px,py), search_r=sr,
                    s_max=s_max, v_min=v_min,
                    area_min=area_min, area_max=area_max,
                    max_aspect=max_aspect)
            elif state == FAST:
                # Prefer motion-confirmed blobs
                cands = white_motion_blobs(
                    frames, idx, step, pred_xy=(px,py), search_r=sr,
                    k=mot_k, s_max=s_max, v_min=v_min,
                    area_min=area_min-20, area_max=area_max+200)
                if not cands:
                    cands = detect_white_blobs(
                        curr_bgr, pred_xy=(px,py), search_r=sr,
                        s_max=s_max, v_min=v_min,
                        area_min=area_min, area_max=area_max,
                        max_aspect=max_aspect)
            else:  # LOST
                # Recovery: search around last confirmed position (not drifted pred)
                cands = detect_white_blobs(
                    curr_bgr, pred_xy=last_conf, search_r=sr_base+80,
                    s_max=s_max, v_min=v_min,
                    area_min=area_min, area_max=area_max,
                    max_aspect=max_aspect)
                if not cands:
                    cands = detect_white_blobs(
                        curr_bgr, pred_xy=None, search_r=None,
                        s_max=s_max, v_min=v_min,
                        area_min=area_min, area_max=area_max,
                        max_aspect=max_aspect, check_context=True)
                    if cands:
                        # Pick nearest to last_conf
                        cands.sort(key=lambda c: dist((c['x'],c['y']),last_conf))

            best = cands[0] if cands else None

            # ── Update from kick detection ─────────────────────────
            if kick is not None and (best is None or
                                      kick[2] > (best['total_sc'] if best else 0)):
                kx, ky, kscore = kick
                # Verify kick blob is within field
                if FIELD_Y_MIN <= ky <= FIELD_Y_MAX:
                    new_pos = (int(kx), int(ky))
                    positions[idx] = new_pos
                    vx = (kx-last_pos[0])*0.7+velocity[0]*0.3
                    vy = (ky-last_pos[1])*0.7+velocity[1]*0.3
                    velocity = (vx,vy)
                    last_pos = new_pos; last_conf = new_pos
                    meas = np.array([[kx],[ky]],dtype=np.float32)
                    kf = make_kalman(kx,ky,q=q_fast,r=r_fast)
                    kf.correct(meas)
                    state=FAST; lost=0; slow_cnt=0
                    idx+=step; continue

            # ── Update from primary detection ──────────────────────
            if best is not None and best['total_sc'] >= min_score:
                bx, by = int(best['x']), int(best['y'])
                meas = np.array([[float(bx)],[float(by)]],dtype=np.float32)
                kf.correct(meas)
                vx=(bx-last_pos[0])*0.65+velocity[0]*0.35
                vy=(by-last_pos[1])*0.65+velocity[1]*0.35
                velocity=(vx,vy)
                last_pos=(bx,by); last_conf=(bx,by)
                positions[idx]=(bx,by)
                lost=0

                new_speed=(vx**2+vy**2)**0.5
                if new_speed < slow_speed:
                    slow_cnt+=1
                    if slow_cnt>=3 and state==FAST:
                        state=SLOW
                        kf=make_kalman(bx,by,q=q_slow,r=r_slow)
                else:
                    slow_cnt=0
                    if state==SLOW:
                        state=FAST
                        kf=make_kalman(bx,by,q=q_fast,r=r_fast)
                    elif state==LOST:
                        state=FAST
                        kf=make_kalman(bx,by,q=q_fast,r=r_fast)
            else:
                lost+=1
                positions[idx]=last_pos

                max_lost = (max_lost_slow if state==SLOW
                            else max_lost_fast if state==FAST
                            else max_lost_lost)
                if lost > max_lost and state != LOST:
                    state=LOST
                    lost=0

            idx+=step

    run_dir(seed_frame, seed_pos, +1)
    run_dir(seed_frame, seed_pos, -1)
    return positions


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate(positions, labels, tol30=30, tol50=50):
    errs=[]
    for lb in labels:
        f=lb['frame']
        if f not in positions: continue
        px,py=positions[f]
        errs.append(dist((px,py),(int(lb['px']),int(lb['py']))))
    if not errs:
        return 0,0,0,float('inf'),0.0
    w30=sum(1 for e in errs if e<=tol30)
    w50=sum(1 for e in errs if e<=tol50)
    med=float(np.median(errs))
    score=(w30*2+w50)/(3*len(labels))
    return len(errs),w30,w50,med,score


# ─────────────────────────────────────────────────────────────
# Render
# ─────────────────────────────────────────────────────────────

def render_video(frames, positions, labels, out_path, fps=10):
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    vw=cv2.VideoWriter(out_path,fourcc,fps,(W,H))
    lmap={lb['frame']:lb for lb in labels}
    for i,fp in enumerate(frames):
        fr=cv2.imread(fp)
        if fr is None: continue
        if i in positions:
            px,py=positions[i]
            cv2.circle(fr,(px,py),10,(0,255,0),2)
        if i in lmap:
            lb=lmap[i]
            cv2.circle(fr,(int(lb['px']),int(lb['py'])),10,(0,0,255),2)
        vw.write(fr)
    vw.release()
    print(f"Saved {out_path}  (green=pred, red=gt)")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--video',      required=True)
    ap.add_argument('--labels',     required=True)
    ap.add_argument('--frames-dir', default='')
    ap.add_argument('--render',     action='store_true')
    args=ap.parse_args()

    with open(args.labels) as f:
        data=json.load(f)
    labels=data['labels']
    print(f"Loaded {len(labels)} labels")

    frames_dir=args.frames_dir or '/tmp/soccer_v5_frames'
    if args.frames_dir and os.path.isdir(args.frames_dir):
        frames=load_frames(args.frames_dir)
        print(f"Using {len(frames)} cached frames from {args.frames_dir}")
    else:
        os.makedirs(frames_dir,exist_ok=True)
        os.system(f'ffmpeg -i "{args.video}" -vf fps={FPS} '
                  f'-q:v 2 "{frames_dir}/frame_%06d.jpg" -y -loglevel error')
        frames=load_frames(frames_dir)
        print(f"Extracted {len(frames)} frames")

    # Seed from first label
    lmap={lb['frame']:lb for lb in labels}
    if 0 in lmap:
        lb0=lmap[0]
        seed_pos=(int(lb0['px']),int(lb0['py']))
        print(f"\nSeed from label: frame 0 → {seed_pos}")
    else:
        seed_pos=(W//2,H//2)
        print(f"\nSeed: frame 0 → {seed_pos} (default center)")
    seed_frame=0

    # ── Baseline ────────────────────────────────────────────
    print("\n--- Baseline v5b ---")
    t0=time.time()
    pos=track_v5(frames,seed_frame,seed_pos)
    tracked,w30,w50,med,score=evaluate(pos,labels)
    print(f"  Time: {time.time()-t0:.1f}s")
    print(f"  baseline_v5b  {tracked}/{len(labels)}  W30={w30:3d}  W50={w50:3d}  "
          f"Med={med:.1f}  Score={score:.4f}")

    # ── Grid search ─────────────────────────────────────────
    print("\n--- Grid search v5b ---")

    hand_tuned=[
        # Core parameter sweeps targeting the identified failure modes
        dict(s_max=42,v_min=168,area_min=80,area_max=700,max_aspect=3.5,
             q_slow=0.3,r_slow=30,q_fast=4.0,r_fast=80,
             sr_tight=40,sr_base=80,sr_max=300,slow_speed=6.0,
             max_lost_slow=5,max_lost_fast=4,max_lost_lost=20,
             min_score=0.20,mot_k=3,kick_search_r=220,kick_min_disp=28),
        dict(s_max=42,v_min=168,area_min=80,area_max=700,max_aspect=3.5,
             q_slow=0.2,r_slow=20,q_fast=6.0,r_fast=60,
             sr_tight=35,sr_base=100,sr_max=320,slow_speed=5.0,
             max_lost_slow=4,max_lost_fast=3,max_lost_lost=15,
             min_score=0.18,mot_k=3,kick_search_r=240,kick_min_disp=25),
        dict(s_max=50,v_min=160,area_min=60,area_max=900,max_aspect=4.0,
             q_slow=0.5,r_slow=40,q_fast=8.0,r_fast=100,
             sr_tight=45,sr_base=90,sr_max=280,slow_speed=7.0,
             max_lost_slow=6,max_lost_fast=5,max_lost_lost=20,
             min_score=0.16,mot_k=2,kick_search_r=200,kick_min_disp=30),
        dict(s_max=35,v_min=175,area_min=90,area_max=600,max_aspect=3.0,
             q_slow=0.15,r_slow=15,q_fast=5.0,r_fast=50,
             sr_tight=38,sr_base=75,sr_max=260,slow_speed=5.0,
             max_lost_slow=4,max_lost_fast=3,max_lost_lost=12,
             min_score=0.22,mot_k=3,kick_search_r=210,kick_min_disp=32),
        dict(s_max=45,v_min=165,area_min=70,area_max=800,max_aspect=3.5,
             q_slow=0.4,r_slow=35,q_fast=5.0,r_fast=70,
             sr_tight=42,sr_base=85,sr_max=290,slow_speed=6.0,
             max_lost_slow=5,max_lost_fast=4,max_lost_lost=18,
             min_score=0.20,mot_k=4,kick_search_r=230,kick_min_disp=26),
        dict(s_max=40,v_min=170,area_min=75,area_max=750,max_aspect=3.5,
             q_slow=0.3,r_slow=25,q_fast=10.0,r_fast=120,
             sr_tight=45,sr_base=100,sr_max=350,slow_speed=8.0,
             max_lost_slow=6,max_lost_fast=5,max_lost_lost=25,
             min_score=0.18,mot_k=3,kick_search_r=250,kick_min_disp=22),
    ]

    random_combos=[]
    for _ in range(44):
        random_combos.append(dict(
            s_max        =random.choice([30,35,40,45,50,55]),
            v_min        =random.choice([155,160,165,170,175,180]),
            area_min     =random.choice([50,70,90,110]),
            area_max     =random.choice([500,650,800,1000]),
            max_aspect   =random.choice([2.5,3.0,3.5,4.0,5.0]),
            q_slow       =random.choice([0.1,0.2,0.4,0.8,1.5]),
            r_slow       =random.choice([10,20,35,60,100]),
            q_fast       =random.choice([2.0,4.0,6.0,10.0,15.0]),
            r_fast       =random.choice([40,70,120,200]),
            sr_tight     =random.choice([25,30,35,40,48,55]),
            sr_base      =random.choice([60,80,100,130,160]),
            sr_max       =random.choice([240,300,360,420]),
            slow_speed   =random.choice([4.0,5.0,6.0,8.0,10.0]),
            max_lost_slow=random.choice([3,5,7,10]),
            max_lost_fast=random.choice([2,4,6,8]),
            max_lost_lost=random.choice([10,15,20,30]),
            min_score    =random.choice([0.12,0.16,0.20,0.25,0.30]),
            mot_k        =random.choice([2,3,4]),
            kick_search_r=random.choice([160,200,240,280]),
            kick_min_disp=random.choice([20,25,30,38,50]),
        ))

    all_combos=hand_tuned+random_combos
    print(f"  Running {len(all_combos)} combinations...")

    results_log=[]
    best_score=-1; best_pos=None; best_params=None

    for ci,params in enumerate(all_combos):
        pos=track_v5(frames,seed_frame,seed_pos,**params)
        tracked,w30,w50,med,score=evaluate(pos,labels)
        tag=(f"st={params['sr_tight']} sb={params['sr_base']} ss={params['slow_speed']} "
             f"qs={params['q_slow']} rs={params['r_slow']} "
             f"qf={params['q_fast']} rf={params['r_fast']} "
             f"sm={params['s_max']} vm={params['v_min']} k={params['mot_k']}")
        results_log.append((score,w30,w50,med,tracked,tag,params,pos))
        if score>best_score:
            best_score=score; best_params=params; best_pos=pos
        if (ci+1)%10==0:
            top=sorted(results_log,key=lambda r:r[0],reverse=True)[0]
            print(f"    {ci+1}/{len(all_combos)} done — "
                  f"best: score={top[0]:.4f} w30={top[1]} w50={top[2]} med={top[3]:.0f}px")

    results_log.sort(key=lambda r:r[0],reverse=True)

    print(f"\n{'='*80}")
    print(f"  TOP 10:")
    print(f"{'='*80}")
    for sc,w30,w50,med,tr,tag,_,_ in results_log[:10]:
        print(f"  {tag}  {tr}/{len(labels)}  W30={w30:3d}  W50={w50:3d}  "
              f"Med={med:.1f}  Score={sc:.4f}")

    sc,w30,w50,med,tr,tag,best_params,best_pos=results_log[0]
    print(f"\n{'='*80}")
    print(f"  WINNER: {best_params}")
    print(f"  Score={sc:.4f}  Tracked={tr}/{len(labels)}  "
          f"W30={w30}  W50={w50}  Median={med:.1f}px")
    print(f"{'='*80}")

    print("\nPer-frame detail (winner):")
    for f_idx in sorted(lmap):
        lb=lmap[f_idx]; gx,gy=int(lb['px']),int(lb['py'])
        if f_idx in best_pos:
            px,py=best_pos[f_idx]
            err=dist((px,py),(gx,gy))
            flag=' <-- MISS' if err>50 else ''
            print(f"  frame {f_idx:4d}: err={err:6.1f}px  "
                  f"pred=({px:4d},{py:4d})  gt=({gx:4d},{gy:4d}){flag}")
        else:
            print(f"  frame {f_idx:4d}: NO PRED  gt=({gx:4d},{gy:4d}) <-- MISS")

    with open('tracking_analysis_v5.txt','w') as f:
        f.write(f"V5b Winner: score={sc:.4f} w30={w30} w50={w50} med={med:.1f}\n")
        f.write(f"Params: {best_params}\n\n")
        for s,w3,w5,md,t,tg,_,_ in results_log[:20]:
            f.write(f"{s:.4f}  W30={w3}  W50={w5}  Med={md:.1f}  {tg}\n")
    print("\nSaved to tracking_analysis_v5.txt")

    if args.render:
        print("\nRendering tracking_v5_winner.mp4...")
        render_video(frames,best_pos,labels,'tracking_v5_winner.mp4')


if __name__=='__main__':
    main()
