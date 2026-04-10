#!/usr/bin/env python3
"""
TrackNet-style soccer ball detector.

A 9-channel U-Net: takes 3 consecutive RGB frames stacked as input,
outputs a Gaussian heatmap of ball probability. This exploits motion
blur and temporal context — two things frame-by-frame YOLO misses.

Pipeline:
  1. Run build_cache_15fps.py  →  frames_15fps/  +  yolo_cache_test1_15fps.json
  2. python tracknet.py train  →  tracknet_model.pth
  3. python tracknet.py infer  →  tracknet_cache_15fps.json
  4. python analyze_tracking_v9.py ...

Train:
  .venv/bin/python3 tracknet.py train
      [--frames_dir frames_15fps]
      [--labels ball_labels.json]
      [--yolo_cache yolo_cache_test1_15fps.json]
      [--model tracknet_model.pth]
      [--epochs 80] [--batch 4] [--lr 1e-4]
      [--pseudo_conf 0.35]

Infer:
  .venv/bin/python3 tracknet.py infer
      [--frames_dir frames_15fps]
      [--model tracknet_model.pth]
      [--output tracknet_cache_15fps.json]
      [--conf 0.10]
"""

import argparse
import json
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Resolution ──────────────────────────────────────────────────────────────
INPUT_W = 640   # 1920 / 3
INPUT_H = 360   # 1080 / 3
ORIG_W  = 1920
ORIG_H  = 1080
SIGMA   = 5     # Gaussian heatmap sigma in pixels at INPUT resolution
FPS     = 15.0


# ── Model ────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TrackNet(nn.Module):
    """
    U-Net for ball heatmap prediction.

    Input:  [B, 9, H, W]  — 3 consecutive frames, RGB concatenated
    Output: [B, 1, H, W]  — raw logits (apply sigmoid for probabilities)
    """
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.enc1 = ConvBlock(9,   64)
        self.enc2 = ConvBlock(64,  128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 256, dropout=0.3)   # bottleneck + dropout

        # Decoder
        self.up3  = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)   # 256 up + 256 skip

        self.up2  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)   # 128 up + 128 skip

        self.up1  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)    # 64 up + 64 skip

        self.head = nn.Conv2d(64, 1, 1)   # raw logits

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)   # [B, 1, H, W] — logits


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def make_heatmap(px, py, w=INPUT_W, h=INPUT_H, sigma=SIGMA):
    """Gaussian heatmap centered at (px, py) in INPUT resolution. Returns float32 [0,1]."""
    if px < 0 or py < 0:
        return np.zeros((h, w), dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    hm = np.exp(-((X - px) ** 2 + (Y - py) ** 2) / (2 * sigma ** 2))
    return hm.astype(np.float32)


def load_frame(path):
    """Load image, resize to INPUT resolution, return float32 [H, W, 3] in [0, 1]."""
    img = cv2.imread(path)
    if img is None:
        return np.zeros((INPUT_H, INPUT_W, 3), dtype=np.float32)
    img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


# ── Dataset ──────────────────────────────────────────────────────────────────

class BallDataset(Dataset):
    """
    Each sample: triplet of consecutive 15fps frames → Gaussian heatmap at ball position.

    samples: list of dicts with keys:
        frame_indices: [prev, cur, next]  (indices into sorted frame file list)
        px, py: ball position in ORIG resolution (-1,-1 = no ball / negative sample)
        is_gt:  True if from ground-truth labels, False if YOLO pseudo-label
    """
    def __init__(self, samples, frames_dir, augment=False):
        self.samples     = samples
        self.frames_dir  = frames_dir
        self.augment     = augment
        self.frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith('.jpg'))

    def __len__(self):
        return len(self.samples)

    def _load(self, idx):
        idx = max(0, min(idx, len(self.frame_files) - 1))
        return load_frame(os.path.join(self.frames_dir, self.frame_files[idx]))

    def __getitem__(self, idx):
        s  = self.samples[idx]
        fi = s['frame_indices']

        # Stack 3 frames → [H, W, 9]
        imgs    = [self._load(i) for i in fi]
        stacked = np.concatenate(imgs, axis=2)   # [H, W, 9]

        # Scale ball pos from ORIG → INPUT
        px = s['px'] * INPUT_W / ORIG_W if s['px'] >= 0 else -1.0
        py = s['py'] * INPUT_H / ORIG_H if s['py'] >= 0 else -1.0

        # Augmentation
        if self.augment:
            # Horizontal flip
            if random.random() < 0.5:
                stacked = stacked[:, ::-1, :]
                if px >= 0:
                    px = INPUT_W - 1.0 - px

            # Brightness / contrast jitter
            alpha = random.uniform(0.75, 1.25)   # contrast
            beta  = random.uniform(-0.10, 0.10)  # brightness
            stacked = np.clip(stacked * alpha + beta, 0.0, 1.0)

            # Gaussian noise
            if random.random() < 0.3:
                noise   = np.random.randn(*stacked.shape).astype(np.float32) * 0.02
                stacked = np.clip(stacked + noise, 0.0, 1.0)

        tensor = torch.from_numpy(np.ascontiguousarray(stacked.transpose(2, 0, 1)))  # [9,H,W]

        hm        = make_heatmap(px, py)
        hm_tensor = torch.from_numpy(hm).unsqueeze(0)   # [1, H, W]

        return tensor, hm_tensor


# ── Training ─────────────────────────────────────────────────────────────────

def build_samples(labels_path, yolo_cache_path, frames_dir,
                  fps=FPS, pseudo_conf=0.35):
    """
    Return (samples, gt_frame_set) where samples is the full training list.

    GT labels are remapped from their original 10fps timestamps to 15fps frame indices.
    YOLO pseudo-labels add high-confidence detections from frames without GT.
    """
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith('.jpg'))
    n = len(frame_files)

    # Ground-truth labels: remap timestamp → 15fps frame index
    with open(labels_path) as f:
        labels_data = json.load(f)
    gt = {}
    for lb in labels_data['labels']:
        f15 = int(round(lb['t'] * fps))
        if 0 < f15 < n - 1:
            gt[f15] = (lb['px'], lb['py'])
    print(f'  GT labels: {len(gt)} (remapped to {fps}fps frame indices)')

    # YOLO pseudo-labels: high-confidence ball detections not in GT
    pseudo = {}
    if yolo_cache_path and os.path.exists(yolo_cache_path):
        with open(yolo_cache_path) as f:
            cache = {int(k): v for k, v in json.load(f).items()}
        for fi, det in cache.items():
            if fi in gt or fi == 0 or fi >= n - 1:
                continue
            best_ball = None
            best_conf = pseudo_conf
            for bx, by, bc in det.get('balls', []):
                if bc > best_conf:
                    best_conf = bc
                    best_ball = (bx, by)
            if best_ball:
                pseudo[fi] = best_ball
        print(f'  YOLO pseudo-labels (conf>{pseudo_conf}): {len(pseudo)}')
    else:
        print(f'  No YOLO cache found — training on GT only')

    # Build sample list
    samples = []
    for fi, (px, py) in gt.items():
        samples.append({'frame_indices': [fi - 1, fi, fi + 1],
                        'px': px, 'py': py, 'is_gt': True})
    for fi, (px, py) in pseudo.items():
        samples.append({'frame_indices': [fi - 1, fi, fi + 1],
                        'px': px, 'py': py, 'is_gt': False})

    return samples, set(gt.keys())


def focal_bce_loss(logits, targets, pos_weight=200.0, gamma=2.0):
    """
    Focal BCE: down-weights easy background pixels, up-weights ball region.

    pos_weight compensates for the extreme class imbalance (ball ~0.03% of pixels).
    gamma reduces loss contribution from easy negatives (standard focal loss).
    """
    pw  = torch.tensor([pos_weight], dtype=torch.float32, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw,
                                              reduction='none')
    pt    = torch.sigmoid(logits)
    p_t   = targets * pt + (1 - targets) * (1 - pt)
    focal = ((1.0 - p_t) ** gamma) * bce
    return focal.mean()


def train(args):
    device = get_device()
    print(f'Device: {device}')

    print('Building training samples...')
    samples, gt_frames = build_samples(args.labels, args.yolo_cache,
                                       args.frames_dir, pseudo_conf=args.pseudo_conf)
    gt_s     = [s for s in samples if s['is_gt']]
    pseudo_s = [s for s in samples if not s['is_gt']]
    print(f'  Total: {len(samples)} samples')

    # Val: 10% of GT, stratified by time (take every 10th)
    gt_s.sort(key=lambda s: s['frame_indices'][1])
    val_s   = gt_s[::10]
    train_gt = [s for s in gt_s if s not in val_s]
    train_s  = train_gt + pseudo_s
    random.shuffle(train_s)
    print(f'  Train: {len(train_s)}  Val: {len(val_s)}')

    train_ds = BallDataset(train_s, args.frames_dir, augment=True)
    val_ds   = BallDataset(val_s,   args.frames_dir, augment=False)

    pin = device.type not in ('mps',)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=2, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=2, pin_memory=pin)

    model = TrackNet().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'TrackNet parameters: {n_params:,}')

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                        eta_min=args.lr * 0.01)

    best_val = float('inf')
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, hms in train_dl:
            imgs = imgs.to(device)
            hms  = hms.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss   = focal_bce_loss(logits, hms)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, hms in val_dl:
                imgs = imgs.to(device)
                hms  = hms.to(device)
                val_loss += focal_bce_loss(model(imgs), hms).item()
        val_loss /= max(len(val_dl), 1)
        sched.step()

        elapsed = time.time() - t0
        marker  = ''
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.model)
            marker = f'  ← saved ({args.model})'

        if epoch % 5 == 0 or epoch == 1:
            print(f'Epoch {epoch:3d}/{args.epochs}  '
                  f'train={train_loss:.4f}  val={val_loss:.4f}  '
                  f'lr={sched.get_last_lr()[0]:.1e}  '
                  f't={elapsed:.0f}s{marker}')

    print(f'\nDone. Best val loss: {best_val:.4f}')
    print(f'Model saved to {args.model}')


# ── Inference ─────────────────────────────────────────────────────────────────

def find_peaks(heatmap, conf_thresh=0.10, nms_radius=20, max_peaks=3):
    """
    Find up to max_peaks local maxima in the heatmap above conf_thresh.
    Uses iterative argmax + suppression.
    Returns list of (px_input, py_input, conf).
    """
    hm     = heatmap.copy()
    peaks  = []
    for _ in range(max_peaks):
        val = float(hm.max())
        if val < conf_thresh:
            break
        idx = np.argmax(hm)
        row, col = np.unravel_index(idx, hm.shape)
        peaks.append((float(col), float(row), val))
        r0 = max(0, row - nms_radius);  r1 = min(hm.shape[0], row + nms_radius + 1)
        c0 = max(0, col - nms_radius);  c1 = min(hm.shape[1], col + nms_radius + 1)
        hm[r0:r1, c0:c1] = 0.0
    return peaks


def infer(args):
    device = get_device()
    print(f'Device: {device}')

    if not os.path.exists(args.model):
        print(f'ERROR: model file not found: {args.model}')
        print('Run "python tracknet.py train" first.')
        return

    model = TrackNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f'Loaded {args.model}')

    frame_files = sorted(f for f in os.listdir(args.frames_dir) if f.endswith('.jpg'))
    n = len(frame_files)
    print(f'Running inference on {n} frames...')

    scale_x = ORIG_W / INPUT_W
    scale_y = ORIG_H / INPUT_H

    cache = {}
    t0    = time.time()

    with torch.no_grad():
        for i in range(n):
            i_prev = max(0, i - 1)
            i_next = min(n - 1, i + 1)

            imgs = [load_frame(os.path.join(args.frames_dir, frame_files[j]))
                    for j in (i_prev, i, i_next)]
            stacked = np.concatenate(imgs, axis=2).transpose(2, 0, 1)  # [9, H, W]
            tensor  = torch.from_numpy(np.ascontiguousarray(stacked)).unsqueeze(0).to(device)

            heatmap = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

            peaks = find_peaks(heatmap, conf_thresh=args.conf)
            # Scale peaks back to original resolution
            balls = [[px * scale_x, py * scale_y, conf] for px, py, conf in peaks]
            cache[i] = {'balls': balls}

            if (i + 1) % 250 == 0:
                elapsed = time.time() - t0
                eta     = elapsed / (i + 1) * (n - i - 1)
                print(f'  {i+1}/{n}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s')

    with open(args.output, 'w') as f:
        json.dump({str(k): v for k, v in cache.items()}, f)

    total_dets = sum(len(v['balls']) for v in cache.values())
    print(f'Saved {args.output}  ({len(cache)} frames, {total_dets} ball detections)')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap  = argparse.ArgumentParser(description='TrackNet soccer ball detector')
    sub = ap.add_subparsers(dest='cmd')

    # train
    t = sub.add_parser('train', help='Train TrackNet')
    t.add_argument('--frames_dir',  default='frames_15fps')
    t.add_argument('--labels',      default='ball_labels.json')
    t.add_argument('--yolo_cache',  default='yolo_cache_test1_15fps.json')
    t.add_argument('--model',       default='tracknet_model.pth')
    t.add_argument('--epochs',      type=int,   default=80)
    t.add_argument('--batch',       type=int,   default=4)
    t.add_argument('--lr',          type=float, default=1e-4)
    t.add_argument('--pseudo_conf', type=float, default=0.35,
                   help='Min YOLO conf for pseudo-labels (default 0.35)')

    # infer
    i = sub.add_parser('infer', help='Run inference, save cache')
    i.add_argument('--frames_dir', default='frames_15fps')
    i.add_argument('--model',      default='tracknet_model.pth')
    i.add_argument('--output',     default='tracknet_cache_15fps.json')
    i.add_argument('--conf',       type=float, default=0.10,
                   help='Min heatmap peak value to report (default 0.10)')

    args = ap.parse_args()
    if args.cmd == 'train':
        train(args)
    elif args.cmd == 'infer':
        infer(args)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
