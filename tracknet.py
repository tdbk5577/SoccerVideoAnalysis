#!/usr/bin/env python3
"""
TrackNet-style soccer ball detector.

A 9-channel U-Net: takes 3 consecutive RGB frames stacked as input,
outputs a Gaussian heatmap of ball probability.

Commands:
  .venv/bin/python3 tracknet.py train [options]
  .venv/bin/python3 tracknet.py infer [options]

Train:
  --frames_dir  frames_15fps
  --labels      ball_labels.json
  --model       tracknet_model.pth
  --epochs      80  --batch 4  --lr 1e-4
  --resume              (continue from existing checkpoint)

Infer:
  --frames_dir  frames_15fps
  --model       tracknet_model.pth
  --output      tracknet_cache_15fps.json
  --conf        0.10
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

# ── Resolution ────────────────────────────────────────────────────────────────
INPUT_W = 640    # 1920 / 3
INPUT_H = 360    # 1080 / 3
ORIG_W  = 1920
ORIG_H  = 1080
SIGMA   = 5      # Gaussian sigma at INPUT resolution
FPS     = 15.0


# ── Model ─────────────────────────────────────────────────────────────────────

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
    Input:  [B, 9, H, W]  — 3 consecutive RGB frames concatenated
    Output: [B, 1, H, W]  — raw logits (sigmoid for probabilities)
    """
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.enc1 = ConvBlock(9,   64)
        self.enc2 = ConvBlock(64,  128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 256, dropout=0.3)   # bottleneck

        self.up3  = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)

        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def make_heatmap(px, py, w=INPUT_W, h=INPUT_H, sigma=SIGMA):
    if px < 0 or py < 0:
        return np.zeros((h, w), dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    return np.exp(-((X - px)**2 + (Y - py)**2) / (2*sigma**2)).astype(np.float32)


def load_frame(path):
    img = cv2.imread(path)
    if img is None:
        return np.zeros((INPUT_H, INPUT_W, 3), dtype=np.float32)
    img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


# ── Dataset ───────────────────────────────────────────────────────────────────

class BallDataset(Dataset):
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
        s       = self.samples[idx]
        stacked = np.concatenate([self._load(i) for i in s['frame_indices']], axis=2)

        px = s['px'] * INPUT_W / ORIG_W if s['px'] >= 0 else -1.0
        py = s['py'] * INPUT_H / ORIG_H if s['py'] >= 0 else -1.0

        if self.augment:
            if random.random() < 0.5:            # horizontal flip
                stacked = stacked[:, ::-1, :]
                if px >= 0:
                    px = INPUT_W - 1.0 - px
            alpha = random.uniform(0.75, 1.25)   # brightness/contrast
            beta  = random.uniform(-0.10, 0.10)
            stacked = np.clip(stacked * alpha + beta, 0.0, 1.0)
            if random.random() < 0.3:             # Gaussian noise
                stacked = np.clip(stacked + np.random.randn(*stacked.shape).astype(np.float32)*0.02,
                                  0.0, 1.0)

        tensor    = torch.from_numpy(np.ascontiguousarray(stacked.transpose(2, 0, 1)))
        hm_tensor = torch.from_numpy(make_heatmap(px, py)).unsqueeze(0)
        return tensor, hm_tensor


# ── Training ──────────────────────────────────────────────────────────────────

def build_samples(labels_path, frames_dir, fps=FPS):
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith('.jpg'))
    n = len(frame_files)

    with open(labels_path) as f:
        labels_data = json.load(f)

    samples = []
    for lb in labels_data['labels']:
        f15 = int(round(lb['t'] * fps))
        if 0 < f15 < n - 1:
            samples.append({
                'frame_indices': [f15 - 1, f15, f15 + 1],
                'px': lb['px'], 'py': lb['py'],
            })

    print(f'  {len(samples)} training samples from {len(labels_data["labels"])} GT labels')
    return samples


def focal_bce_loss(logits, targets, pos_weight=200.0, gamma=2.0):
    pw  = torch.tensor([pos_weight], dtype=torch.float32, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction='none')
    p_t = targets * torch.sigmoid(logits) + (1 - targets) * (1 - torch.sigmoid(logits))
    return (((1.0 - p_t) ** gamma) * bce).mean()


def train(args):
    device = get_device()
    print(f'Device: {device}')

    samples = build_samples(args.labels, args.frames_dir)

    # Val: every 10th sample (stratified by time)
    samples.sort(key=lambda s: s['frame_indices'][1])
    val_s   = samples[::10]
    train_s = [s for s in samples if s not in val_s]
    random.shuffle(train_s)
    print(f'  Train: {len(train_s)}  Val: {len(val_s)}')

    # num_workers=0 avoids macOS multiprocessing issues
    train_dl = DataLoader(BallDataset(train_s, args.frames_dir, augment=True),
                          batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl   = DataLoader(BallDataset(val_s,   args.frames_dir, augment=False),
                          batch_size=args.batch, shuffle=False, num_workers=0)

    model = TrackNet().to(device)
    print(f'TrackNet parameters: {sum(p.numel() for p in model.parameters()):,}')

    start_epoch = 1
    if args.resume and os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f'Resumed from {args.model}')

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                        eta_min=args.lr * 0.01)

    best_val = float('inf')
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, hms in train_dl:
            imgs = imgs.to(device); hms = hms.to(device)
            opt.zero_grad()
            loss = focal_bce_loss(model(imgs), hms)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, hms in val_dl:
                val_loss += focal_bce_loss(model(imgs.to(device)), hms.to(device)).item()
        val_loss /= max(len(val_dl), 1)
        sched.step()

        marker = ''
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.model)
            marker = f'  ← saved'

        if epoch % 5 == 0 or epoch == 1:
            print(f'Epoch {epoch:3d}/{args.epochs}  '
                  f'train={train_loss:.4f}  val={val_loss:.4f}  '
                  f'lr={sched.get_last_lr()[0]:.1e}  '
                  f't={time.time()-t0:.0f}s{marker}')

    print(f'\nDone. Best val={best_val:.4f}  Model: {args.model}')


# ── Inference ─────────────────────────────────────────────────────────────────

def find_peaks(heatmap, conf_thresh=0.10, nms_radius=20, max_peaks=3):
    hm, peaks = heatmap.copy(), []
    for _ in range(max_peaks):
        val = float(hm.max())
        if val < conf_thresh:
            break
        row, col = np.unravel_index(np.argmax(hm), hm.shape)
        peaks.append((float(col), float(row), val))
        r0, r1 = max(0, row-nms_radius), min(hm.shape[0], row+nms_radius+1)
        c0, c1 = max(0, col-nms_radius), min(hm.shape[1], col+nms_radius+1)
        hm[r0:r1, c0:c1] = 0.0
    return peaks


def infer(args):
    device = get_device()
    if not os.path.exists(args.model):
        print(f'ERROR: {args.model} not found. Run "tracknet.py train" first.')
        return

    model = TrackNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f'Loaded {args.model}  |  Device: {device}')

    frame_files = sorted(f for f in os.listdir(args.frames_dir) if f.endswith('.jpg'))
    n = len(frame_files)
    scale_x, scale_y = ORIG_W / INPUT_W, ORIG_H / INPUT_H

    cache, t0 = {}, time.time()
    with torch.no_grad():
        for i in range(n):
            imgs    = [load_frame(os.path.join(args.frames_dir, frame_files[j]))
                       for j in (max(0,i-1), i, min(n-1,i+1))]
            stacked = np.concatenate(imgs, axis=2).transpose(2, 0, 1)
            tensor  = torch.from_numpy(np.ascontiguousarray(stacked)).unsqueeze(0).to(device)
            heatmap = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

            peaks = find_peaks(heatmap, conf_thresh=args.conf)
            cache[i] = {'balls': [[px*scale_x, py*scale_y, c] for px,py,c in peaks]}

            if (i+1) % 250 == 0:
                elapsed = time.time() - t0
                print(f'  {i+1}/{n}  elapsed={elapsed:.0f}s  ETA={elapsed/(i+1)*(n-i-1):.0f}s')

    with open(args.output, 'w') as f:
        json.dump({str(k): v for k, v in cache.items()}, f)
    total = sum(len(v['balls']) for v in cache.values())
    print(f'Saved {args.output}  ({n} frames, {total} detections)')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap  = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd')

    t = sub.add_parser('train')
    t.add_argument('--frames_dir', default='frames_15fps')
    t.add_argument('--labels',     default='ball_labels.json')
    t.add_argument('--model',      default='tracknet_model.pth')
    t.add_argument('--epochs',     type=int,   default=80)
    t.add_argument('--batch',      type=int,   default=4)
    t.add_argument('--lr',         type=float, default=1e-4)
    t.add_argument('--resume',     action='store_true',
                   help='Continue from existing checkpoint')

    i = sub.add_parser('infer')
    i.add_argument('--frames_dir', default='frames_15fps')
    i.add_argument('--model',      default='tracknet_model.pth')
    i.add_argument('--output',     default='tracknet_cache_15fps.json')
    i.add_argument('--conf',       type=float, default=0.10)

    args = ap.parse_args()
    if   args.cmd == 'train': train(args)
    elif args.cmd == 'infer': infer(args)
    else: ap.print_help()


if __name__ == '__main__':
    main()
