#!/usr/bin/env python3
"""
Soccer Coaching POC
--------------------
Two-pass pipeline:
  Pass 1 — CV only: YOLOv8 scans every frame and flags coaching situations
            geometrically (cross, press, support angle, final third).
            No API calls.
  Pass 2 — Claude only: flagged frames are sent to Claude with a
            situation-specific prompt. Everything else is skipped.

Requirements:
  - FFmpeg installed (brew install ffmpeg)
  - pip install anthropic ultralytics opencv-python numpy

Usage:
  python soccer_poc.py --video path/to/your/clip.mp4
  python soccer_poc.py --video clip.mp4 --no-cv   # raw image mode, no CV filtering
"""

import anthropic
import argparse
import base64
import os
import subprocess
import tempfile

import cv2
import numpy as np
from ultralytics import YOLO

# ── Configuration ─────────────────────────────────────────────────────────────

FRAMES_PER_SECOND  = 2   # rate for YOLO + Claude analysis
TRACKING_FPS       = 10  # rate for ball tracking only (higher = more accurate LK)

API_KEY = None  # set via: export ANTHROPIC_API_KEY="your_key_here"

# HSV jersey color ranges — tweak if detection is off for your video
TEAM_COLOR_HSV = {
    # Broader ranges to handle color variation from overhead cameras
    "blue":   [([90,  40,  40],  [135, 255, 255])],           # covers navy, royal, sky
    "red":    [([0,   40,  40],  [15,  255, 255]),             # covers red and orange-red
               ([155, 40,  40],  [180, 255, 255])],
    "orange": [([8,   80,  80],  [25,  255, 255])],            # orange overlaps with red range intentionally
    "green":  [([35,  40,  40],  [85,  255, 255])],
    "yellow": [([18,  80,  80],  [38,  255, 255])],
    "white":  [([0,   0,   180], [180, 40,  255])],
    "black":  [([0,   0,   0],   [180, 255, 60])],
    "purple": [([125, 40,  40],  [160, 255, 255])],
}

# Colors that should be treated as the same team.
# From an elevated side camera, blue jerseys often appear teal/green — so green is aliased to blue.
# Orange is aliased to red for similar reasons.
COLOR_ALIASES = {
    "red":    ["red", "orange"],
    "orange": ["orange", "red"],
    "blue":   ["blue", "green"],   # elevated camera makes blue look teal/green
    "green":  ["green", "blue"],
    "yellow": ["yellow"],
    "white":  ["white"],
    "black":  ["black"],
    "purple": ["purple"],
}

# What each flagged situation tells Claude to look for
SITUATION_PROMPTS = {
    "possession": (
        "Blue has clear possession of the ball. Evaluate the decision made — "
        "whether a pass, dribble, hold, or cross. Was it the best option available? "
        "Look at where teammates are, the pressure on the ball, and available space."
    ),
    "cross": (
        "This is a crossing/wide delivery situation. Evaluate: "
        "(1) Was the delivery into the dangerous zone between the 6-yard box and penalty spot, or was it at the goalkeeper? "
        "(2) Are attacking players crashing the box at full speed, or jogging/standing?"
    ),
    "support": (
        "This is a support angle situation. Evaluate: "
        "Are the nearby blue teammates positioned at angles that make them easy to pass to, "
        "or are they too close, in line with, or behind the ball carrier?"
    ),
    "final_third": (
        "This is a final third possession situation. Evaluate: "
        "Did the blue player in possession make the right decision — shoot, pass, or dribble — "
        "given the defensive pressure and the options available around them?"
    ),
    "press": (
        "This is a pressing situation. Evaluate: "
        "Is the blue presser's angle cutting off the most dangerous forward pass? "
        "Are they aware of runners behind them that they are not tracking?"
    ),
    "cover": (
        "This is a defensive cover/positioning situation. Evaluate: "
        "Are blue defenders correctly tracking opposition runners, covering dangerous space, "
        "and maintaining a compact defensive shape?"
    ),
}

# Which phase each situation belongs to
SITUATION_PHASE = {
    "possession":  "attacking",
    "cross":       "attacking",
    "support":     "attacking",
    "final_third": "attacking",
    "press":       "defending",
    "cover":       "defending",
}

# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(video_path: str, output_dir: str, fps: int, prefix: str = "frame") -> list[str]:
    output_pattern = os.path.join(output_dir, f"{prefix}_%04d.jpg")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2", "-loglevel", "error",
        output_pattern
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr}")
    frames = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir) if f.startswith(prefix) and f.endswith(".jpg")
    ])
    return frames


# ── Computer vision — detection ───────────────────────────────────────────────

def load_yolo() -> YOLO:
    print("🔍 Loading YOLOv8 model...")
    return YOLO("yolov8m.pt")


def count_color_pixels(hsv: np.ndarray, color: str) -> int:
    total = 0
    for lo, hi in TEAM_COLOR_HSV[color]:
        total += int(cv2.countNonZero(cv2.inRange(hsv, np.array(lo), np.array(hi))))
    return total


def classify_jersey(frame: np.ndarray, bbox: list) -> str:
    x1, y1, x2, y2 = map(int, bbox)
    jersey_h = max(1, (y2 - y1) * 2 // 5)
    region = frame[y1:y1 + jersey_h, x1:x2]
    if region.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    scores = {c: count_color_pixels(hsv, c) for c in TEAM_COLOR_HSV}
    best = max(scores, key=scores.get)
    return best if scores[best] > 20 else "unknown"


def dist(a: dict, b: dict) -> float:
    return ((a["cx"] - b["cx"]) ** 2 + (a["cy"] - b["cy"]) ** 2) ** 0.5


def detect_scene(frame_path: str, model: YOLO, team: str, opponent: str) -> dict:
    frame = cv2.imread(frame_path)
    if frame is None:
        return {"players": [], "ball": None}

    h, w = frame.shape[:2]
    players, ball = [], None

    for box in model(frame, verbose=False)[0].boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        cx   = (bbox[0] + bbox[2]) / 2 / w
        cy   = (bbox[1] + bbox[3]) / 2 / h

        if cls == 0 and conf > 0.25:
            color = classify_jersey(frame, bbox)
            players.append({"color": color, "cx": cx, "cy": cy})
        elif cls == 32 and conf > 0.1:
            ball = {"cx": cx, "cy": cy}

    return {"players": players, "ball": ball}


# ── Computer vision — situation flagging ──────────────────────────────────────

def _detect_ball_candidates(frame_gray: np.ndarray, prev_gray: np.ndarray,
                             w: int, h: int, players: list) -> list:
    """
    Find small moving circular blobs near players — ball candidates.
    Players list is used to reject false positives far from the action.
    """
    diff = cv2.absdiff(frame_gray, prev_gray)
    _, motion = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN,  kernel)
    motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 5 < area < 600:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"] / w
                cy = M["m01"] / M["m00"] / h
                perim = cv2.arcLength(cnt, True)
                circ  = (4 * np.pi * area / perim ** 2) if perim > 0 else 0
                if circ > 0.3 and players:
                    nearest_player = min(
                        ((p["cx"] - cx) ** 2 + (p["cy"] - cy) ** 2) ** 0.5
                        for p in players
                    )
                    if nearest_player < 0.15:   # must be near a player
                        candidates.append({"cx": cx, "cy": cy, "area": area, "circ": circ})
    return candidates


def track_ball_across_frames(frames: list[str], player_positions: list[list],
                              w: int, h: int) -> list[dict | None]:
    """
    Two-phase ball tracking:
      Phase A — scan all consecutive frame pairs to find the best seed detection
      Phase B — use Lucas-Kanade optical flow to track forward and backward from seed

    Returns a list of ball positions (cx, cy) per frame, or None if not tracked.
    LK uses a large search window (51x51) and pyramid levels to handle fast ball movement.
    """
    n = len(frames)
    positions: list[dict | None] = [None] * n

    # ── Phase A: find seed frame with most reliable ball detection ────────────
    grays = []
    for fp in frames:
        img = cv2.imread(fp)
        grays.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img is not None else None)

    # Collect all candidates across all frames first
    all_candidates = []
    for i in range(1, n):
        if grays[i] is None or grays[i - 1] is None:
            continue
        cands = _detect_ball_candidates(grays[i], grays[i - 1], w, h, player_positions[i])
        for c in cands:
            all_candidates.append((i, c))

    # Pick seed by movement: real ball moves across frames, field markings don't
    best_seed_idx  = None
    best_seed_ball = None
    best_score     = -1.0

    for idx, cand in all_candidates:
        nearby = [(j, c) for j, c in all_candidates if 0 < abs(j - idx) <= 5]
        if not nearby:
            continue
        displacements = [
            ((c["cx"] - cand["cx"]) ** 2 + (c["cy"] - cand["cy"]) ** 2) ** 0.5
            for _, c in nearby
        ]
        avg_movement = sum(displacements) / len(displacements)
        score = avg_movement * cand["circ"]
        if score > best_score:
            best_score     = score
            best_seed_idx  = idx
            best_seed_ball = cand

    if best_seed_ball is None:
        print("⚠️  Could not find ball in any frame — possession detection will use player-position heuristic.")
        return positions

    positions[best_seed_idx] = best_seed_ball
    print(f"  Ball seed found at frame {best_seed_idx} "
          f"(cx={best_seed_ball['cx']:.2f}, cy={best_seed_ball['cy']:.2f})")

    lk_params = dict(
        winSize=(51, 51),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    def _track_direction(start_idx: int, start_ball: dict, step: int):
        """Track forward (step=+1) or backward (step=-1) from start_idx."""
        pt  = np.array([[start_ball["cx"] * w, start_ball["cy"] * h]],
                       dtype=np.float32).reshape(-1, 1, 2)
        prev_g = grays[start_idx]

        for i in range(start_idx + step, (n if step == 1 else -1), step):
            if grays[i] is None:
                break
            curr_g = grays[i]
            next_pt, status, _ = cv2.calcOpticalFlowPyrLK(prev_g, curr_g, pt, None, **lk_params)

            tracked = False
            if status is not None and status[0][0] == 1:
                nx, ny   = next_pt[0][0]
                new_ball = {"cx": float(nx / w), "cy": float(ny / h)}

                # Validate: ball must stay near players and within frame bounds
                in_bounds = 0.0 < new_ball["cx"] < 1.0 and 0.0 < new_ball["cy"] < 1.0
                near_player = False
                if player_positions[i] and in_bounds:
                    nearest = min(
                        ((p["cx"] - new_ball["cx"]) ** 2 + (p["cy"] - new_ball["cy"]) ** 2) ** 0.5
                        for p in player_positions[i]
                    )
                    near_player = nearest < 0.20

                if in_bounds and near_player:
                    positions[i] = new_ball
                    pt     = next_pt
                    prev_g = curr_g
                    tracked = True

            if not tracked:
                # Try to re-detect from motion before giving up
                prev_i = i - step
                if 0 <= prev_i < n and grays[prev_i] is not None:
                    cands = _detect_ball_candidates(
                        curr_g, grays[prev_i], w, h, player_positions[i]
                    )
                    if cands:
                        redetect = max(cands, key=lambda c: c["circ"])
                        positions[i] = redetect
                        pt = np.array(
                            [[redetect["cx"] * w, redetect["cy"] * h]],
                            dtype=np.float32
                        ).reshape(-1, 1, 2)
                        prev_g = curr_g
                        continue
                break   # tracking lost, stop

    _track_direction(best_seed_idx, best_seed_ball, step=+1)
    _track_direction(best_seed_idx, best_seed_ball, step=-1)

    tracked_count = sum(1 for p in positions if p is not None)
    print(f"  Ball tracked in {tracked_count}/{n} frames")
    return positions


def possession_heuristic(scene: dict, team: str, opponent: str) -> str:
    """
    Fallback when ball isn't detected: the team with more players in the
    opponent's half is likely attacking and therefore in possession.
    """
    team_colors = COLOR_ALIASES.get(team, [team])
    opp_colors  = COLOR_ALIASES.get(opponent, [opponent])
    team_p = [p for p in scene["players"] if p["color"] in team_colors]
    opp_p  = [p for p in scene["players"] if p["color"] in opp_colors]
    if not team_p or not opp_p:
        return "unknown"

    team_centroid_x = sum(p["cx"] for p in team_p) / len(team_p)
    opp_centroid_x  = sum(p["cx"] for p in opp_p)  / len(opp_p)

    # Whichever team's centroid is further from their own goal is likely attacking
    # We don't know which goal is which, so we look at spread: attacking team pushes more players forward
    team_advanced = sum(1 for p in team_p if p["cx"] > 0.5) / len(team_p)
    opp_advanced  = sum(1 for p in opp_p  if p["cx"] > 0.5) / len(opp_p)

    # Also try left side
    team_advanced_l = sum(1 for p in team_p if p["cx"] < 0.5) / len(team_p)
    opp_advanced_l  = sum(1 for p in opp_p  if p["cx"] < 0.5) / len(opp_p)

    team_push = max(team_advanced, team_advanced_l)
    opp_push  = max(opp_advanced,  opp_advanced_l)

    if team_push > opp_push + 0.15:
        return "team"
    if opp_push > team_push + 0.15:
        return "opponent"
    return "unknown"


def determine_possession(ball_positions: list, scenes: list[dict], frame_idx: int,
                         team: str, opponent: str) -> str:
    """
    Primary: check which team is consistently nearest the ball across a window of frames.
    Fallback: use player position heuristic if ball not reliably detected.
    """
    team_colors = COLOR_ALIASES.get(team, [team])
    opp_colors  = COLOR_ALIASES.get(opponent, [opponent])

    team_score = 0
    opp_score  = 0
    ball_found = 0
    window     = range(max(0, frame_idx - 3), min(len(ball_positions), frame_idx + 4))

    for i in window:
        if i >= len(ball_positions) or i >= len(scenes):
            continue
        ball = ball_positions[i]
        if ball is None:
            continue
        ball_found += 1
        players = scenes[i]["players"]
        team_p  = [p for p in players if p["color"] in team_colors]
        opp_p   = [p for p in players if p["color"] in opp_colors]
        d_team  = min((dist(p, ball) for p in team_p), default=1.0)
        d_opp   = min((dist(p, ball) for p in opp_p),  default=1.0)
        if d_team < 0.15 and d_team < d_opp:
            team_score += 1
        elif d_opp < 0.15 and d_opp < d_team:
            opp_score += 1

    # If ball was detected in enough frames, use proximity result
    if ball_found >= 2:
        if team_score > opp_score:
            return "team"
        if opp_score > team_score:
            return "opponent"

    # Fallback: player position heuristic
    if frame_idx < len(scenes):
        return possession_heuristic(scenes[frame_idx], team, opponent)

    return "unknown"


def get_carrier(scene: dict, team: str, opponent: str):
    """Return (carrier_dict, in_possession) using YOLO ball if available."""
    ball = scene["ball"]
    if not ball:
        return None, False
    team_colors = COLOR_ALIASES.get(team, [team])
    opp_colors  = COLOR_ALIASES.get(opponent, [opponent])
    known = [p for p in scene["players"] if p["color"] in team_colors + opp_colors]
    if not known:
        return None, False
    carrier = min(known, key=lambda p: dist(p, ball))
    return carrier, dist(carrier, ball) < 0.1


def flag_situations(scene: dict, team: str, opponent: str, phases: str,
                    possession: str = "unknown") -> list[str]:
    """
    Geometric checks — no API calls.
    Returns list of situation keys filtered by the requested phase(s).

    Coordinate system (normalized 0-1):
      cx: left→right across the frame
      cy: top→bottom across the frame
    Touchlines are near cy=0 and cy=1.
    Goals are near cx=0 and cx=1.
    """
    ball        = scene["ball"]
    players     = scene["players"]
    team_colors = COLOR_ALIASES.get(team, [team])
    opp_colors  = COLOR_ALIASES.get(opponent, [opponent])
    team_p      = [p for p in players if p["color"] in team_colors]
    opp_p       = [p for p in players if p["color"] in opp_colors]
    carrier, in_possession = get_carrier(scene, team, opponent)
    situations = []

    # Determine possession from multi-frame ball tracking (primary)
    # or fall back to YOLO ball detection (secondary)
    team_colors = COLOR_ALIASES.get(team, [team])
    opp_colors  = COLOR_ALIASES.get(opponent, [opponent])
    team_has_ball = (
        possession == "team" or
        (in_possession and carrier is not None and carrier["color"] in team_colors)
    )
    opp_has_ball = (
        possession == "opponent" or
        (in_possession and carrier is not None and carrier["color"] in opp_colors)
    )

    # ── OFFENSIVE SITUATIONS — only when blue has possession ──────────────────

    # 1. POSSESSION — blue clearly has the ball
    if team_has_ball:
        situations.append("possession")

    # 2. CROSS — blue attacking near a goal end (blue players in final third + box)
    # Only flag when blue has possession
    if team_has_ball:
        for goal_side in ["left", "right"]:
            if goal_side == "left":
                final_third = [p for p in team_p if p["cx"] < 0.30]
                in_box      = [p for p in team_p if p["cx"] < 0.20]
            else:
                final_third = [p for p in team_p if p["cx"] > 0.70]
                in_box      = [p for p in team_p if p["cx"] > 0.80]
            if len(final_third) >= 2 and len(in_box) >= 1:
                situations.append("cross")
                break

    # 3. SUPPORT ANGLE — blue has ball, 2+ teammates nearby
    if team_has_ball and carrier is not None:
        close_teammates = [p for p in team_p if p is not carrier and dist(p, carrier) < 0.25]
        if len(close_teammates) >= 2:
            situations.append("support")

    # 4. FINAL THIRD WITH PRESSURE — blue has ball near goal, defenders close
    if team_has_ball and carrier is not None:
        near_goal = carrier["cx"] < 0.33 or carrier["cx"] > 0.67
        if near_goal:
            nearby_defenders = [p for p in opp_p if dist(p, carrier) < 0.25]
            if nearby_defenders:
                situations.append("final_third")

    # ── DEFENSIVE SITUATIONS — only when opponent has possession ─────────────

    # 5. PRESS — opponent has ball, blue presser nearby, runner behind presser
    if opp_has_ball and carrier is not None:
        pressers = [p for p in team_p if dist(p, carrier) < 0.15]
        if pressers:
            other_opps = [p for p in opp_p if p is not carrier]
            runners_behind = [p for p in other_opps if dist(p, pressers[0]) < 0.25]
            if runners_behind:
                situations.append("press")

    # 5. COVER — opponent in possession, blue defenders back, checking shape
    if in_possession and carrier["color"] == opponent:
        deep_defenders = [p for p in team_p if dist(p, carrier) > 0.2]
        if len(deep_defenders) >= 2:
            situations.append("cover")

    # Filter by phase
    if phases == "attacking":
        situations = [s for s in situations if SITUATION_PHASE[s] == "attacking"]
    elif phases == "defending":
        situations = [s for s in situations if SITUATION_PHASE[s] == "defending"]

    return situations


def field_zone(cx: float, cy: float) -> str:
    # cx: left goal (0) → right goal (1); cy: top touchline (0) → bottom touchline (1)
    end  = "left goal end" if cx < 0.33 else ("right goal end" if cx > 0.66 else "midfield")
    side = "near top touchline" if cy < 0.2 else ("near bottom touchline" if cy > 0.8 else "central")
    return f"{end}, {side}"


def build_scene_description(scene: dict, team: str, opponent: str) -> str:
    ball    = scene["ball"]
    team_p  = [p for p in scene["players"] if p["color"] in COLOR_ALIASES.get(team, [team])]
    opp_p   = [p for p in scene["players"] if p["color"] in COLOR_ALIASES.get(opponent, [opponent])]
    carrier, in_possession = get_carrier(scene, team, opponent)
    lines   = []

    if ball:
        lines.append(f"Ball: {field_zone(ball['cx'], ball['cy'])}.")
    else:
        lines.append("Ball not detected.")

    if in_possession:
        lines.append(f"Carrier: {carrier['color']} at {field_zone(carrier['cx'], carrier['cy'])}.")
        defenders = opp_p if carrier["color"] == team else team_p
        nearby_def = sorted(defenders, key=lambda p: dist(p, carrier))[:3]
        if nearby_def:
            pressure = []
            for d in nearby_def:
                d_val = dist(d, carrier)
                label = "tight" if d_val < 0.07 else ("medium" if d_val < 0.15 else "loose")
                pressure.append(f"{label} at {field_zone(d['cx'], d['cy'])}")
            lines.append(f"Defensive pressure: {', '.join(pressure)}.")
        if carrier["color"] == team:
            open_tm = [
                field_zone(p["cx"], p["cy"]) for p in team_p
                if p is not carrier and min((dist(p, op) for op in opp_p), default=1.0) > 0.12
            ]
            lines.append(f"Open {team} teammates: {', '.join(open_tm[:3]) or 'none detected'}.")
    else:
        lines.append("No player in clear possession.")

    lines.append(f"{team.capitalize()} players: {len(team_p)} | Opponents: {len(opp_p)}.")
    return "\n".join(lines)


# ── Claude analysis ───────────────────────────────────────────────────────────

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def prompt_config() -> dict:
    print("\n⚙️  Analysis Configuration\n" + "-" * 40)
    team     = input("Which team to analyze? (e.g. blue / red): ").strip().lower() or "blue"
    opponent = input("What color is the opposing team?: ").strip().lower() or "red"
    phases   = input("Which phases? (attacking / defending / both) [both]: ").strip().lower() or "both"
    print("-" * 40 + "\n")
    return {"team": team, "opponent": opponent, "phases": phases}


def build_system_prompt(team: str, phases: str) -> str:
    if phases == "attacking":
        phase_scope = "Focus only on attacking decisions."
    elif phases == "defending":
        phase_scope = "Focus only on defensive decisions."
    else:
        phase_scope = "Evaluate both attacking and defensive decisions."

    return f"""You are an expert soccer coach reviewing flagged match footage.

Only evaluate {team} team players. {phase_scope}

You will receive a frame, a scene description from computer vision, and a description of the specific situation that was detected. Use these to evaluate whether a {team} player made a suboptimal decision.

Only flag a play if:
- A specific {team} player made an identifiable decision
- A clearly better option was available
- You can say exactly what they should have done

If no clearly better option existed — including contested situations, scrambles, or plays where the decision was reasonable — respond only with: "No decision point."

If there is a coaching point, respond in exactly this format:
{team.capitalize()} #[number or "player"] [what they did]. Should have [what they should have done]. [One sentence why.]

No other text."""


def analyze_frames(flagged: list, api_key: str, config: dict) -> list[dict]:
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("No API key. Run: export ANTHROPIC_API_KEY='your_key_here'")

    client        = anthropic.Anthropic(api_key=key)
    system_prompt = build_system_prompt(config["team"], config["phases"])
    results       = []
    total         = len(flagged)

    print(f"🤖 Pass 2: Sending {total} candidate frames to Claude...\n")

    for idx, (frame_idx, frame_path, scene, situations) in enumerate(flagged):
        timestamp  = frame_idx / FRAMES_PER_SECOND
        time_label = f"{int(timestamp)//60:02d}:{timestamp%60:04.1f}"
        print(f"  Frame {idx+1}/{total} → {time_label} ({', '.join(situations)})...")

        scene_desc       = build_scene_description(scene, config["team"], config["opponent"])
        situation_detail = "\n".join(f"- {SITUATION_PROMPTS[s]}" for s in situations)
        image_data       = encode_image(frame_path)

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=150,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data},
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Timestamp: {time_label}\n\n"
                            f"Scene data:\n{scene_desc}\n\n"
                            f"Detected situation(s):\n{situation_detail}\n\n"
                            f"Evaluate the {config['team']} team decision in this frame."
                        )
                    }
                ],
            }],
        )

        results.append({
            "timestamp": time_label,
            "situations": situations,
            "analysis": message.content[0].text,
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        })

    return results


# ── Output ────────────────────────────────────────────────────────────────────

def print_report(results: list[dict], total_frames: int):
    print("\n" + "=" * 60)
    print("  COACHING REPORT")
    print("=" * 60 + "\n")

    total_input = total_output = 0
    lines = []

    for r in results:
        total_input  += r["input_tokens"]
        total_output += r["output_tokens"]
        if r["analysis"].strip().lower() == "no decision point.":
            continue
        line = f"[{r['timestamp']}] {r['analysis']}\n"
        print(line)
        lines.append(line)

    input_cost  = (total_input  / 1_000_000) * 3.00
    output_cost = (total_output / 1_000_000) * 15.00

    summary = (
        f"\n{'='*60}\n"
        f"  USAGE SUMMARY\n"
        f"{'='*60}\n"
        f"  Total frames     : {total_frames}\n"
        f"  Flagged by CV    : {len(results)}\n"
        f"  Decision points  : {sum(1 for r in results if r['analysis'].strip().lower() != 'no decision point.')}\n"
        f"  Input tokens     : {total_input:,}\n"
        f"  Output tokens    : {total_output:,}\n"
        f"  Estimated cost   : ${input_cost + output_cost:.4f}\n"
        f"{'='*60}\n"
    )
    print(summary)
    lines.append(summary)

    with open("play_by_play_output.txt", "w") as f:
        f.write("\n".join(lines))
    print("📄 Report saved to: play_by_play_output.txt")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Soccer coaching POC — YOLOv8 + Claude")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--no-cv", action="store_true", help="Skip CV filtering, send all frames to Claude")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return

    config = prompt_config()
    model  = load_yolo()

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n📽  Extracting {FRAMES_PER_SECOND} fps from: {args.video}")
        frames = extract_frames(args.video, tmpdir, FRAMES_PER_SECOND, prefix="analysis")
        print(f"✅ Extracted {len(frames)} analysis frames")
        if not frames:
            print("❌ No frames extracted.")
            return

        if args.no_cv:
            # Skip CV — treat every frame as a candidate
            flagged = [(i, fp, {}, ["cross", "press", "support", "final_third"]) for i, fp in enumerate(frames)]
        else:
            # Pass 1: CV scan — no Claude calls
            print(f"\n🔍 Pass 1: Scanning {len(frames)} frames for situations...")

            # Step 1a: extract dense frames for ball tracking (higher fps = smaller inter-frame motion)
            print(f"\n📽  Extracting {FRAMES_PER_SECOND} fps analysis frames + {TRACKING_FPS} fps tracking frames...")
            track_frames = extract_frames(args.video, tmpdir, TRACKING_FPS, prefix="track")
            print(f"✅ {len(frames)} analysis frames | {len(track_frames)} tracking frames")

            ref_img = cv2.imread(frames[0]) if frames else None
            h_ref   = ref_img.shape[0] if ref_img is not None else 1080
            w_ref   = ref_img.shape[1] if ref_img is not None else 1920

            # Step 1b: detect players in every analysis frame (2fps)
            scenes = []
            for frame_path in frames:
                scenes.append(detect_scene(frame_path, model, config["team"], config["opponent"]))

            # Step 1c: track ball across dense tracking frames (10fps) — LK works at small inter-frame motion
            print("\n🏃 Tracking ball across video...")
            track_players = [
                [{"cx": p["cx"], "cy": p["cy"]} for p in detect_scene(fp, model, config["team"], config["opponent"])["players"]]
                for fp in track_frames
            ]
            ball_track = track_ball_across_frames(track_frames, track_players, w_ref, h_ref)

            # Step 1d: map 10fps ball positions back to 2fps analysis frame indices
            ratio = TRACKING_FPS // FRAMES_PER_SECOND
            ball_positions = [ball_track[min(i * ratio, len(ball_track) - 1)] for i in range(len(frames))]

            # Step 1b: determine possession per frame using multi-frame ball tracking
            flagged = []
            for i, frame_path in enumerate(frames):
                possession = determine_possession(
                    ball_positions, scenes, i, config["team"], config["opponent"]
                )
                situations = flag_situations(
                    scenes[i], config["team"], config["opponent"],
                    config["phases"], possession
                )
                if situations:
                    flagged.append((i, frame_path, scenes[i], situations))

            print(f"✅ {len(flagged)} candidate frames found ({len(frames) - len(flagged)} skipped)\n")

        if not flagged:
            print("No coaching situations detected in this clip.")
            return

        # Pass 2: Claude only on flagged frames
        results = analyze_frames(flagged, API_KEY, config)
        print_report(results, len(frames))


if __name__ == "__main__":
    main()
