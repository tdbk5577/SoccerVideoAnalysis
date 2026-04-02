#!/usr/bin/env python3
"""
Soccer Play-by-Play POC
------------------------
Extracts frames from a soccer video, runs YOLOv8 to detect players and the ball,
classifies players by jersey color, builds a structured scene description, and
sends it to Claude for decision-point coaching feedback.

Requirements:
  - FFmpeg installed (brew install ffmpeg)
  - pip install anthropic ultralytics opencv-python numpy

Usage:
  python soccer_poc.py --video path/to/your/clip.mp4
  python soccer_poc.py --video clip.mp4 --no-cv   # skip CV, send raw image only
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

FRAME_INTERVAL_SECONDS = 1

# Set your API key via: export ANTHROPIC_API_KEY="your_key_here"
API_KEY = None

# HSV color ranges for jersey classification.
# Tweak these if your team colors aren't being detected correctly.
# You can test with: python3 -c "import cv2; img=cv2.imread('frame.jpg'); ..."
TEAM_COLOR_HSV = {
    "blue":   [([100, 50,  50],  [130, 255, 255])],
    "red":    [([0,   50,  50],  [10,  255, 255]),
               ([160, 50,  50],  [180, 255, 255])],  # red wraps in HSV
    "green":  [([40,  50,  50],  [80,  255, 255])],
    "yellow": [([20,  100, 100], [35,  255, 255])],
    "white":  [([0,   0,   200], [180, 30,  255])],
    "black":  [([0,   0,   0],   [180, 255, 50])],
    "orange": [([5,   100, 100], [20,  255, 255])],
    "purple": [([130, 50,  50],  [160, 255, 255])],
}

# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(video_path: str, interval: int, output_dir: str) -> list[str]:
    """Use FFmpeg to pull one frame every `interval` seconds from the video."""
    print(f"\n📽  Extracting frames every {interval}s from: {video_path}")

    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps=1/{interval}",
        "-q:v", "2",
        "-loglevel", "error",
        output_pattern
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ FFmpeg error:\n{result.stderr}")
        raise RuntimeError("FFmpeg failed. Is it installed? Run: brew install ffmpeg")

    frames = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".jpg")
    ])
    print(f"✅ Extracted {len(frames)} frames")
    return frames


# ── Computer vision ───────────────────────────────────────────────────────────

def load_yolo():
    """Load YOLOv8 nano — downloads weights automatically on first run."""
    print("🔍 Loading YOLOv8 model...")
    return YOLO("yolov8n.pt")


def count_color_pixels(hsv_region: np.ndarray, color: str) -> int:
    """Count pixels in an HSV region matching a named color."""
    total = 0
    for (lo, hi) in TEAM_COLOR_HSV[color]:
        mask = cv2.inRange(hsv_region, np.array(lo), np.array(hi))
        total += int(cv2.countNonZero(mask))
    return total


def classify_jersey_color(frame: np.ndarray, bbox: list) -> str:
    """Return the dominant jersey color for a player bounding box."""
    x1, y1, x2, y2 = map(int, bbox)
    # Use only the upper ~40% of the box (torso/jersey, not legs or grass)
    jersey_bottom = y1 + max(1, (y2 - y1) * 2 // 5)
    region = frame[y1:jersey_bottom, x1:x2]
    if region.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    scores = {color: count_color_pixels(hsv, color) for color in TEAM_COLOR_HSV}
    best_color = max(scores, key=scores.get)
    return best_color if scores[best_color] > 20 else "unknown"


def euclidean(a: dict, b: dict) -> float:
    return ((a["cx"] - b["cx"]) ** 2 + (a["cy"] - b["cy"]) ** 2) ** 0.5


def field_zone(cx: float, cy: float) -> str:
    """Describe rough field position from normalized coordinates."""
    col = "left channel" if cx < 0.33 else ("right channel" if cx > 0.66 else "center")
    row = "attacking third" if cy < 0.33 else ("defensive third" if cy > 0.66 else "midfield")
    return f"{col}, {row}"


def detect_scene(frame_path: str, model, team: str, opponent: str) -> dict | None:
    """Run YOLO on a frame, classify players, and return structured scene data."""
    frame = cv2.imread(frame_path)
    if frame is None:
        return None

    h, w = frame.shape[:2]
    detections = model(frame, verbose=False)[0]

    players = []
    ball = None

    for box in detections.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        cx   = (bbox[0] + bbox[2]) / 2 / w
        cy   = (bbox[1] + bbox[3]) / 2 / h

        if cls == 0 and conf > 0.4:   # person
            color = classify_jersey_color(frame, bbox)
            players.append({"color": color, "cx": cx, "cy": cy})

        elif cls == 32 and conf > 0.25:  # sports ball
            ball = {"cx": cx, "cy": cy}

    return {"players": players, "ball": ball}


def build_scene_description(scene: dict, team: str, opponent: str) -> str:
    """Convert raw detection data into a structured text description for Claude."""
    if not scene:
        return "Scene detection unavailable."

    players  = scene["players"]
    ball     = scene["ball"]
    team_p   = [p for p in players if p["color"] == team]
    opp_p    = [p for p in players if p["color"] == opponent]
    lines    = []

    if not ball:
        lines.append("Ball not detected in this frame.")
    else:
        lines.append(f"Ball location: {field_zone(ball['cx'], ball['cy'])}.")

        # Who has the ball?
        all_known = [p for p in players if p["color"] in (team, opponent)]
        if all_known:
            carrier = min(all_known, key=lambda p: euclidean(p, ball))
            carrier_dist = euclidean(carrier, ball)

            if carrier_dist < 0.08:
                lines.append(f"Ball carrier: {carrier['color']} player at {field_zone(carrier['cx'], carrier['cy'])}.")

                # Defensive pressure on the carrier
                defenders = opp_p if carrier["color"] == team else team_p
                nearby = sorted(defenders, key=lambda p: euclidean(p, carrier))[:3]
                if nearby:
                    pressure_desc = []
                    for d in nearby:
                        dist_val = euclidean(d, carrier)
                        label = "tight" if dist_val < 0.07 else ("medium" if dist_val < 0.15 else "loose")
                        pressure_desc.append(f"{label} ({field_zone(d['cx'], d['cy'])})")
                    lines.append(f"Pressure on carrier: {', '.join(pressure_desc)}.")

                # Open teammates of the target team
                if carrier["color"] == team:
                    teammates = [p for p in team_p if p is not carrier]
                    open_tm = []
                    for tm in teammates:
                        nearest_opp_dist = min((euclidean(tm, op) for op in opp_p), default=1.0)
                        if nearest_opp_dist > 0.12:
                            open_tm.append(field_zone(tm["cx"], tm["cy"]))
                    if open_tm:
                        lines.append(f"Open {team} teammates: {', '.join(open_tm[:3])}.")
                    else:
                        lines.append(f"No clearly open {team} teammates detected.")
            else:
                lines.append("No player in clear possession of the ball.")

    lines.append(f"{team.capitalize()} players visible: {len(team_p)} | Opponent players visible: {len(opp_p)}.")
    return "\n".join(lines)


# ── Claude analysis ───────────────────────────────────────────────────────────

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def prompt_config() -> dict:
    """Interactively prompt the user for analysis settings."""
    print("\n⚙️  Analysis Configuration\n" + "-" * 40)

    team = input("Which team to analyze? (e.g. blue / red): ").strip().lower() or "blue"
    opponent = input(f"What color is the opposing team? (e.g. red / white): ").strip().lower() or "red"
    phases = input("Which phases? (attacking / defending / both) [both]: ").strip().lower() or "both"
    interval_input = input(f"Seconds between frames [default: {FRAME_INTERVAL_SECONDS}]: ").strip()
    interval = int(interval_input) if interval_input.isdigit() else FRAME_INTERVAL_SECONDS

    print("-" * 40 + "\n")
    return {"team": team, "opponent": opponent, "phases": phases, "interval": interval}


def build_system_prompt(team: str, phases: str) -> str:
    if phases == "attacking":
        phase_scope = "Focus only on attacking decisions (passes, dribbles, shots, runs in behind, crossing, combination play)."
    elif phases == "defending":
        phase_scope = "Focus only on defensive decisions (positioning, pressing, tracking runners, intercepting, tackling, covering)."
    else:
        phase_scope = "Evaluate decisions in both attacking and defensive phases."

    return f"""You are an expert soccer coach analyzing match footage frame by frame.

Only evaluate {team} team players. Ignore all other players.
{phase_scope}

You will receive:
1. A frame from the match
2. A structured scene description from computer vision (player positions, ball location, pressure, open teammates)

Use the scene description as ground truth for spatial facts. Use the image to identify what decision was actually made.

If a {team} player made a suboptimal decision, respond in exactly this format:
{team.capitalize()} #[number or "player"] [what they did]. Should have [what they should have done]. [One sentence reason why.]

If no improvable decision is visible, respond only with: "No decision point."
No other text."""


def analyze_frames(frames: list[str], api_key: str, config: dict, model) -> list[dict]:
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("No API key. Run: export ANTHROPIC_API_KEY='your_key_here'")

    client        = anthropic.Anthropic(api_key=key)
    system_prompt = build_system_prompt(config["team"], config["phases"])
    results       = []
    total         = len(frames)

    print(f"\n🤖 Analyzing {total} frames...\n")

    for i, frame_path in enumerate(frames):
        timestamp = i * config["interval"]
        time_label = f"{timestamp // 60:02d}:{timestamp % 60:02d}"
        print(f"  Frame {i+1}/{total} → {time_label}...")

        # CV scene detection
        if model is not None:
            scene = detect_scene(frame_path, model, config["team"], config["opponent"])
            scene_desc = build_scene_description(scene, config["team"], config["opponent"])
        else:
            scene_desc = "Computer vision disabled."

        image_data = encode_image(frame_path)

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
                            f"Evaluate any {config['team']} team decision visible in this frame."
                        )
                    }
                ],
            }],
        )

        results.append({
            "timestamp": time_label,
            "analysis": message.content[0].text,
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        })

    return results


# ── Output ────────────────────────────────────────────────────────────────────

def print_report(results: list[dict]):
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
    total_cost  = input_cost + output_cost

    summary = (
        f"\n{'='*60}\n"
        f"  USAGE SUMMARY\n"
        f"{'='*60}\n"
        f"  Frames analyzed  : {len(results)}\n"
        f"  Decision points  : {sum(1 for r in results if r['analysis'].strip().lower() != 'no decision point.')}\n"
        f"  Input tokens     : {total_input:,}\n"
        f"  Output tokens    : {total_output:,}\n"
        f"  Estimated cost   : ${total_cost:.4f}\n"
        f"{'='*60}\n"
    )
    print(summary)
    lines.append(summary)

    output_file = "play_by_play_output.txt"
    with open(output_file, "w") as f:
        f.write("\n".join(lines))
    print(f"📄 Report saved to: {output_file}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Soccer coaching POC using YOLOv8 + Claude")
    parser.add_argument("--video",  required=True, help="Path to video file")
    parser.add_argument("--no-cv",  action="store_true", help="Skip computer vision, send raw image only")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return

    config = prompt_config()
    model  = None if args.no_cv else load_yolo()

    with tempfile.TemporaryDirectory() as tmpdir:
        frames = extract_frames(args.video, config["interval"], tmpdir)
        if not frames:
            print("❌ No frames extracted.")
            return
        results = analyze_frames(frames, API_KEY, config, model)
        print_report(results)


if __name__ == "__main__":
    main()
