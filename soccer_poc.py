#!/usr/bin/env python3
"""
Soccer Play-by-Play POC
------------------------
Extracts frames from a soccer video and sends them to Claude
for play-by-play analysis.

Requirements:
  - FFmpeg installed (brew install ffmpeg)
  - pip install anthropic --break-system-packages

Usage:
  python soccer_poc.py --video path/to/your/clip.mp4
"""

import anthropic
import argparse
import base64
import os
import subprocess
import tempfile

# ── Configuration ────────────────────────────────────────────────────────────

# How often to grab a frame (in seconds).
# 5 seconds is a good balance between detail and cost for a first test.
FRAME_INTERVAL_SECONDS = 1

# Set your API key via: export ANTHROPIC_API_KEY="your_key_here"
API_KEY = None

# ── Frame extraction ─────────────────────────────────────────────────────────

def extract_frames(video_path: str, interval: int, output_dir: str) -> list[str]:
    """Use FFmpeg to pull one frame every `interval` seconds from the video."""
    print(f"\n📽  Extracting frames every {interval} seconds from: {video_path}")

    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps=1/{interval}",   # 1 frame per N seconds
        "-q:v", "2",                   # high quality JPEG
        "-loglevel", "error",          # suppress noisy output
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


# ── Claude analysis ───────────────────────────────────────────────────────────

def encode_image(path: str) -> str:
    """Encode an image file to base64 for the Claude API."""
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def prompt_config() -> dict:
    """Interactively prompt the user for analysis settings."""
    print("\n⚙️  Analysis Configuration\n" + "-"*40)

    team = input("Which team to analyze? (e.g. blue / red / both): ").strip().lower() or "both"

    phases_input = input("Which phases? (attacking / defending / both) [both]: ").strip().lower() or "both"

    interval_input = input(f"Seconds between frames [default: {FRAME_INTERVAL_SECONDS}]: ").strip()
    interval = int(interval_input) if interval_input.isdigit() else FRAME_INTERVAL_SECONDS

    print("-"*40 + "\n")
    return {"team": team, "phases": phases_input, "interval": interval}


def build_system_prompt(team: str, phases: str) -> str:
    if team == "both":
        team_focus = "any player on either team"
        team_filter = "Do not filter by team — evaluate decisions from both sides."
    else:
        team_focus = f"players on the {team} team"
        team_filter = f"Only evaluate {team} team players. Ignore decisions made by the opposing team."

    if phases == "attacking":
        phase_scope = "Focus only on attacking decisions (passes, dribbles, shots, runs in behind, crossing, combination play)."
    elif phases == "defending":
        phase_scope = "Focus only on defensive decisions (positioning, pressing, tracking runners, intercepting, tackling, covering)."
    else:
        phase_scope = "Evaluate decisions in both attacking and defensive phases."

    return f"""You are an expert soccer coach analyzing match footage frame by frame.

{team_filter}
{phase_scope}

If you see a clear decision by {team_focus} that could have been better, respond in exactly this format:
[kit color] #[number] [what they did]. Should have [what they should have done]. [One sentence reason why.]

If no improvable decision is visible, respond only with: "No decision point."
Do not add any other text, context, or commentary."""


def analyze_frames(frames: list[str], api_key: str, config: dict) -> list[dict]:
    """Send each frame to Claude and get a soccer play description."""

    # Use API_KEY from config above, or fall back to environment variable
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key or key == "YOUR_API_KEY_HERE":
        raise ValueError(
            "No API key found. Either set API_KEY in the script, "
            "or run: export ANTHROPIC_API_KEY='your_key_here'"
        )

    client = anthropic.Anthropic(api_key=key)
    results = []
    total = len(frames)

    print(f"\n🤖 Sending {total} frames to Claude for analysis...\n")

    system_prompt = build_system_prompt(config["team"], config["phases"])

    for i, frame_path in enumerate(frames):
        timestamp = i * FRAME_INTERVAL_SECONDS
        minutes = timestamp // 60
        seconds = timestamp % 60
        time_label = f"{minutes:02d}:{seconds:02d}"

        print(f"  Analyzing frame {i+1}/{total} → {time_label}...")

        image_data = encode_image(frame_path)

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Timestamp {time_label} — describe the run of play in this frame."
                        }
                    ],
                }
            ],
        )

        analysis = message.content[0].text
        results.append({
            "timestamp": time_label,
            "frame": os.path.basename(frame_path),
            "analysis": analysis,
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        })

        print(f"  ✓ Done\n")

    return results


# ── Output ────────────────────────────────────────────────────────────────────

def print_report(results: list[dict]):
    """Print the play-by-play report to the terminal and save to a file."""

    print("\n" + "="*60)
    print("  PLAY-BY-PLAY REPORT")
    print("="*60 + "\n")

    total_input_tokens = 0
    total_output_tokens = 0
    lines = []

    for r in results:
        total_input_tokens += r["input_tokens"]
        total_output_tokens += r["output_tokens"]
        if r["analysis"].strip().lower() == "no decision point.":
            continue
        line = f"[{r['timestamp']}] {r['analysis']}\n"
        print(line)
        lines.append(line)

    # Cost estimate (Claude Sonnet pricing as of early 2026)
    input_cost  = (total_input_tokens  / 1_000_000) * 3.00
    output_cost = (total_output_tokens / 1_000_000) * 15.00
    total_cost  = input_cost + output_cost

    summary = (
        f"\n{'='*60}\n"
        f"  USAGE SUMMARY\n"
        f"{'='*60}\n"
        f"  Frames analyzed : {len(results)}\n"
        f"  Input tokens    : {total_input_tokens:,}\n"
        f"  Output tokens   : {total_output_tokens:,}\n"
        f"  Estimated cost  : ${total_cost:.4f}\n"
        f"{'='*60}\n"
    )
    print(summary)
    lines.append(summary)

    # Save to file
    output_file = "play_by_play_output.txt"
    with open(output_file, "w") as f:
        f.write("\n".join(lines))
    print(f"📄 Report saved to: {output_file}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Soccer play-by-play POC using Claude")
    parser.add_argument("--video", required=True, help="Path to your video file (MP4 recommended)")
    parser.add_argument("--interval", type=int, default=FRAME_INTERVAL_SECONDS,
                        help=f"Seconds between frames (default: {FRAME_INTERVAL_SECONDS})")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return

    config = prompt_config()

    # Use a temp directory for extracted frames (auto-cleaned up after)
    with tempfile.TemporaryDirectory() as tmpdir:
        frames = extract_frames(args.video, config["interval"], tmpdir)

        if not frames:
            print("❌ No frames were extracted. Check the video file.")
            return

        results = analyze_frames(frames, API_KEY, config)
        print_report(results)


if __name__ == "__main__":
    main()
