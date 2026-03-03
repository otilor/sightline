"""
Auto-label full frames by game mode using scoreboard OCR.

Reads scoreboard crops, detects mode keywords, and copies matching full frames
into class folders for training a mode classifier.

Usage:
    python3 scripts/auto_label_modes.py
"""
import cv2
import pathlib
import shutil
import pytesseract
import re
import sys

# Mode keywords that appear on scoreboard/HUD
MODE_KEYWORDS = {
    "hp": ["hardpoint", "hard point", "hp"],
    "snd": ["search", "destroy", "search & destroy", "s&d", "snd", "search and destroy"],
    "control": ["control", "ctrl"],
}

# Non-gameplay keywords (menus, replays, intermissions)
NON_GAMEPLAY = ["best of", "map", "veto", "ban", "pick", "listen in", "interview",
                "replay", "round", "halftime", "half time", "final"]


def classify_scoreboard(scoreboard_path: pathlib.Path) -> str:
    """Read scoreboard crop and determine game mode via OCR."""
    img = cv2.imread(str(scoreboard_path))
    if img is None:
        return "unknown"

    # Convert to grayscale and threshold for better OCR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Run OCR
    text = pytesseract.image_to_string(thresh, config="--psm 6").lower().strip()

    # Check for mode keywords
    for mode, keywords in MODE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return mode

    # Check for non-gameplay
    for kw in NON_GAMEPLAY:
        if kw in text:
            return "menu"

    return "unknown"


def extract_and_label(vod_dir: pathlib.Path, frames_dir: pathlib.Path, out_dir: pathlib.Path, interval: int = 10):
    """Extract full frames from VODs and auto-label using corresponding scoreboard crops."""

    scoreboard_dir = frames_dir / "scoreboard"
    if not scoreboard_dir.exists():
        print("No scoreboard frames found. Run extract_frames.py first.")
        sys.exit(1)

    # Create output class folders
    classes = ["hp", "snd", "control", "menu", "unknown"]
    for cls in classes:
        (out_dir / cls).mkdir(parents=True, exist_ok=True)

    # Group scoreboard crops by VOD
    scoreboards = sorted(scoreboard_dir.glob("*.png"))
    print(f"Found {len(scoreboards)} scoreboard crops")

    # For each scoreboard crop, classify it and extract corresponding full frame
    stats = {cls: 0 for cls in classes}
    vods_cache = {}

    for i, sb_path in enumerate(scoreboards):
        mode = classify_scoreboard(sb_path)
        stats[mode] += 1

        # Parse timestamp from filename: VOD_NAME_TIMESTAMP.png
        fname = sb_path.stem
        # Find the VOD file and timestamp
        parts = fname.rsplit("_", 1)
        if len(parts) != 2:
            continue
        vod_stem = parts[0]
        ts_str = parts[1]  # e.g. "00120.0s"

        # Extract full frame from VOD at this timestamp
        ts = float(ts_str.replace("s", ""))

        # Find matching VOD
        vod_path = None
        for ext in [".mp4", ".webm", ".mkv"]:
            candidate = vod_dir / f"{vod_stem}{ext}"
            if candidate.exists():
                vod_path = candidate
                break

        if vod_path is None:
            continue

        # Open VOD (cached)
        if vod_stem not in vods_cache:
            cap = cv2.VideoCapture(str(vod_path))
            if not cap.isOpened():
                continue
            vods_cache[vod_stem] = cap

        cap = vods_cache[vod_stem]
        fps = cap.get(cv2.CAP_PROP_FPS) or 60
        frame_idx = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Save to class folder
        out_path = out_dir / mode / f"{vod_stem}_{ts_str}.png"
        cv2.imwrite(str(out_path), frame)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(scoreboards)}...")

    # Close all video captures
    for cap in vods_cache.values():
        cap.release()

    print(f"\n{'='*50}")
    print(f"Auto-labeling complete!")
    print(f"{'='*50}")
    for cls, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {cls:>10}: {count:>5} frames")
    print(f"  {'TOTAL':>10}: {sum(stats.values()):>5}")
    print(f"\nOutput: {out_dir}/")
    print(f"Review the 'unknown' folder and manually sort into correct classes.")


if __name__ == "__main__":
    project = pathlib.Path("/Users/mac/.gemini/antigravity/playground/sidereal-eclipse")
    extract_and_label(
        vod_dir=project / "data" / "vods",
        frames_dir=project / "data" / "frames",
        out_dir=project / "data" / "frames" / "labeled",
        interval=10,
    )
