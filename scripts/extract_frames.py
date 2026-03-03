"""
Extract frames from downloaded VODs for labeling.

Outputs 3 datasets:
  1. data/frames/full/         — full frames every 30s (for mode classification)
  2. data/frames/minimap/      — minimap crops every 5s during gameplay (for YOLO dot detection)
  3. data/frames/killfeed/     — killfeed crops from mid-left (playerA [weapon] playerB notifications)
  4. data/frames/player_stats/ — player stats panel from top-right (K/D, streaks, time)

Usage:
    python3 scripts/extract_frames.py                     # process all VODs
    python3 scripts/extract_frames.py --vod data/vods/specific.mp4
    python3 scripts/extract_frames.py --max-per-vod 200   # limit frames per VOD
"""
import argparse
import cv2
import pathlib
import sys

# ── Region definitions (for 1280x720 VODs) ─────────────────────────────
# These are approximate; adjust after inspecting a few frames.
# Minimap is typically bottom-left corner, ~200x200px at 720p
MINIMAP_ROI = (0, 500, 300, 720)       # (x1, y1, x2, y2)
# Killfeed is mid-left — kill notifications: playerA [weapon_icon] playerB
KILLFEED_ROI = (0, 200, 350, 500)
# Player stats panel is top-right — K/D, streaks, time
PLAYER_STATS_ROI = (880, 0, 1280, 190)
# Scoreboard is top-center
SCOREBOARD_ROI = (450, 0, 810, 200)


def is_gameplay_frame(frame):
    """Quick heuristic: gameplay frames have a visible minimap (bright region in bottom-left)."""
    minimap = frame[MINIMAP_ROI[1]:MINIMAP_ROI[3], MINIMAP_ROI[0]:MINIMAP_ROI[2]]
    gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
    # Gameplay minimaps have clear structure; menus/replays don't
    mean_val = gray.mean()
    std_val = gray.std()
    return 30 < mean_val < 200 and std_val > 20


def extract_from_vod(vod_path: pathlib.Path, out_dir: pathlib.Path, max_frames: int, interval_full: int, interval_detail: int):
    cap = cv2.VideoCapture(str(vod_path))
    if not cap.isOpened():
        print(f"  ✗ Could not open {vod_path.name}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 60
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    stem = vod_path.stem[:80]  # truncate long names

    # Create output dirs
    full_dir = out_dir / "full"
    mini_dir = out_dir / "minimap"
    kill_dir = out_dir / "killfeed"
    stats_dir = out_dir / "player_stats"
    score_dir = out_dir / "scoreboard"
    for d in [full_dir, mini_dir, kill_dir, stats_dir, score_dir]:
        d.mkdir(parents=True, exist_ok=True)

    count = 0
    gameplay_count = 0
    frame_full = int(fps * interval_full)
    frame_detail = int(fps * interval_detail)

    print(f"  Processing: {vod_path.name}")
    print(f"  Duration: {duration/60:.1f} min | FPS: {fps:.0f} | Frames: {total:,}")

    frame_idx = 0
    while frame_idx < total and count < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        ts = frame_idx / fps

        # Full frame every interval_full seconds (for mode classification)
        if frame_idx % frame_full == 0:
            fname = f"{stem}_{ts:07.1f}s.png"
            cv2.imwrite(str(full_dir / fname), frame)
            count += 1

        # Minimap + killfeed every interval_detail seconds (only during gameplay)
        if frame_idx % frame_detail == 0 and is_gameplay_frame(frame):
            gameplay_count += 1
            ts_str = f"{stem}_{ts:07.1f}s"

            # Minimap crop
            minimap = frame[MINIMAP_ROI[1]:MINIMAP_ROI[3], MINIMAP_ROI[0]:MINIMAP_ROI[2]]
            cv2.imwrite(str(mini_dir / f"{ts_str}.png"), minimap)

            # Killfeed crop (mid-left — kill notifications)
            killfeed = frame[KILLFEED_ROI[1]:KILLFEED_ROI[3], KILLFEED_ROI[0]:KILLFEED_ROI[2]]
            cv2.imwrite(str(kill_dir / f"{ts_str}.png"), killfeed)

            # Player stats crop (top-right)
            player_stats = frame[PLAYER_STATS_ROI[1]:PLAYER_STATS_ROI[3], PLAYER_STATS_ROI[0]:PLAYER_STATS_ROI[2]]
            cv2.imwrite(str(stats_dir / f"{ts_str}.png"), player_stats)

            # Scoreboard crop
            scoreboard = frame[SCOREBOARD_ROI[1]:SCOREBOARD_ROI[3], SCOREBOARD_ROI[0]:SCOREBOARD_ROI[2]]
            cv2.imwrite(str(score_dir / f"{ts_str}.png"), scoreboard)

            count += 1

        frame_idx += frame_detail  # Jump forward

        # Progress
        if count % 50 == 0 and count > 0:
            print(f"    {count} frames extracted ({gameplay_count} gameplay)...")

    cap.release()
    print(f"  ✓ Done: {count} total ({gameplay_count} gameplay frames)")
    return count


def main():
    parser = argparse.ArgumentParser(description="Extract frames from CDL VODs for labeling")
    parser.add_argument("--vod", type=str, help="Process a single VOD")
    parser.add_argument("--vod-dir", type=str, default="data/vods", help="Directory containing VODs")
    parser.add_argument("--out-dir", type=str, default="data/frames", help="Output directory")
    parser.add_argument("--max-per-vod", type=int, default=500, help="Max frames per VOD")
    parser.add_argument("--full-interval", type=int, default=30, help="Seconds between full frame captures")
    parser.add_argument("--detail-interval", type=int, default=5, help="Seconds between minimap/killfeed captures")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)

    if args.vod:
        vods = [pathlib.Path(args.vod)]
    else:
        vods = sorted(pathlib.Path(args.vod_dir).glob("*.mp4"))

    if not vods:
        print("No VODs found!")
        sys.exit(1)

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║  Sightline Frame Extractor                  ║")
    print(f"║  VODs: {len(vods):<5}  Output: {str(out_dir):<20}║")
    print(f"╚══════════════════════════════════════════════╝")
    print()

    total = 0
    for vod in vods:
        n = extract_from_vod(vod, out_dir, args.max_per_vod, args.full_interval, args.detail_interval)
        total += n
        print()

    print(f"═══ Total: {total} frames extracted from {len(vods)} VODs ═══")
    print()
    print("Next steps:")
    print(f"  1. Minimap labels:  Upload {out_dir}/minimap/ to Roboflow for YOLO annotation")
    print(f"  2. Mode labels:     Sort {out_dir}/full/ into folders: hp/ snd/ control/ menu/ replay/")
    print(f"  3. Weapon icons:    Crop weapon sprites from {out_dir}/killfeed/ → data/maps/weapons/")
    print(f"  4. Player stats:    {out_dir}/player_stats/ — for OCR training data")


if __name__ == "__main__":
    main()
