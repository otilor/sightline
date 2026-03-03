"""VOD downloader — fetches VODs from YouTube playlists via yt-dlp.

Filters by keyword in title, downloads at 720p, extracts opponent from
CDL broadcast title patterns.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PlaylistEntry:
    """Metadata for a single video in a playlist."""
    video_id: str
    title: str
    url: str
    duration_sec: float | None = None
    opponent: str | None = None


def _extract_opponent(title: str, team: str = "faze") -> str | None:
    """Try to extract the opponent team name from a CDL broadcast title.

    Common patterns:
    - "FaZe vs OpTic | Major 2 Week 1"
    - "Atlanta FaZe vs OpTic Texas | CDL 2025"
    - "OpTic vs FaZe Clan | Stage 3"
    """
    team_lower = team.lower()
    title_lower = title.lower()

    # Pattern: "TeamA vs TeamB"
    vs_pattern = re.compile(
        r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\s*[\|–—\-]|\s*$)",
        re.IGNORECASE,
    )
    match = vs_pattern.search(title)
    if match:
        left, right = match.group(1).strip(), match.group(2).strip()
        if team_lower in left.lower():
            return right
        elif team_lower in right.lower():
            return left

    return None


def list_playlist(
    playlist_url: str,
    keyword_filter: str | None = None,
    limit: int | None = None,
) -> list[PlaylistEntry]:
    """Fetch playlist entries without downloading.

    Parameters
    ----------
    playlist_url : str
        YouTube playlist URL.
    keyword_filter : str, optional
        Only include videos whose title contains this keyword (case-insensitive).
    limit : int, optional
        Maximum number of entries to return.

    Returns
    -------
    list[PlaylistEntry]
        Matching playlist entries.
    """
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--print", "%(id)s\t%(title)s\t%(duration)s\t%(url)s",
        "--no-warnings",
        playlist_url,
    ]

    logger.info("Fetching playlist metadata: %s", playlist_url)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
    except FileNotFoundError:
        raise RuntimeError("yt-dlp is not installed. Install with: pip install yt-dlp")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"yt-dlp failed: {e.stderr}")

    entries: list[PlaylistEntry] = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue

        video_id = parts[0]
        title = parts[1]
        duration = float(parts[2]) if len(parts) > 2 and parts[2] != "NA" else None
        url = parts[3] if len(parts) > 3 else f"https://www.youtube.com/watch?v={video_id}"

        # Apply keyword filter
        if keyword_filter and keyword_filter.lower() not in title.lower():
            continue

        opponent = _extract_opponent(title)

        entries.append(PlaylistEntry(
            video_id=video_id,
            title=title,
            url=url,
            duration_sec=duration,
            opponent=opponent,
        ))

    if limit is not None:
        entries = entries[:limit]

    logger.info("Found %d matching videos%s", len(entries),
                f" (filtered by '{keyword_filter}')" if keyword_filter else "")
    return entries


def download_vods(
    entries: list[PlaylistEntry],
    output_dir: str | Path,
    resolution: int = 720,
    output_format: str = "mp4",
) -> list[Path]:
    """Download VODs from a list of playlist entries.

    Parameters
    ----------
    entries : list[PlaylistEntry]
        Videos to download.
    output_dir : str | Path
        Directory to save downloaded files.
    resolution : int
        Maximum video height (default 720p).
    output_format : str
        Output container format.

    Returns
    -------
    list[Path]
        Paths to downloaded files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []

    for i, entry in enumerate(entries, 1):
        # Sanitize filename
        safe_title = re.sub(r'[^\w\s\-]', '', entry.title).strip()
        safe_title = re.sub(r'\s+', '_', safe_title)
        output_path = output_dir / f"{safe_title}.{output_format}"

        if output_path.exists():
            logger.info("[%d/%d] Already exists, skipping: %s", i, len(entries), output_path.name)
            downloaded.append(output_path)
            continue

        logger.info("[%d/%d] Downloading: %s", i, len(entries), entry.title)

        cmd = [
            "yt-dlp",
            "--format", f"bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]",
            "--merge-output-format", output_format,
            "--output", str(output_path),
            "--no-warnings",
            "--no-playlist",
            entry.url,
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=True)
            downloaded.append(output_path)
            logger.info("[%d/%d] Downloaded: %s", i, len(entries), output_path.name)
        except subprocess.CalledProcessError as e:
            logger.error("[%d/%d] Failed to download %s: %s", i, len(entries), entry.title, e.stderr)

    logger.info("Download complete: %d/%d successful", len(downloaded), len(entries))
    return downloaded
