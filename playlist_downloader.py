from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from yt_dlp import YoutubeDL

ROOT_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = ROOT_DIR / "videos"


def _slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s_-]+", "-", text.strip())
    text = text.strip("-")
    return text or "playlist"


def fetch_playlist(playlist_url: str) -> tuple[str, list[dict[str, str]]]:
    """
    Return playlist title and list of flat entries with id/title.
    No downloads happen here.
    """
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,
        "dump_single_json": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)

    title = info.get("title") or "playlist"
    entries = info.get("entries") or []

    videos = []
    for entry in entries:
        vid = entry.get("id")
        etitle = entry.get("title") or "video"
        if not vid:
            continue
        videos.append(
            {
                "id": vid,
                "title": etitle,
                "url": f"https://www.youtube.com/watch?v={vid}",
            }
        )
    return title, videos


def download_selected(
    playlist_title: str,
    videos: Iterable[dict[str, str]],
    selected_ids: set[str],
    base_dir: Path | str = VIDEOS_DIR,
) -> list[Path]:
    """
    Download selected videos into videos/<playlist-slug>/.
    Returns list of downloaded file paths.
    """
    base_dir = Path(base_dir)
    playlist_slug = _slugify(playlist_title)
    target_dir = base_dir / playlist_slug
    target_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "quiet": True,
        "noprogress": True,
        "outtmpl": str(target_dir / "%(title)s [%(id)s].%(ext)s"),
        "ignoreerrors": True,
        "retries": 2,
        "fragment_retries": 2,
        "merge_output_format": "mp4",
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }
        ],
    }

    downloaded: list[Path] = []
    with YoutubeDL(ydl_opts) as ydl:
        for video in videos:
            if video["id"] not in selected_ids:
                continue
            url = video["url"]
            result = ydl.download([url])
            # yt_dlp returns numeric; we rely on file template for path construction
            # collect existing files matching id
            for ext in (".mp4", ".mkv", ".webm", ".mov", ".avi"):
                candidate = target_dir / f"{video['title']} [{video['id']}]{ext}"
                if candidate.exists():
                    downloaded.append(candidate)
                    break

    return downloaded
