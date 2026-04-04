from __future__ import annotations

import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
VIDEO_DIR = ROOT_DIR / "videos"
AUDIO_DIR = ROOT_DIR / "audios"
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi"}


def video_output_stem(video_path: Path, base_dir: Path) -> str:
    relative = video_path.relative_to(base_dir)
    parts = relative.with_suffix("").parts
    cleaned = []
    for part in parts:
        cleaned.append("".join(ch for ch in part if ch.isalnum() or ch in {"-", "_"}))
    return "__".join(cleaned) or "audio"


def to_audio(
    video_dir: str | Path = VIDEO_DIR,
    output_dir: str | Path = AUDIO_DIR,
    video_files: list[str] | None = None,
) -> None:
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)

    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    output_dir.mkdir(exist_ok=True)

    requested_files = set(video_files or [])
    if requested_files:
        video_paths = sorted((video_dir / rel) for rel in requested_files)
    else:
        video_paths = sorted(
            path
            for path in video_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        )
    if not video_paths:
        if requested_files:
            print("No new videos needed audio extraction.")
            return
        raise RuntimeError(f"No supported video files found in {video_dir}")

    for video_path in video_paths:
        print(f"Processing video {video_path.name}")
        output_name = video_output_stem(video_path, video_dir)
        output_path = output_dir / f"{output_name}.mp3"

        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "+discardcorrupt",
            "-err_detect",
            "ignore_err",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "libmp3lame",
            str(output_path),
        ]

        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            error_message = result.stderr.strip() or "Unknown ffmpeg error"
            raise RuntimeError(f"Failed to extract audio from {video_path.name}: {error_message}")
