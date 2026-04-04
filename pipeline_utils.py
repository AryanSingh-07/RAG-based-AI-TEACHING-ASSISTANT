from __future__ import annotations

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = ROOT_DIR / "videos"
DATAFRAME_PATH = ROOT_DIR / "dataframe.joblib"
PROCESSED_LIST_PATH = ROOT_DIR / "processed_videos.joblib"
PIPELINE_STATE_PATH = ROOT_DIR / "pipeline_state.json"
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi"}


def canonical_video_name(filename: str) -> str:
    return Path(filename).stem.split(" [")[0]


def file_fingerprint(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "name": path.name,
        "canonical_name": canonical_video_name(path.name),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def get_current_video_fingerprints(videos_dir: str | Path = VIDEOS_DIR) -> dict[str, dict[str, int | str]]:
    videos_dir = Path(videos_dir)
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    fingerprints: dict[str, dict[str, int | str]] = {}
    for file_path in sorted(
        path
        for path in videos_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    ):
        rel = str(file_path.relative_to(videos_dir))
        fingerprints[rel] = file_fingerprint(file_path)
    return fingerprints


def load_pipeline_state(path: str | Path = PIPELINE_STATE_PATH) -> dict[str, object]:
    path = Path(path)
    if not path.exists():
        return {"videos": {}}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_pipeline_state(state: dict[str, object], path: str | Path = PIPELINE_STATE_PATH) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)


def get_pipeline_plan(force: bool = False) -> dict[str, object]:
    current_videos = get_current_video_fingerprints()
    saved_state = load_pipeline_state()
    processed_videos = saved_state.get("videos", {})

    if force:
        mode = "full"
    elif not DATAFRAME_PATH.exists() or not PROCESSED_LIST_PATH.exists() or not processed_videos:
        mode = "full"
    else:
        removed_videos = sorted(set(processed_videos) - set(current_videos))
        changed_videos = sorted(
            name
            for name, fingerprint in current_videos.items()
            if name in processed_videos and processed_videos.get(name) != fingerprint
        )
        new_videos = sorted(name for name in current_videos if name not in processed_videos)

        if removed_videos:
            mode = "full"
        elif any(name in processed_videos for name in changed_videos):
            mode = "full"
        elif new_videos:
            mode = "incremental"
        else:
            mode = "noop"

        return {
            "mode": mode,
            "current_videos": sorted(current_videos),
            "processed_videos": sorted(processed_videos),
            "new_videos": new_videos,
            "changed_videos": changed_videos,
            "removed_videos": removed_videos,
            "current_fingerprints": current_videos,
        }

    return {
        "mode": mode,
        "current_videos": sorted(current_videos),
        "processed_videos": sorted(processed_videos),
        "new_videos": sorted(current_videos),
        "changed_videos": [],
        "removed_videos": [],
        "current_fingerprints": current_videos,
    }
