from __future__ import annotations

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
JSON_DIR = ROOT_DIR / "json_data"
CLEAN_JSON_DIR = ROOT_DIR / "clean_json_data"
MIN_CHUNK_CHARACTERS = 280
MAX_CHUNK_CHARACTERS = 900
MAX_CHUNK_DURATION_SECONDS = 75.0


def _flush_chunk(chunk_buffer: list[dict[str, float | str]], output: list[dict[str, float | str]], video_name: str) -> None:
    if not chunk_buffer:
        return

    text = " ".join(str(item["text"]).strip() for item in chunk_buffer).strip()
    if not text:
        return

    output.append(
        {
            "video_name": video_name,
            "text": text,
            "start": float(chunk_buffer[0]["start"]),
            "end": float(chunk_buffer[-1]["end"]),
        }
    )


def cleaning_json(
    json_dir: str | Path = JSON_DIR,
    output_dir: str | Path = CLEAN_JSON_DIR,
    json_files: list[str] | None = None,
) -> None:
    json_dir = Path(json_dir)
    output_dir = Path(output_dir)

    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")

    output_dir.mkdir(exist_ok=True)
    requested_files = set(json_files or [])

    for file_path in sorted(path for path in json_dir.glob("*.json") if not requested_files or path.name in requested_files):
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        clean_json = []
        chunk_buffer: list[dict[str, float | str]] = []
        current_text_length = 0

        for segment in data.get("segments", []):
            text = segment.get("text", "").strip()
            if not text:
                continue

            start = float(segment["start"])
            end = float(segment["end"])
            segment_payload = {
                "text": text,
                "start": start,
                "end": end,
            }

            if not chunk_buffer:
                chunk_buffer.append(segment_payload)
                current_text_length = len(text)
                continue

            chunk_start = float(chunk_buffer[0]["start"])
            proposed_length = current_text_length + 1 + len(text)
            proposed_duration = end - chunk_start

            if (
                proposed_length <= MAX_CHUNK_CHARACTERS
                and (
                    current_text_length < MIN_CHUNK_CHARACTERS
                    or proposed_duration <= MAX_CHUNK_DURATION_SECONDS
                )
            ):
                chunk_buffer.append(segment_payload)
                current_text_length = proposed_length
                continue

            _flush_chunk(chunk_buffer, clean_json, file_path.stem)
            chunk_buffer = [segment_payload]
            current_text_length = len(text)

        _flush_chunk(chunk_buffer, clean_json, file_path.stem)

        output_path = output_dir / file_path.name
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                {"chunks": clean_json, "full_text": data.get("text", "")},
                file,
                indent=4,
                ensure_ascii=False,
            )
