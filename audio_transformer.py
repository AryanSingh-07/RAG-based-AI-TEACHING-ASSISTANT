from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

ROOT_DIR = Path(__file__).resolve().parent
AUDIO_DIR = ROOT_DIR / "audios"
JSON_DIR = ROOT_DIR / "json_data"
WHISPER_MODEL_NAME = "small"


def _resolve_compute_settings() -> tuple[str, str]:
    preferred_device = os.environ.get("WHISPER_DEVICE", "cpu").strip().lower()
    if preferred_device == "cuda":
        return "cuda", "float16"
    return "cpu", "int8"


def to_json(
    audio_dir: str | Path = AUDIO_DIR,
    output_dir: str | Path = JSON_DIR,
    audio_files: list[str] | None = None,
) -> None:
    from faster_whisper import WhisperModel

    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    output_dir.mkdir(exist_ok=True)
    requested_files = set(audio_files or [])
    files_to_process = sorted(
        path
        for path in audio_dir.iterdir()
        if path.is_file() and (not requested_files or path.name in requested_files)
    )
    if not files_to_process:
        print("No audio files needed transcription.")
        return

    device, compute_type = _resolve_compute_settings()
    model = WhisperModel(WHISPER_MODEL_NAME, device=device, compute_type=compute_type)

    for file_path in files_to_process:
        print(f"Processing audio {file_path.name}")
        segments, info = model.transcribe(
            str(file_path),
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        segment_payload = []
        transcript_parts: list[str] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            transcript_parts.append(text)
            segment_payload.append(
                {
                    "id": segment.id,
                    "seek": getattr(segment, "seek", 0),
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": text,
                    "tokens": getattr(segment, "tokens", []),
                    "temperature": getattr(segment, "temperature", 0.0),
                    "avg_logprob": getattr(segment, "avg_logprob", 0.0),
                    "compression_ratio": getattr(segment, "compression_ratio", 0.0),
                    "no_speech_prob": getattr(segment, "no_speech_prob", 0.0),
                }
            )

        result = {
            "text": " ".join(transcript_parts).strip(),
            "segments": segment_payload,
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
        }

        output_path = output_dir / f"{file_path.stem}.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=4, default=str)

    print()
