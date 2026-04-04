from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from pipeline_utils import (
    get_current_video_fingerprints,
    get_pipeline_plan,
    save_pipeline_state,
)

ROOT_DIR = Path(__file__).resolve().parent
DATAFRAME_PATH = ROOT_DIR / "dataframe.joblib"
PROCESSED_LIST_PATH = ROOT_DIR / "processed_videos.joblib"
VIDEOS_DIR = ROOT_DIR / "videos"


def get_current_videos() -> list[str]:
    return sorted(get_current_video_fingerprints())


def should_reprocess(force: bool = False) -> bool:
    return get_pipeline_plan(force=force)["mode"] != "noop"


def get_pipeline_status() -> dict[str, object]:
    plan = get_pipeline_plan(force=False)
    current_videos = get_current_videos()
    processed_videos = []

    if PROCESSED_LIST_PATH.exists():
        processed_videos = sorted(joblib.load(PROCESSED_LIST_PATH))

    return {
        "videos_dir": str(VIDEOS_DIR),
        "dataframe_exists": DATAFRAME_PATH.exists(),
        "processed_cache_exists": PROCESSED_LIST_PATH.exists(),
        "current_videos": current_videos,
        "processed_videos": processed_videos,
        "needs_reprocess": plan["mode"] != "noop",
        "pipeline_mode": plan["mode"],
        "new_videos": plan["new_videos"],
        "changed_videos": plan["changed_videos"],
        "removed_videos": plan["removed_videos"],
    }


def run_pipeline(force: bool = False, selected_videos: list[str] | None = None) -> None:
    import audio_transformer
    import data_processor
    import json_processor
    import video_tranformer

    if selected_videos:
        plan = {
            "mode": "selected",
            "new_videos": selected_videos,
            "current_videos": get_current_video_fingerprints(),
            "current_fingerprints": {},
        }
    else:
        plan = get_pipeline_plan(force=force)
    mode = plan["mode"]

    if mode == "noop":
        print("Cached dataframe is up to date. Skipping processing steps.\n")
        return

    target_videos = None if mode == "full" else plan["new_videos"]
    if mode == "full":
        target_audio = None
        target_json = None
    else:
        target_stems = [
            video_tranformer.video_output_stem(VIDEOS_DIR / name, VIDEOS_DIR)
            for name in target_videos
        ]
        target_audio = [f"{stem}.mp3" for stem in target_stems]
        target_json = [f"{stem}.json" for stem in target_stems]

    print(f"Pipeline mode: {mode}\n")
    print("Processing videos...\n")
    video_tranformer.to_audio(video_files=target_videos)

    print("\nConverting audio files to transcript JSON...\n")
    audio_transformer.to_json(audio_files=target_audio)

    print("\nCleaning transcript JSON...\n")
    json_processor.cleaning_json(json_files=target_json)

    print("\nCreating embeddings and building the dataframe...\n")
    data_processor.build_dataframe(json_files=target_json, reset=mode == "full")

    current_videos = sorted(get_current_video_fingerprints())
    joblib.dump(current_videos, PROCESSED_LIST_PATH)
    save_pipeline_state({"videos": get_current_video_fingerprints()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG teaching assistant for educational videos."
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Rebuild the pipeline even if cached data already exists.",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Ask a single question without entering interactive mode.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of relevant transcript chunks to retrieve.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip the question-answer loop after processing.",
    )
    parser.add_argument(
        "--generate-notes",
        action="store_true",
        help="Generate lecture notes for the current video library after processing.",
    )
    parser.add_argument(
        "--notes-videos",
        nargs="+",
        help="Generate notes only for the provided relative video paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if should_reprocess(force=args.force_reprocess):
        run_pipeline(force=args.force_reprocess)
    else:
        print("Cached dataframe is up to date. Skipping processing steps.\n")

    if args.generate_notes or args.notes_videos:
        import notes_generator

        note_targets = args.notes_videos or get_current_videos()
        results = notes_generator.generate_notes(note_targets)
        for result in results:
            print(
                f"Generated notes for {result['video']} -> {result['markdown_path']}"
            )

    if args.no_interactive and not args.question:
        return

    import get_output

    get_output.get_response(question=args.question, top_k=args.top_k)


if __name__ == "__main__":
    main()
