from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parent
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def create_embeddings(texts: list[str]):
    if not texts:
        return []
    return get_embedding_model().encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def build_dataframe(
    json_dir: str | Path = ROOT_DIR / "clean_json_data",
    videos_dir: str | Path = ROOT_DIR / "videos",
    df_out: str | Path = ROOT_DIR / "dataframe.joblib",
    videos_out: str | Path = ROOT_DIR / "processed_videos.joblib",
    json_files: list[str] | None = None,
    reset: bool = False,
) -> None:
    json_dir = Path(json_dir)
    videos_dir = Path(videos_dir)
    df_out = Path(df_out)
    videos_out = Path(videos_out)

    if not json_dir.exists():
        raise FileNotFoundError(f"Clean JSON directory not found: {json_dir}")

    if not reset and df_out.exists():
        dataframe = joblib.load(df_out)
        records = dataframe.to_dict(orient="records")
        chunk_id = int(dataframe["id"].max()) + 1 if not dataframe.empty else 0
        existing_video_names = set(dataframe["video_name"].astype(str))
    else:
        records = []
        chunk_id = 0
        existing_video_names: set[str] = set()

    requested_files = set(json_files or [])
    json_paths = sorted(path for path in json_dir.glob("*.json") if not requested_files or path.name in requested_files)

    for path in json_paths:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        video_name = path.stem
        if video_name in existing_video_names:
            continue

        chunks = data.get("chunks", [])
        texts = [chunk.get("text", "").strip() for chunk in chunks if chunk.get("text", "").strip()]

        if not texts:
            continue

        print(f"Embedding -> {path.name}")
        embeddings = create_embeddings(texts)

        if len(embeddings) != len(texts):
            raise RuntimeError(f"Embedding count mismatch for {path.name}")

        emb_idx = 0
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue

            record = {
                "id": chunk_id,
                "video_name": chunk["video_name"],
                "text": text,
                "start": float(chunk["start"]),
                "end": float(chunk["end"]),
                "embedding": embeddings[emb_idx],
            }
            records.append(record)
            chunk_id += 1
            emb_idx += 1

    if not records:
        raise RuntimeError("No transcript chunks were available to embed.")

    dataframe = pd.DataFrame(records)
    if "embedding" in dataframe.columns:
        dataframe["embedding"] = dataframe["embedding"].apply(lambda value: np.asarray(value, dtype=np.float32))
    joblib.dump(dataframe, df_out)

    if videos_dir.exists():
        video_files = sorted(
            str(file.relative_to(videos_dir))
            for file in videos_dir.rglob("*")
            if file.is_file()
        )
        joblib.dump(video_files, videos_out)

    print(f"Saved {len(dataframe)} chunks to {df_out.name}")


if __name__ == "__main__":
    build_dataframe()
