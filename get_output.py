from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ROOT_DIR = Path(__file__).resolve().parent
DATAFRAME_PATH = ROOT_DIR / "dataframe.joblib"
RESPONSE_PATH = ROOT_DIR / "response.txt"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OPENROUTER_MODEL = "openrouter/free"

load_dotenv()

_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def create_embedding(texts: list[str]) -> np.ndarray:
    if not texts:
        raise ValueError("Expected at least one text to embed.")
    return get_embedding_model().encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def format_timestamp(seconds: float | str) -> str:
    total_seconds = max(0, round(float(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def retrieve_relevant_chunks(question: str, top_k: int = 5):
    if not DATAFRAME_PATH.exists():
        raise FileNotFoundError(
            "No dataframe found. Run the processing pipeline before asking questions."
        )

    df = joblib.load(DATAFRAME_PATH)
    if df.empty:
        raise RuntimeError("The dataframe is empty. Reprocess the source videos.")

    question_embedding = create_embedding([question])[0]
    embeddings = np.vstack(df.embedding.values)
    similarity = cosine_similarity(embeddings, [question_embedding]).flatten()

    result_count = max(1, min(top_k, len(df)))
    max_indices = similarity.argsort()[::-1][:result_count]

    relevant_df = df.loc[max_indices].copy()
    relevant_df["score"] = similarity[max_indices]
    relevant_df["start"] = relevant_df["start"].apply(format_timestamp)
    relevant_df["end"] = relevant_df["end"].apply(format_timestamp)

    return relevant_df[["video_name", "text", "start", "end", "score"]]


def build_prompt(question: str, subtitle_chunks: list[dict]) -> str:
    return f"""
You are an assistant helping students learn from course videos.

Relevant transcript chunks:
{subtitle_chunks}

Student question:
"{question}"

Instructions:
1. Answer only using the transcript evidence above.
2. If the answer is present, explain it clearly in natural language.
3. Mention the most relevant video title and timestamp ranges in mm:ss format.
4. If the question is unrelated or unsupported by the transcript, say you can only answer from the available course content.
5. Keep the response suitable for terminal output. Do not use markdown bold.
""".strip()


def inference(prompt: str) -> str:
    api_key = os.getenv("MIMO_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Missing MIMO_API_KEY. Add it to your .env file before asking questions."
        )

    model_name = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL).strip() or DEFAULT_OPENROUTER_MODEL
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or "No response returned."


def answer_question(question: str, top_k: int = 5) -> str:
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("Question cannot be empty.")

    relevant_chunks = retrieve_relevant_chunks(cleaned_question, top_k=top_k)
    subtitle_chunks = relevant_chunks.drop(columns=["score"]).to_dict(orient="records")
    prompt = build_prompt(cleaned_question, subtitle_chunks)
    response = inference(prompt)

    RESPONSE_PATH.write_text(response, encoding="utf-8")
    return response


def get_response(question: str | None = None, top_k: int = 5) -> None:
    if question is not None:
        print(answer_question(question, top_k=top_k))
        return

    while True:
        print("\n")
        current_question = input("Ask your question (or type 'exit' to quit): ").strip()

        if current_question.lower() in {"exit", "quit"}:
            print("Session ended.")
            return

        if not current_question:
            print("Please enter a question.")
            continue

        print(answer_question(current_question, top_k=top_k))
