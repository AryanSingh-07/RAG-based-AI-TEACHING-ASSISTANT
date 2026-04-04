# RAG-Based AI Teaching Assistant for Educational Videos

This project turns educational videos into a searchable knowledge base, then answers questions with transcript-backed responses and timestamp references.

## What It Does

- Converts videos into audio with `ffmpeg`
- Transcribes audio locally with Whisper
- Cleans transcript segments into retrieval-ready chunks
- Builds embeddings with `sentence-transformers`
- Retrieves the most relevant chunks for a question
- Uses an OpenRouter-hosted LLM to generate a student-friendly answer

## Requirements

- Python 3.10 or higher recommended
- `ffmpeg` available on your system PATH
- An OpenRouter-compatible API key stored as `MIMO_API_KEY`

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
MIMO_API_KEY=your_openrouter_api_key_here
```

## Usage

Place your source videos in [`videos/`](./videos), then run:

```bash
python main.py
```

The app will:

1. Detect whether videos changed since the last build
2. Re-run the pipeline only when needed
3. Start an interactive question loop

Useful CLI options:

```bash
python main.py --force-reprocess
python main.py --question "What is Java bytecode?"
python main.py --question "What is TensorFlow?" --top-k 3
python main.py --no-interactive
```

Type `exit` or `quit` to leave the interactive Q&A loop.

## Web Dashboard

For a friendlier interface, launch the dashboard:

```bash
streamlit run dashboard.py
```

The dashboard lets you:

- Upload videos directly into the project
- Rebuild the processing pipeline from the browser
- Ask questions with a textarea instead of the terminal
- Inspect the retrieved transcript chunks and similarity scores

## Project Structure

```text
.
├── main.py
├── dashboard.py
├── video_tranformer.py
├── audio_transformer.py
├── json_processor.py
├── data_processor.py
├── get_output.py
├── videos/
├── audios/
├── json_data/
├── clean_json_data/
├── dataframe.joblib
├── processed_videos.joblib
└── response.txt
```

## Notes

- Transcript embeddings are cached in `dataframe.joblib`
- Processed video filenames are cached in `processed_videos.joblib`
- The latest generated answer is written to `response.txt`
- If you add, remove, or rename a video, the pipeline rebuilds automatically

## Recent Improvements

- Added a proper CLI entrypoint with flags for forced rebuilds and one-shot questions
- Reduced repeated model loading for better performance
- Improved path handling and file validation
- Added clearer runtime errors for missing data and missing API keys
- Cleaned dependencies to match the actual codebase

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE).
