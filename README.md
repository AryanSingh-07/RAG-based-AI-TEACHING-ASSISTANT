# RAG-Based AI Teaching Assistant for Educational Videos

This project turns educational videos into a searchable knowledge base, then answers questions with transcript-backed responses and timestamp references.

## What It Does

- Converts videos into audio with `ffmpeg`
- Transcribes audio locally with Whisper
- Cleans transcript segments into retrieval-ready chunks
- Builds embeddings with `sentence-transformers`
- Retrieves the most relevant chunks for a question
- Uses an OpenRouter-hosted LLM to generate a student-friendly answer
- Generates lecture notes with important slide captures
- Exports notes as Markdown, DOCX, and PDF

## Requirements

- Python 3.10 or higher recommended
- `ffmpeg` available on your system PATH
- An OpenRouter-compatible API key stored as `MIMO_API_KEY` or `OPENROUTER_API_KEY`
- For slide OCR in lecture notes:
  - Tesseract OCR installed on your system PATH
  - the Python dependencies from `requirements.txt`

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
MIMO_API_KEY=your_openrouter_api_key_here
# or
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Optional model overrides:

```env
OPENROUTER_MODEL=openrouter/free
LECTURE_NOTES_MODEL=openrouter/free
WHISPER_DEVICE=cpu
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
python main.py --generate-notes --no-interactive
python main.py --notes-videos "Angular\\Learn NgModule in Angular with Examples [oqZ4-ULwfbc].mp4" --no-interactive
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
- Process selected videos from the library sidebar
- Import videos from a YouTube playlist
- Ask questions with a textarea instead of the terminal
- Inspect the retrieved transcript chunks and similarity scores
- Generate lecture notes for selected videos
- Preview generated notes as PDF in the browser
- Download notes as PDF, DOCX, or Markdown from the main screen

The project includes `.streamlit/config.toml` with file watching disabled to avoid the known Streamlit + `torch.classes` watcher crash.

## Lecture Notes Workflow

The notes pipeline reuses your processed transcripts instead of starting from scratch:

1. Process a video so a transcript JSON exists in `json_data/`
2. Generate notes from the dashboard Library section or with the CLI
3. The notes pipeline:
   - builds larger transcript sections
   - removes near-duplicate transcript segments
   - samples OCR-rich frames from each section
   - keeps distinct important slides
   - writes notes to `output/notes/`

Generated note files:

- `output/notes/<video>.md`
- `output/notes/<video>.docx`
- `output/notes/<video>.pdf`
- `output/notes/frames/<video>/...`

## Project Structure

```text
.
├── main.py
├── dashboard.py
├── notes_generator.py
├── video_tranformer.py
├── audio_transformer.py
├── json_processor.py
├── data_processor.py
├── get_output.py
├── .streamlit/
│   └── config.toml
├── videos/
├── audios/
├── json_data/
├── clean_json_data/
├── output/
│   └── notes/
├── dataframe.joblib
├── processed_videos.joblib
└── response.txt
```

## Notes

- Transcript embeddings are cached in `dataframe.joblib`
- Processed video filenames are cached in `processed_videos.joblib`
- The latest generated answer is written to `response.txt`
- If you add, remove, or rename a video, the pipeline rebuilds automatically
- Generated notes and note preview assets are treated as build artifacts and are ignored by git

## Recent Improvements

- Added a proper CLI entrypoint with flags for forced rebuilds and one-shot questions
- Reduced repeated model loading for better performance
- Improved path handling and file validation
- Added clearer runtime errors for missing data and missing API keys
- Cleaned dependencies to match the actual codebase
- Added playlist-aware incremental processing
- Added larger transcript chunking for retrieval
- Added lecture notes generation with OCR-backed important slides
- Added PDF preview and notes downloads in the dashboard
- Added a Streamlit config workaround for `torch.classes` watcher errors

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE).
