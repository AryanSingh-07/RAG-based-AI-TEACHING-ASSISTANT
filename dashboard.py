from __future__ import annotations

from pathlib import Path

import streamlit as st

import get_output
import main
import playlist_downloader
from pipeline_utils import canonical_video_name

ROOT_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = ROOT_DIR / "videos"
SUPPORTED_UPLOAD_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".avi"}


st.set_page_config(
    page_title="RAG Video Assistant",
    page_icon="VA",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --bg-page-top: #eef4f1;
            --bg-page-bottom: #dbe8e2;
            --bg-card: rgba(251, 254, 252, 0.82);
            --bg-card-strong: rgba(248, 252, 250, 0.96);
            --bg-sidebar-top: #18362f;
            --bg-sidebar-bottom: #102821;
            --bg-toolbar: rgba(244, 250, 247, 0.9);
            --text-strong: #13211e;
            --text-muted: #536762;
            --text-soft: #6d827c;
            --border-soft: rgba(19, 33, 30, 0.12);
            --border-strong: rgba(19, 33, 30, 0.26);
            --accent-dark: #1b4d42;
            --accent-dark-hover: #245f52;
            --accent-mid: #2f7d6b;
            --accent-light: #eff8f4;
            --accent-wash: rgba(47, 125, 107, 0.12);
            --shadow-color: rgba(18, 46, 40, 0.12);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(72, 153, 128, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(228, 177, 102, 0.18), transparent 32%),
                linear-gradient(180deg, var(--bg-page-top) 0%, var(--bg-page-bottom) 100%);
            color: var(--text-strong);
            font-family: 'Space Grotesk', sans-serif;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: -0.02em;
            color: var(--text-strong);
        }

        code, pre {
            font-family: 'IBM Plex Mono', monospace !important;
        }

        .hero-card {
            padding: 1.4rem 1.5rem;
            border-radius: 22px;
            background: var(--bg-card);
            border: 1px solid var(--border-soft);
            box-shadow: 0 18px 45px var(--shadow-color);
            margin-bottom: 1rem;
            backdrop-filter: blur(12px);
        }

        .answer-card {
            padding: 1.2rem 1.3rem;
            border-radius: 20px;
            background: var(--bg-card-strong);
            border: 1px solid var(--border-soft);
            box-shadow: 0 14px 32px rgba(18, 46, 40, 0.08);
            margin-top: 0.9rem;
        }

        .metric-card {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: rgba(249, 253, 251, 0.76);
            border: 1px solid var(--border-soft);
            min-height: 120px;
            box-shadow: 0 12px 28px rgba(18, 46, 40, 0.07);
        }

        .metric-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-soft);
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            margin-top: 0.35rem;
            color: var(--accent-mid);
        }

        .metric-subtle {
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-top: 0.35rem;
        }

        .stApp,
        .stApp label,
        .stApp .stMarkdown,
        .stApp .stText,
        .stApp .stCaption,
        .stApp p,
        .stApp span,
        .stApp div,
        .stApp li {
            color: var(--text-strong);
        }

        .stApp [data-testid="stCaptionContainer"] {
            color: var(--text-muted);
        }

        [data-testid="stSidebar"] * {
            color: #eef8f4 !important;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--bg-sidebar-top) 0%, var(--bg-sidebar-bottom) 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }

        [data-testid="stSidebar"] .stTable,
        [data-testid="stSidebar"] .stDataFrame {
            background: rgba(26, 55, 47, 0.6) !important;
            border-radius: 14px !important;
            border: 1px solid rgba(238, 248, 244, 0.18) !important;
        }

        [data-testid="stSidebar"] .stTable th,
        [data-testid="stSidebar"] .stTable td,
        [data-testid="stSidebar"] .stDataFrame th,
        [data-testid="stSidebar"] .stDataFrame td {
            color: #eef8f4 !important;
            background: transparent !important;
        }

        /* Sidebar caption and highlight text */
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] p {
            color: #eef8f4 !important;
        }

        header[data-testid="stHeader"] {
            background: var(--bg-toolbar) !important;
            border-bottom: 1px solid var(--border-soft);
            backdrop-filter: blur(10px);
        }

        [data-testid="stToolbar"],
        [data-testid="stToolbar"] *,
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        [data-testid="stMainMenu"] {
            color: var(--text-strong) !important;
        }

        [data-testid="stToolbar"] button,
        [data-testid="stToolbar"] [role="button"],
        [data-testid="stMainMenu"] button,
        [data-testid="stStatusWidget"] button {
            background: rgba(248, 252, 250, 0.95) !important;
            color: var(--text-strong) !important;
            border: 1px solid var(--border-soft) !important;
            border-radius: 12px !important;
        }

        [data-testid="stToolbar"] button:hover,
        [data-testid="stToolbar"] [role="button"]:hover,
        [data-testid="stMainMenu"] button:hover,
        [data-testid="stStatusWidget"] button:hover {
            border-color: var(--border-strong) !important;
            background: #f3e8da !important;
        }

        .stTextArea textarea,
        .stTextInput input,
        .stNumberInput input,
        .stSelectbox [data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div {
            background: var(--bg-card-strong) !important;
            color: var(--text-strong) !important;
            border: 1px solid var(--border-soft) !important;
            border-radius: 14px !important;
            box-shadow: 0 4px 14px rgba(18, 46, 40, 0.04) !important;
        }

        .stTextArea textarea:focus,
        .stTextInput input:focus,
        .stNumberInput input:focus,
        .stSelectbox [data-baseweb="select"] > div:focus-within,
        .stMultiSelect [data-baseweb="select"] > div:focus-within {
            border-color: var(--accent-mid) !important;
            box-shadow: 0 0 0 1px var(--accent-mid) !important;
            outline: none !important;
        }

        .stTextArea textarea::placeholder,
        .stTextInput input::placeholder {
            color: var(--text-muted) !important;
        }

        .stCheckbox {
            padding: 0.2rem 0;
        }

        .stCheckbox [data-baseweb="checkbox"] > div {
            border-color: rgba(238, 248, 244, 0.42) !important;
            background: rgba(255, 255, 255, 0.04) !important;
        }

        .stCheckbox input:checked + div,
        .stCheckbox [data-baseweb="checkbox"] input:checked ~ div {
            background: #8fd3bb !important;
            border-color: #8fd3bb !important;
        }

        .stSlider [data-baseweb="slider"] {
            padding-top: 0.5rem;
        }

        .stSlider [role="slider"] {
            background: var(--accent-mid) !important;
            box-shadow: none !important;
        }

        .stSlider [data-testid="stTickBar"],
        .stSlider div[data-baseweb="slider"] > div > div {
            background: rgba(47, 125, 107, 0.2) !important;
        }

        [data-testid="stFileUploader"] {
            background: transparent;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: var(--bg-card-strong);
            border: 1px solid var(--border-soft);
            border-radius: 18px;
            box-shadow: 0 8px 24px rgba(18, 46, 40, 0.05);
        }

        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] p,
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploaderDropzoneInstructions"] span,
        [data-testid="stFileUploaderDropzoneInstructions"] small {
            color: var(--text-strong) !important;
        }

        [data-testid="stFileUploaderDropzone"] section {
            border-color: var(--border-soft) !important;
        }

        .stButton > button,
        [data-testid="stFileUploaderDropzone"] button {
            background: var(--accent-light) !important;
            color: var(--text-strong) !important;
            border: 1px solid var(--border-soft) !important;
            border-radius: 14px !important;
            font-weight: 600 !important;
            box-shadow: 0 8px 20px rgba(18, 46, 40, 0.06);
            text-shadow: none !important;
        }

        .stButton > button:hover,
        [data-testid="stFileUploaderDropzone"] button:hover {
            background: #e2f3ec !important;
            border-color: var(--border-strong) !important;
        }

        .stButton > button[kind="primary"],
        .stButton > button[data-testid*="primary"],
        .stButton button[type="primary"],
        .stButton button[data-testid="baseButton-primary"] {
            background: var(--accent-dark) !important;
            color: #ffffff !important;
            border-color: var(--accent-dark) !important;
            font-weight: 700 !important;
            letter-spacing: 0.01em;
            text-shadow: 0 1px 6px rgba(0, 0, 0, 0.28);
        }

        .stButton > button[kind="primary"]:hover,
        .stButton > button[data-testid*="primary"]:hover,
        .stButton button[type="primary"]:hover,
        .stButton button[data-testid="baseButton-primary"]:hover {
            background: var(--accent-dark-hover) !important;
            border-color: var(--accent-dark-hover) !important;
        }

        .stButton > button span,
        .stButton button span {
            color: inherit !important;
        }

        [data-testid="stSidebar"] .stButton > button,
        [data-testid="stSidebar"] .stButton > button *,
        [data-testid="stSidebar"] .stButton > button:disabled,
        [data-testid="stSidebar"] .stButton > button:disabled * {
            color: var(--text-strong) !important;
            text-shadow: none !important;
            -webkit-text-fill-color: var(--text-strong) !important;
            opacity: 1 !important;
        }

        [data-testid="stSidebar"] .stButton > button[kind="primary"],
        [data-testid="stSidebar"] .stButton > button[kind="primary"] *,
        [data-testid="stSidebar"] .stButton > button[data-testid*="primary"],
        [data-testid="stSidebar"] .stButton > button[data-testid*="primary"] *,
        [data-testid="stSidebar"] .stButton button[type="primary"],
        [data-testid="stSidebar"] .stButton button[type="primary"] *,
        [data-testid="stSidebar"] .stButton button[data-testid="baseButton-primary"],
        [data-testid="stSidebar"] .stButton button[data-testid="baseButton-primary"] * {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }

        /* Sidebar select readability */
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: rgba(26, 55, 47, 0.65) !important;
            border: 1px solid rgba(238, 248, 244, 0.22) !important;
            color: #eef8f4 !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] * {
            color: #eef8f4 !important;
        }

        .stButton > button:focus,
        [data-testid="stFileUploaderDropzone"] button:focus {
            box-shadow: 0 0 0 2px var(--accent-wash) !important;
            outline: none !important;
        }

        .stTable,
        .stDataFrame {
            background: var(--bg-card);
            border-radius: 18px;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--border-soft);
            border-radius: 18px;
            overflow: hidden;
        }

        .stAlert {
            background: var(--bg-card-strong);
            color: var(--text-strong);
            border: 1px solid var(--border-soft);
        }

        [data-testid="stStatusWidget"] p,
        [data-testid="stToolbar"] p {
            color: var(--text-strong) !important;
        }

        hr {
            border-color: rgba(19, 33, 30, 0.08) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric(label: str, value: str, subtle: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtle">{subtle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_video_label(path_or_name: str) -> str:
    rel_path = Path(path_or_name)
    pretty_name = canonical_video_name(rel_path.name)
    if len(rel_path.parts) > 1:
        return f"{rel_path.parts[0]} / {pretty_name}"
    return pretty_name


def save_uploaded_files(uploaded_files) -> list[str]:
    VIDEOS_DIR.mkdir(exist_ok=True)
    saved_files: list[str] = []
    skipped_files: list[str] = []

    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
            st.warning(f"Skipped unsupported file: {uploaded_file.name}")
            continue

        destination = VIDEOS_DIR / uploaded_file.name
        if destination.exists():
            skipped_files.append(uploaded_file.name)
            continue

        destination.write_bytes(uploaded_file.getbuffer())
        saved_files.append(uploaded_file.name)

    if skipped_files:
        st.info(
            "Skipped already existing video(s): " + ", ".join(skipped_files)
        )

    return saved_files


def run_pipeline_from_ui(force_reprocess: bool) -> None:
    with st.status("Running processing pipeline...", expanded=True) as status:
        try:
            plan = main.get_pipeline_status()
            if not force_reprocess and not plan["needs_reprocess"]:
                status.update(
                    label="Pipeline is already up to date.",
                    state="complete",
                    expanded=False,
                )
                st.success("The cached knowledge base already matches the current videos.")
                return

            planned_mode = "full" if force_reprocess else plan["pipeline_mode"]
            status.write(f"Pipeline mode: {planned_mode}")
            if plan["new_videos"]:
                status.write("New videos: " + ", ".join(plan["new_videos"]))
            if plan["changed_videos"]:
                status.write("Changed videos trigger full rebuild: " + ", ".join(plan["changed_videos"]))
            if plan["removed_videos"]:
                status.write("Removed videos trigger full rebuild: " + ", ".join(plan["removed_videos"]))

            main.run_pipeline(force=force_reprocess)

            status.update(
                label="Pipeline complete. Your dashboard is ready for questions.",
                state="complete",
                expanded=False,
            )
            st.success("Processing finished successfully.")
        except Exception as exc:
            status.update(label="Pipeline failed.", state="error", expanded=True)
            st.error(str(exc))


def ask_question_from_ui(question: str, top_k: int) -> None:
    cleaned_question = question.strip()
    if not cleaned_question:
        st.warning("Enter a question before running retrieval.")
        return

    st.session_state["last_error"] = None
    st.session_state["last_response"] = None
    st.session_state["last_chunks"] = None

    try:
        response = get_output.answer_question(cleaned_question, top_k=top_k)
        chunks = get_output.retrieve_relevant_chunks(cleaned_question, top_k=top_k)
    except Exception as exc:
        st.session_state["last_error"] = str(exc)
        return

    st.session_state["last_question"] = cleaned_question
    st.session_state["last_response"] = response
    st.session_state["last_chunks"] = chunks


def render_sidebar(status: dict[str, object]) -> tuple[bool, bool, list[str], bool, bool]:
    with st.sidebar:
        st.markdown("## Control Room")
        st.caption("Manage videos, rebuild the knowledge base, and tune retrieval.")

        force_reprocess = st.checkbox("Force full rebuild", value=False)
        process_clicked = st.button("Process Rebuild", use_container_width=True, type="primary")

        st.divider()
        st.markdown("### Library")
        videos = status["current_videos"]
        # Group by top-level folder (course/playlist)
        groups = {}
        for relpath in videos:
            parts = Path(relpath).parts
            group = parts[0] if len(parts) > 1 else "(root)"
            groups.setdefault(group, []).append(relpath)

        group_names = ["All"] + sorted(groups)
        selected_group = st.selectbox("Course / Playlist", group_names, index=0)

        display_list = videos if selected_group == "All" else groups.get(selected_group, [])

        selected_specific: list[str] = []
        process_selected_clicked = False
        generate_notes_clicked = False
        if display_list:
            selected_specific = st.multiselect(
                "Select videos to process",
                options=display_list,
                default=[],
                format_func=format_video_label,
            )
            st.caption("These will run without forcing a full rebuild.")
            process_selected_clicked = st.button(
                "Process Selected Videos",
                use_container_width=True,
                disabled=not selected_specific,
            )
            generate_notes_clicked = st.button(
                "Generate Notes",
                use_container_width=True,
                disabled=not selected_specific,
            )
        else:
            st.caption("No videos found yet.")

        st.divider()
        st.markdown("### Processed Videos")
        processed_videos = status["processed_videos"]
        if processed_videos:
            processed_rows = [{"Video": format_video_label(video_name)} for video_name in processed_videos]
            st.table(processed_rows)
        else:
            st.caption("No processed videos available yet.")

    return force_reprocess, process_clicked, selected_specific, process_selected_clicked, generate_notes_clicked


def render_dashboard() -> None:
    apply_theme()
    status = main.get_pipeline_status()
    force_reprocess, process_clicked, selected_specific, process_selected_clicked, generate_notes_clicked = render_sidebar(status)

    if process_clicked:
        run_pipeline_from_ui(force_reprocess=force_reprocess)
        status = main.get_pipeline_status()

    if process_selected_clicked:
        try:
            main.run_pipeline(force=False, selected_videos=selected_specific)
            st.success(f"Processed {len(selected_specific)} selected video(s).")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    if generate_notes_clicked:
        try:
            import notes_generator

            results = notes_generator.generate_notes(selected_specific)
            st.success(f"Generated notes for {len(results)} selected video(s).")
            for result in results:
                st.info(f"Saved notes to {result['markdown_path']}")
        except Exception as exc:
            st.error(str(exc))

    st.markdown(
        """
        <div class="hero-card">
            <h1>Video Knowledge Dashboard</h1>
            <p>Upload course videos, build the transcript index, and ask grounded questions with timestamped evidence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric(
            "Videos Loaded",
            str(len(status["current_videos"])),
            "Source files currently available in the project.",
        )
    with col2:
        render_metric(
            "Knowledge Base",
            "Ready" if status["dataframe_exists"] else "Missing",
            "Embeddings cache for semantic retrieval.",
        )
    with col3:
        render_metric(
            "Sync State",
            "Needs rebuild" if status["needs_reprocess"] else "In sync",
            "Checks whether your processed cache matches the video folder.",
        )

    left_col, right_col = st.columns([1.05, 1.4], gap="large")

    with left_col:
        st.subheader("Upload Videos")
        uploaded_files = st.file_uploader(
            "Drop new course videos here",
            accept_multiple_files=True,
            type=[suffix.lstrip(".") for suffix in sorted(SUPPORTED_UPLOAD_SUFFIXES)],
        )

        if st.button("Save Uploaded Files", use_container_width=True):
            if not uploaded_files:
                st.info("Choose one or more video files first.")
            else:
                saved_files = save_uploaded_files(uploaded_files)
                if saved_files:
                    st.success(f"Saved {len(saved_files)} file(s) to the videos folder.")
                    st.rerun()

        st.subheader("Pipeline Snapshot")
        st.write(f"Processed cache exists: {'Yes' if status['processed_cache_exists'] else 'No'}")
        st.write(f"Embeddings dataframe exists: {'Yes' if status['dataframe_exists'] else 'No'}")
        st.write(f"Needs reprocess: {'Yes' if status['needs_reprocess'] else 'No'}")

    with right_col:
        st.subheader("Ask the Assistant")
        question = st.text_area(
            "Question",
            placeholder="Ask something grounded in the uploaded videos...",
            height=140,
        )
        top_k = st.slider("Retrieved chunks", min_value=1, max_value=10, value=5)

        if st.button("Generate Answer", use_container_width=True, type="secondary"):
            ask_question_from_ui(question, top_k)

        error_message = st.session_state.get("last_error")
        response = st.session_state.get("last_response")
        chunks = st.session_state.get("last_chunks")

        if error_message:
            st.error(f"Answer generation failed: {error_message}")

        if response:
            st.markdown("### Answer")
            st.markdown(
                f"""
                <div class="answer-card">
                    {response}
                </div>
                """,
                unsafe_allow_html=True,
            )

    if chunks is not None:
        chunks = chunks.copy()
        chunks["video_name"] = chunks["video_name"].map(format_video_label)
        st.markdown("### Retrieved Evidence")
        st.dataframe(
            chunks,
            use_container_width=True,
            hide_index=True,
            column_config={
                "video_name": "Video",
                "text": "Transcript",
                "start": "Start",
                "end": "End",
                "score": st.column_config.NumberColumn("Score", format="%.3f"),
            },
        )

    st.divider()
    st.subheader("YouTube Playlist Import (Beta)")
    playlist_url = st.text_input("Playlist URL", placeholder="https://www.youtube.com/playlist?list=...")

    if "playlist_cache" not in st.session_state:
        st.session_state["playlist_cache"] = {}

    col_fetch, col_download = st.columns([1, 1], gap="medium")
    selected_ids: set[str] = set()

    with col_fetch:
        if st.button("Fetch Playlist", type="secondary", disabled=not playlist_url.strip()):
            try:
                title, videos = playlist_downloader.fetch_playlist(playlist_url.strip())
                st.session_state["playlist_cache"] = {
                    "title": title,
                    "videos": videos,
                    "url": playlist_url.strip(),
                }
                st.success(f"Found {len(videos)} videos in '{title}'.")
            except Exception as exc:
                st.error(f"Failed to fetch playlist: {exc}")

        cache = st.session_state.get("playlist_cache", {})
        if cache:
            vids = cache.get("videos", [])
            titles = [f"{v['title']} ({v['id']})" for v in vids]
            ids = [v["id"] for v in vids]
            selected = st.multiselect(
                "Select videos to download",
                options=ids,
                format_func=lambda vid: next((t for t, i in zip(titles, ids) if i == vid), vid),
            )
            selected_ids = set(selected)

    with col_download:
        if st.button("Download Selected", type="primary", disabled=not selected_ids):
            cache = st.session_state.get("playlist_cache", {})
            vids = cache.get("videos", [])
            title = cache.get("title", "playlist")
            try:
                downloaded = playlist_downloader.download_selected(title, vids, selected_ids)
                if downloaded:
                    st.success(f"Downloaded {len(downloaded)} video(s) to videos/{playlist_downloader._slugify(title)}/")
                    st.session_state["playlist_cache"] = {}
                    st.experimental_rerun()
                else:
                    st.info("No files were downloaded. Check your selection or playlist access.")
            except Exception as exc:
                st.error(f"Download failed: {exc}")


if __name__ == "__main__":
    render_dashboard()
