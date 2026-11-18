"""PolicyPulse – Policy Sentiment Intelligence Platform.

Core engine: VADER (lexicon/rule-based sentiment analysis).
Input: user-uploaded CSV/XLS/XLSX datasets. No live X/Twitter API connection.
"""
from __future__ import annotations

import codecs
import csv
import html
import itertools
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
APP_NAME = "PolicyPulse"
CHUNK_SIZE = 10_000
TABLE_PREVIEW_LIMIT = 200
GEMINI_DEFAULT_MODEL = "gemini-flash-latest"

SENTIMENT_COLORS = {
    "Positive": "#22C55E",
    "Negative": "#EF4444",
    "Neutral": "#94A3B8",
}

TEXT_COLUMN_CANDIDATES = [
    "tweet_text", "full_text", "text", "tweet", "post_text", "post",
    "content", "message", "status", "comment", "review", "body",
]
DATE_COLUMN_CANDIDATES = [
    "created_at", "createdat", "date", "datetime", "timestamp", "time",
    "posted_at", "tweet_date",
]
CATEGORY_COLUMN_CANDIDATES = [
    "policy_category", "policy_area", "category", "policy", "topic",
    "subject", "government_policy",
]

STOP_WORDS = {
    "the", "and", "for", "you", "your", "are", "was", "were", "with",
    "that", "this", "from", "have", "has", "had", "but", "not", "they",
    "them", "their", "there", "then", "than", "when", "what", "why",
    "how", "who", "will", "would", "could", "should", "can", "cannot",
    "into", "about", "after", "before", "over", "under", "very", "more",
    "less", "some", "any", "all", "our", "out", "his", "her", "its",
    "she", "him", "he", "it", "we", "us", "my", "me", "i", "a", "an",
    "of", "to", "in", "on", "at", "by", "as", "is", "be", "been",
    "being", "do", "does", "did", "done", "so", "if", "or", "because",
    "also", "just", "like", "get", "got", "people", "government", "nigeria",
    "nigerian", "policy", "policies", "rt", "amp", "https", "http", "www",
}

# ---------------------------------------------------------------------------
# 2. File inspection and robust readers
# ---------------------------------------------------------------------------
def _file_path(file_obj: Any) -> str | None:
    """Accept a Gradio filepath string or a file-like object and return a path."""
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        return file_obj.get("path") or file_obj.get("name")
    return getattr(file_obj, "name", None)


def _detect_csv_settings(path: str) -> tuple[str, str]:
    """Select a decodable sample encoding and infer a likely CSV delimiter."""
    with open(path, "rb") as stream:
        raw = stream.read(1_000_000)
    decoded = raw.decode("latin-1")  # Fallback sample for delimiter sniffing.
    chosen_encoding = "latin-1"
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        decoder = codecs.getincrementaldecoder(encoding)()
        try:
            with open(path, "rb") as stream:
                while True:
                    block = stream.read(1_000_000)
                    if not block:
                        break
                    decoder.decode(block)
                decoder.decode(b"", final=True)
            chosen_encoding = encoding
            decoded = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    # latin-1 can decode every byte value, so this is a safe final fallback.
    try:
        delimiter = csv.Sniffer().sniff(decoded[:100_000], delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","
    return chosen_encoding, delimiter


def read_headers(file_obj: Any) -> list[str]:
    """Read only the header row for the column-selection control."""
    path = _file_path(file_obj)
    if not path or not os.path.isfile(path):
        raise ValueError("Please choose a file first.")
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        enc, sep = _detect_csv_settings(path)
        frame = pd.read_csv(path, nrows=0, encoding=enc, sep=sep, engine="python")
    elif suffix == ".xlsx":
        frame = pd.read_excel(path, nrows=0, engine="openpyxl")
    elif suffix == ".xls":
        frame = pd.read_excel(path, nrows=0, engine="xlrd")
    else:
        raise ValueError("Unsupported file type. Upload CSV, XLSX or XLS.")
    return [str(c) for c in frame.columns]


def detect_column(columns: Iterable[Any], candidates: list[str]) -> str | None:
    """Return a matching column name using case/space-insensitive matching."""
    column_list = [str(c) for c in columns]
    normalised = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in column_list}
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalised:
            return normalised[key]
    # Fallback for common headers such as "Tweet Text (English)".
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        for normal_key, original in normalised.items():
            if key and key in normal_key:
                return original
    return None


def _iter_uploaded_frames(path: str):
    """Yield CSV chunks; Excel files are read as one DataFrame."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        encoding, separator = _detect_csv_settings(path)
        reader = pd.read_csv(
            path,
            encoding=encoding,
            sep=separator,
            engine="python",
            chunksize=CHUNK_SIZE,
            on_bad_lines="error",
        )
        for chunk in reader:
            yield chunk
    elif suffix == ".xlsx":
        yield pd.read_excel(path, engine="openpyxl")
    elif suffix == ".xls":
        yield pd.read_excel(path, engine="xlrd")
    else:
        raise ValueError("Unsupported file type. Upload CSV, XLSX or XLS.")

# ---------------------------------------------------------------------------
# 3. Text preparation and VADER classification
# ---------------------------------------------------------------------------
def prepare_text(value: Any) -> str:
    """Remove URL strings and excess whitespace; retain VADER sentiment cues."""
    if pd.isna(value):
        return ""
    text = str(value)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    # Intentionally retain punctuation, capitalisation, emojis, hashtags and negation.
    return text


def classify_compound(compound_score: float) -> str:
    """Apply the standard VADER compound thresholds."""
    if compound_score >= 0.05:
        return "Positive"
    if compound_score <= -0.05:
        return "Negative"
    return "Neutral"


def score_text(text: str, analyzer: SentimentIntensityAnalyzer | None = None) -> tuple[float, str]:
    """Return a VADER compound score and its three-class sentiment label."""
    engine = analyzer or SentimentIntensityAnalyzer()
    score = float(engine.polarity_scores(text)["compound"])
    return score, classify_compound(score)

# ---------------------------------------------------------------------------
# 4. Dataset processing (chunked for CSV)
# ---------------------------------------------------------------------------
def process_dataframe(
    frame: pd.DataFrame,
    text_column: str,
    source_label: str = "Uploaded dataset",
    progress: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare and score an in-memory DataFrame. Empty text rows are excluded."""
    if frame is None or frame.empty:
        raise ValueError("The uploaded dataset is empty.")
    if text_column not in frame.columns:
        raise ValueError(f"Text column '{text_column}' was not found in the dataset.")

    analyzer = SentimentIntensityAnalyzer()
    output_parts: list[pd.DataFrame] = []
    blank_count = 0
    total = len(frame)
    step = max(1, min(5_000, total))

    for start in range(0, total, step):
        part = frame.iloc[start:start + step].copy()
        cleaned = part[text_column].map(prepare_text)
        valid_mask = cleaned.ne("")
        blank_count += int((~valid_mask).sum())
        part = part.loc[valid_mask].copy()
        cleaned = cleaned.loc[valid_mask]
        if not part.empty:
            scores = cleaned.map(lambda value: float(analyzer.polarity_scores(value)["compound"]))
            part["compound_score"] = scores.astype(float)
            part["sentiment"] = part["compound_score"].map(classify_compound)
            output_parts.append(part)
        if progress is not None:
            progress(min(0.95, (start + len(frame.iloc[start:start + step])) / max(total, 1) * 0.95),
                     desc=f"Scoring records {min(start + step, total):,} of {total:,}")

    if not output_parts:
        raise ValueError("No usable text records were found in the selected column.")
    result = pd.concat(output_parts, ignore_index=True)
    date_col = detect_column(result.columns, DATE_COLUMN_CANDIDATES)
    category_col = detect_column(result.columns, CATEGORY_COLUMN_CANDIDATES)
    metadata = {
        "source": source_label,
        "text_column": text_column,
        "input_rows": int(total),
        "analysed_rows": int(len(result)),
        "empty_rows_skipped": int(blank_count),
        "date_column": date_col,
        "category_column": category_col,
    }
    if progress is not None:
        progress(1.0, desc="Analysis complete")
    return result, metadata


def process_uploaded_file(file_obj: Any, selected_column: str = "Auto-detect", progress: Any = None):
    """Read and score an uploaded file. CSV is scored in chunks and staged to disk."""
    path = _file_path(file_obj)
    if not path or not os.path.isfile(path):
        raise ValueError("The uploaded file is no longer available. Upload it again.")
    source_name = Path(path).name
    suffix = Path(path).suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise ValueError("Upload a .csv, .xlsx or .xls file.")

    chosen_column = None if selected_column == "Auto-detect" else selected_column
    total_input_rows = 0
    blank_count = 0
    analyzer = SentimentIntensityAnalyzer()
    parts: list[pd.DataFrame] = []  # Used only for Excel, which pandas reads as one sheet.
    staged_csv_path: str | None = None
    wrote_header = False
    if progress is not None:
        progress(0.02, desc="Reading and validating the dataset")

    try:
        if suffix == ".csv":
            enc, sep = _detect_csv_settings(path)
            reader = pd.read_csv(path, encoding=enc, sep=sep, engine="python",
                                 chunksize=CHUNK_SIZE, on_bad_lines="error")
            iterator = iter(reader)
            try:
                first_chunk = next(iterator)
            except StopIteration:
                raise ValueError("The uploaded CSV contains a header but no data rows.")
            if chosen_column is None:
                chosen_column = detect_column(first_chunk.columns, TEXT_COLUMN_CANDIDATES)
            if chosen_column is None:
                raise ValueError("Could not identify the tweet-text column. Select the correct column in the dropdown.")

            # Each scored chunk is written immediately to a temporary file. This avoids
            # holding every chunk and a second concatenated copy in RAM at the same time.
            stage_file = tempfile.NamedTemporaryFile(prefix="policypulse_stage_", suffix=".csv", delete=False)
            staged_csv_path = stage_file.name
            stage_file.close()
            for index, chunk in enumerate(itertools.chain((first_chunk,), iterator), start=1):
                if chosen_column not in chunk.columns:
                    raise ValueError(f"Text column '{chosen_column}' is missing from a CSV chunk.")
                total_input_rows += len(chunk)
                cleaned = chunk[chosen_column].map(prepare_text)
                valid_mask = cleaned.ne("")
                blank_count += int((~valid_mask).sum())
                valid = chunk.loc[valid_mask].copy()
                if not valid.empty:
                    valid["compound_score"] = cleaned.loc[valid_mask].map(
                        lambda value: float(analyzer.polarity_scores(value)["compound"])
                    ).astype(float)
                    valid["sentiment"] = valid["compound_score"].map(classify_compound)
                    valid.to_csv(staged_csv_path, mode="a", header=not wrote_header,
                                 index=False, encoding="utf-8-sig" if not wrote_header else "utf-8")
                    wrote_header = True
                if progress is not None:
                    progress(None, desc=f"Scored {total_input_rows:,} rows across {index:,} batches")
            if not wrote_header:
                raise ValueError("No usable text records were found in the selected column.")
            result = pd.read_csv(staged_csv_path, low_memory=False)
        else:
            frame = next(_iter_uploaded_frames(path))
            if chosen_column is None:
                chosen_column = detect_column(frame.columns, TEXT_COLUMN_CANDIDATES)
            if chosen_column is None:
                raise ValueError("Could not identify the tweet-text column. Select the correct column in the dropdown.")
            total_input_rows = len(frame)
            cleaned = frame[chosen_column].map(prepare_text)
            valid_mask = cleaned.ne("")
            blank_count = int((~valid_mask).sum())
            valid = frame.loc[valid_mask].copy()
            if not valid.empty:
                valid["compound_score"] = cleaned.loc[valid_mask].map(
                    lambda value: float(analyzer.polarity_scores(value)["compound"])
                ).astype(float)
                valid["sentiment"] = valid["compound_score"].map(classify_compound)
                parts.append(valid)
            if not parts:
                raise ValueError("No usable text records were found in the selected column.")
            result = pd.concat(parts, ignore_index=True)

        metadata = {
            "source": source_name,
            "text_column": chosen_column,
            "input_rows": int(total_input_rows),
            "analysed_rows": int(len(result)),
            "empty_rows_skipped": int(blank_count),
            "date_column": detect_column(result.columns, DATE_COLUMN_CANDIDATES),
            "category_column": detect_column(result.columns, CATEGORY_COLUMN_CANDIDATES),
            "file_size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
            "file_type": suffix,
        }
        if progress is not None:
            progress(1.0, desc="Sentiment analysis complete")
        return result, metadata
    finally:
        if staged_csv_path and os.path.exists(staged_csv_path):
            try:
                os.remove(staged_csv_path)
            except OSError:
                pass

# ---------------------------------------------------------------------------
# 5. Dashboard metrics, trends and frequent terms
# ---------------------------------------------------------------------------
def build_metrics(data: pd.DataFrame) -> dict[str, Any]:
    counts = data["sentiment"].value_counts().reindex(["Positive", "Negative", "Neutral"], fill_value=0)
    total = int(counts.sum())
    percentages = {key: (float(value) / total * 100 if total else 0.0) for key, value in counts.items()}
    return {"total": total, "counts": counts.to_dict(), "percentages": percentages}


def frequent_terms(data: pd.DataFrame, text_column: str, limit: int = 12) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for raw in data[text_column].dropna().astype(str):
        text = prepare_text(raw).lower()
        tokens = re.findall(r"#?[\w']+", text, flags=re.UNICODE)
        for token in tokens:
            token = token.lstrip("#'")
            if len(token) >= 3 and token not in STOP_WORDS and not token.isdigit():
                counter[token] += 1
    return counter.most_common(limit)


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font={"color": "#CBD5E1", "size": 14})
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin={"l": 20, "r": 20, "t": 30, "b": 20})
    return fig


def build_charts(data: pd.DataFrame, metadata: dict[str, Any]):
    metrics = build_metrics(data)
    sentiment_frame = pd.DataFrame({"Sentiment": list(metrics["counts"].keys()),
                                    "Posts": list(metrics["counts"].values())})
    distribution = px.pie(
        sentiment_frame, names="Sentiment", values="Posts", hole=0.62,
        color="Sentiment", color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment distribution",
    )
    distribution.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", font_color="#E5EDF5",
                               legend_title_text="", margin={"l": 10, "r": 10, "t": 55, "b": 10})

    date_col = metadata.get("date_column")
    if date_col and date_col in data.columns:
        dates = pd.to_datetime(data[date_col], errors="coerce", utc=True).dt.date
        timeline_data = data.assign(_analysis_date=dates).dropna(subset=["_analysis_date"])
        if not timeline_data.empty:
            timeline_data = timeline_data.groupby(["_analysis_date", "sentiment"]).size().reset_index(name="Posts")
            timeline = px.line(timeline_data, x="_analysis_date", y="Posts", color="sentiment",
                               markers=True, color_discrete_map=SENTIMENT_COLORS,
                               title="Sentiment over time (from uploaded dates)")
            timeline.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(0,0,0,0)", font_color="#E5EDF5",
                                   legend_title_text="", margin={"l": 20, "r": 15, "t": 55, "b": 25})
        else:
            timeline = _empty_figure("Date field found, but no dates could be parsed.")
    else:
        timeline = _empty_figure("No date column detected. Add a date/timestamp column to view a timeline.")

    category_col = metadata.get("category_column")
    if category_col and category_col in data.columns:
        category_data = data[[category_col, "sentiment"]].copy()
        category_data[category_col] = category_data[category_col].fillna("Unspecified").astype(str)
        top_categories = category_data[category_col].value_counts().head(12).index
        category_data = category_data[category_data[category_col].isin(top_categories)]
        category_data = category_data.groupby([category_col, "sentiment"]).size().reset_index(name="Posts")
        category_fig = px.bar(category_data, x=category_col, y="Posts", color="sentiment",
                              barmode="group", color_discrete_map=SENTIMENT_COLORS,
                              title="Sentiment by policy/category field")
        category_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(0,0,0,0)", font_color="#E5EDF5",
                                   legend_title_text="", margin={"l": 20, "r": 15, "t": 55, "b": 70},
                                   xaxis_tickangle=-25)
    else:
        category_fig = _empty_figure("No policy/category column detected. The app will not invent policy categories.")
    return distribution, timeline, category_fig


def build_kpi_html(data: pd.DataFrame, metadata: dict[str, Any]) -> str:
    metrics = build_metrics(data)
    cards = [
        ("Posts analysed", f"{metrics['total']:,}", "#E2E8F0"),
        ("Positive", f"{metrics['percentages']['Positive']:.1f}%", SENTIMENT_COLORS["Positive"]),
        ("Negative", f"{metrics['percentages']['Negative']:.1f}%", SENTIMENT_COLORS["Negative"]),
        ("Neutral", f"{metrics['percentages']['Neutral']:.1f}%", SENTIMENT_COLORS["Neutral"]),
    ]
    card_html = "".join(
        f'<div class="kpi-card"><div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value" style="color:{color}">{value}</div></div>'
        for label, value, color in cards
    )
    source = html.escape(str(metadata.get("source", "Dataset")))
    note = (f"Source: {source} · {metadata.get('input_rows', metrics['total']):,} input rows · "
            f"{metadata.get('empty_rows_skipped', 0):,} blank text rows skipped")
    if metadata.get("synthetic"):
        note = "SYNTHETIC DEMONSTRATION DATA — not real X/Twitter posts and not evidence of public opinion."
    return f'<div class="kpi-grid">{card_html}</div><div class="dataset-note">{html.escape(note)}</div>'


def build_topics_html(data: pd.DataFrame, metadata: dict[str, Any]) -> str:
    terms = frequent_terms(data, metadata["text_column"])
    if not terms:
        return '<div class="info-box">No frequent terms were found after basic filtering.</div>'
    items = "".join(
        f'<div class="term-row"><span>{html.escape(term)}</span><b>{count:,}</b></div>'
        for term, count in terms
    )
    return ('<div class="info-box"><b>Frequent terms in this uploaded dataset</b>'
            '<p>These are simple word-frequency indicators, not semantic topic modelling or live X trends.</p>'
            f'<div class="terms-list">{items}</div></div>')

# ---------------------------------------------------------------------------
# 6. Search, single-post analysis and synthetic demonstration
# ---------------------------------------------------------------------------
def visible_columns(data: pd.DataFrame, metadata: dict[str, Any]) -> list[str]:
    preferred = [metadata.get("text_column"), metadata.get("date_column"),
                 metadata.get("category_column"), "compound_score", "sentiment"]
    chosen = [c for c in preferred if c and c in data.columns]
    return list(dict.fromkeys(chosen))


def search_records(state: dict[str, Any] | None, query: str, sentiment: str):
    if not state or "data" not in state:
        return pd.DataFrame(), "Upload and analyse a dataset first."
    data = state["data"]
    metadata = state["metadata"]
    filtered = data
    if sentiment and sentiment != "All sentiments":
        filtered = filtered[filtered["sentiment"] == sentiment]
    query = (query or "").strip()
    text_col = metadata["text_column"]
    if query:
        filtered = filtered[filtered[text_col].astype(str).str.contains(re.escape(query), case=False, na=False)]
    count = len(filtered)
    table = filtered[visible_columns(filtered, metadata)].head(TABLE_PREVIEW_LIMIT)
    message = f"{count:,} matching records. Showing up to {TABLE_PREVIEW_LIMIT:,} rows in the table."
    return table, message


def analyse_single_post(text: str):
    cleaned = prepare_text(text)
    if not cleaned:
        return "Enter a post before running sentiment analysis.", ""
    score, sentiment = score_text(cleaned)
    colour = SENTIMENT_COLORS[sentiment]
    result = (f'<div class="single-result"><span class="sentiment-pill" style="background:{colour}">'
              f'{sentiment}</span><span class="score-value">Compound score: {score:.4f}</span></div>')
    note = "VADER uses lexical and rule-based cues. This is an automated estimate, not a definitive interpretation."
    return result, note


def create_synthetic_demo() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create clearly labelled fictional posts for demonstration only."""
    demo_rows = [
        ("I appreciate the effort to improve public hospitals; access is getting better.", "Healthcare", "2026-01-03"),
        ("The new transport plan could make commuting easier if it is implemented well.", "Transport", "2026-01-04"),
        ("This tax change is unfair and puts too much pressure on small businesses.", "Taxation", "2026-01-05"),
        ("I am disappointed with the electricity situation. Power supply is still terrible.", "Electricity", "2026-01-06"),
        ("The education announcement is interesting, but I need to see the details.", "Education", "2026-01-07"),
        ("The fuel price is too high. This decision is hurting ordinary families.", "Energy", "2026-01-08"),
        ("More transparent communication about the policy would be helpful.", "Governance", "2026-01-09"),
        ("I support investment in rural roads because communities need better access.", "Transport", "2026-01-10"),
        ("The policy has not solved the problem and the results are disappointing.", "Governance", "2026-01-11"),
        ("The government should explain how the programme will be funded.", "Economy", "2026-01-12"),
        ("Good progress on the school renovation project. Students deserve safe classrooms.", "Education", "2026-01-13"),
        ("I cannot tell whether this policy will work yet; more information is needed.", "Economy", "2026-01-14"),
    ]
    frame = pd.DataFrame(demo_rows, columns=["tweet_text", "policy_category", "created_at"])
    scored, meta = process_dataframe(frame, "tweet_text", "Synthetic demonstration data")
    meta["synthetic"] = True
    meta["date_column"] = "created_at"
    meta["category_column"] = "policy_category"
    return scored, meta


def render_dashboard(data: pd.DataFrame, metadata: dict[str, Any]):
    fig1, fig2, fig3 = build_charts(data, metadata)
    table = data[visible_columns(data, metadata)].head(TABLE_PREVIEW_LIMIT)
    status = (f"Analysis complete: {len(data):,} posts scored using VADER. "
              f"Text field: '{metadata['text_column']}'.")
    if metadata.get("synthetic"):
        status = "Synthetic demo loaded. These fictional examples must not be cited as real public opinion."
    return status, {"data": data, "metadata": metadata}, build_kpi_html(data, metadata), fig1, fig2, fig3, build_topics_html(data, metadata), table


def handle_upload(file_obj: Any, selected_column: str, progress=gr.Progress(track_tqdm=False)):
    try:
        data, metadata = process_uploaded_file(file_obj, selected_column, progress)
        return render_dashboard(data, metadata)
    except Exception as exc:
        empty = _empty_figure("Upload a valid dataset to display analysis.")
        error = f"Analysis could not be completed: {html.escape(str(exc))}"
        return error, None, "", empty, empty, empty, "", pd.DataFrame()


def load_demo(progress=gr.Progress(track_tqdm=False)):
    try:
        data, metadata = create_synthetic_demo()
        return render_dashboard(data, metadata)
    except Exception as exc:
        empty = _empty_figure("Synthetic demo could not load.")
        return f"Demo error: {html.escape(str(exc))}", None, "", empty, empty, empty, "", pd.DataFrame()

# ---------------------------------------------------------------------------
# 7. Optional Gemini assistant (separate from VADER classification)
# ---------------------------------------------------------------------------
def _gemini_interaction(api_key: str, model: str, prompt: str) -> str:
    """Call Gemini Interactions API without storing the interaction server-side."""
    endpoint = "https://generativelanguage.googleapis.com/v1beta/interactions"
    payload = {
        "model": (model or GEMINI_DEFAULT_MODEL).strip(),
        "input": prompt,
        "store": False,
        "system_instruction": (
            "You are the PolicyPulse dashboard assistant. Explain dataset summaries carefully. "
            "Do not invent facts or claim the dataset represents all Nigerians. Distinguish correlation "
            "from causation. VADER labels are automated estimates and may misread Nigerian Pidgin, "
            "sarcasm, slang, irony and context. Never claim access to live X/Twitter data."
        ),
    }
    response = requests.post(endpoint, headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                             json=payload, timeout=60)
    response.raise_for_status()
    body = response.json()
    for step in reversed(body.get("steps", [])):
        if step.get("type") == "model_output":
            content = step.get("content", [])
            text_parts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("text")]
            if text_parts:
                return "\n".join(text_parts).strip()
    # Defensive fallback for response variants.
    if body.get("output_text"):
        return str(body["output_text"])
    raise RuntimeError("Gemini returned no readable text. Check the model name and API access.")


def local_assistant_reply(question: str, state: dict[str, Any] | None) -> str:
    if not state or "data" not in state:
        return ("I can answer general questions about how PolicyPulse works. Upload and analyse a dataset "
                "to ask questions about its sentiment results. Enter a Gemini API key to enable the optional AI assistant.")
    data, meta = state["data"], state["metadata"]
    metrics = build_metrics(data)
    top = ", ".join(f"{name} ({count})" for name, count in frequent_terms(data, meta["text_column"], 5)) or "not available"
    return (
        f"Current dataset summary: {metrics['total']:,} posts analysed. "
        f"Positive: {metrics['percentages']['Positive']:.1f}% ({metrics['counts']['Positive']:,}); "
        f"Negative: {metrics['percentages']['Negative']:.1f}% ({metrics['counts']['Negative']:,}); "
        f"Neutral: {metrics['percentages']['Neutral']:.1f}% ({metrics['counts']['Neutral']:,}). "
        f"Frequent terms include: {top}. These figures describe only the uploaded dataset. "
        "VADER is rule/lexicon based and may misread sarcasm, Nigerian Pidgin, slang or context. "
        "For a natural-language explanation, enter a Gemini API key."
    )


def chat_reply(message: str, history: list[dict[str, Any]] | None, state: dict[str, Any] | None,
               api_key: str, model: str):
    history = list(history or [])
    question = (message or "").strip()
    if not question:
        return history, ""
    try:
        if api_key and api_key.strip():
            context = "No dataset is loaded. Do not make claims about results." 
            if state and "data" in state:
                data, meta = state["data"], state["metadata"]
                m = build_metrics(data)
                context = (
                    f"Uploaded dataset summary only: {m['total']} analysed records; "
                    f"positive {m['percentages']['Positive']:.2f}% ({m['counts']['Positive']}); "
                    f"negative {m['percentages']['Negative']:.2f}% ({m['counts']['Negative']}); "
                    f"neutral {m['percentages']['Neutral']:.2f}% ({m['counts']['Neutral']}); "
                    f"source={meta.get('source')}; synthetic={meta.get('synthetic', False)}. "
                    "This dataset is not necessarily representative of Nigeria."
                )
            recent_turns = []
            for turn in (history or [])[-8:]:
                if isinstance(turn, dict):
                    role = str(turn.get("role", "user"))
                    content = str(turn.get("content", ""))
                else:
                    continue
                if content:
                    recent_turns.append(f"{role}: {content}")
            conversation = "\n".join(recent_turns)
            prompt = context + "\nRecent conversation:\n" + conversation + "\nUser question: " + question
            answer = _gemini_interaction(api_key.strip(), model, prompt)
        else:
            answer = local_assistant_reply(question, state)
    except requests.HTTPError as exc:
        answer = f"Gemini request failed (HTTP {exc.response.status_code if exc.response is not None else 'unknown'}). Verify the API key, model access and quota."
    except Exception as exc:
        answer = f"Assistant unavailable: {exc}"
    history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
    return history[-20:], ""

# ---------------------------------------------------------------------------
# 8. Export analysed records
# ---------------------------------------------------------------------------
def export_results(state: dict[str, Any] | None):
    if not state or "data" not in state:
        raise gr.Error("Analyse a dataset before exporting results.")
    data = state["data"]
    tmp = tempfile.NamedTemporaryFile(prefix="policypulse_results_", suffix=".csv", delete=False)
    tmp_path = tmp.name
    tmp.close()
    data.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    return tmp_path

# ---------------------------------------------------------------------------
# 9. Interface styling and Gradio application
# ---------------------------------------------------------------------------
CUSTOM_CSS = r"""
:root { color-scheme: dark; }
.gradio-container { background: #081321 !important; color: #E7EEF7 !important; max-width: 1500px !important; }
body, .dark { background: #081321 !important; }
footer { display: none !important; }
#pp-header { padding: 24px 26px; border: 1px solid #1B3449; border-radius: 18px; background: linear-gradient(120deg,#10243A,#0B1B2C 65%,#10343B); margin-bottom: 16px; }
#pp-header h1 { color: #F2F7FC; font-size: 30px; margin: 0 0 4px 0; letter-spacing: .4px; }
#pp-header p { color: #A9BECE; margin: 0; }
.kpi-grid { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 12px; margin: 8px 0 12px; }
.kpi-card { background: #13283B; border: 1px solid #24445A; border-radius: 14px; padding: 16px; min-height: 82px; }
.kpi-label { color: #AABCCB; font-size: 12px; text-transform: uppercase; letter-spacing: .7px; }
.kpi-value { font-size: 27px; font-weight: 750; margin-top: 7px; }
.dataset-note { color: #96AABD; font-size: 12px; margin: 4px 0 10px; }
.info-box { background: #102438; border: 1px solid #24445A; border-radius: 14px; padding: 16px; color: #DFEAF3; }
.info-box p { color: #9FB5C7; font-size: 13px; }
.terms-list { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 7px; }
.term-row { background: #172F44; border-radius: 9px; padding: 9px 12px; display: flex; justify-content: space-between; }
.single-result { display:flex; align-items:center; gap:14px; padding:12px; background:#102438; border-radius:12px; }
.sentiment-pill { color:#071321; border-radius:999px; padding:6px 12px; font-weight:800; }
.score-value { color:#E7EEF7; font-weight:650; }
button.primary { background: #0F9D91 !important; border-color: #0F9D91 !important; color:#06191B !important; }
.tab-nav button { color: #BDD0DF !important; }
label, .prose, p, span { color: inherit; }
@media (max-width: 760px) { .kpi-grid { grid-template-columns: repeat(2,minmax(0,1fr)); } .terms-list { grid-template-columns: 1fr; } #pp-header h1 { font-size:24px; } }
"""


def inspect_upload(file_obj: Any):
    try:
        columns = read_headers(file_obj)
        choices = ["Auto-detect"] + columns
        detected = detect_column(columns, TEXT_COLUMN_CANDIDATES)
        value = detected if detected else "Auto-detect"
        return gr.update(choices=choices, value=value), f"File ready. {len(columns)} columns detected."
    except Exception as exc:
        return gr.update(choices=["Auto-detect"], value="Auto-detect"), f"File inspection failed: {exc}"


def build_app():
    theme = gr.themes.Soft(primary_hue="teal", secondary_hue="cyan", neutral_hue="slate")
    with gr.Blocks(title=f"{APP_NAME} | Policy Sentiment Intelligence") as demo:
        analysis_state = gr.State(None)
        gr.HTML(
            "<div id='pp-header'><h1>PolicyPulse</h1>"
            "<p>Policy Sentiment Intelligence Platform · Analyse uploaded public-opinion datasets</p></div>"
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                upload = gr.File(label="Upload dataset (CSV, XLSX or XLS)", file_types=[".csv", ".xlsx", ".xls"], type="filepath")
            with gr.Column(scale=3):
                text_column = gr.Dropdown(label="Tweet / post text column", choices=["Auto-detect"], value="Auto-detect")
            with gr.Column(scale=2):
                inspect_button = gr.Button("Inspect file", variant="secondary")
                analyse_button = gr.Button("Analyse dataset", variant="primary")
            with gr.Column(scale=2):
                demo_button = gr.Button("Load synthetic demo", variant="secondary")
        file_status = gr.Markdown("Choose a file, then inspect its columns.")
        upload.change(inspect_upload, inputs=upload, outputs=[text_column, file_status])
        inspect_button.click(inspect_upload, inputs=upload, outputs=[text_column, file_status])
        status = gr.Markdown("Upload a dataset, inspect its columns, then select **Analyse dataset**.")

        with gr.Tabs():
            with gr.Tab("Executive dashboard"):
                kpis = gr.HTML()
                with gr.Row():
                    distribution_plot = gr.Plot(label="Sentiment distribution")
                    timeline_plot = gr.Plot(label="Sentiment over time")
                with gr.Row():
                    category_plot = gr.Plot(label="Policy/category comparison")
                    terms = gr.HTML()
                preview = gr.Dataframe(label="Analysed records (preview)", interactive=False, wrap=True)
                export_button = gr.Button("Export all analysed records (CSV)", variant="primary")
                export_file = gr.File(label="Download analysed CSV")
                export_button.click(export_results, inputs=analysis_state, outputs=export_file)

            with gr.Tab("Explore records"):
                with gr.Row():
                    search_box = gr.Textbox(label="Search post text", placeholder="Enter a word or phrase")
                    sentiment_filter = gr.Dropdown(
                        label="Sentiment filter",
                        choices=["All sentiments", "Positive", "Negative", "Neutral"],
                        value="All sentiments",
                    )
                    search_button = gr.Button("Search", variant="primary")
                search_status = gr.Markdown("Upload and analyse a dataset to search its records.")
                search_table = gr.Dataframe(label="Matching records (maximum 200 shown)", interactive=False, wrap=True)
                search_button.click(search_records, inputs=[analysis_state, search_box, sentiment_filter],
                                     outputs=[search_table, search_status])
                search_box.submit(search_records, inputs=[analysis_state, search_box, sentiment_filter],
                                  outputs=[search_table, search_status])
                sentiment_filter.change(search_records, inputs=[analysis_state, search_box, sentiment_filter],
                                         outputs=[search_table, search_status])

            with gr.Tab("Single-post analysis"):
                single_text = gr.Textbox(label="Enter one post", lines=5,
                                         placeholder="Paste a single public-policy post here...")
                single_button = gr.Button("Analyse this post", variant="primary")
                single_result = gr.HTML()
                single_note = gr.Markdown()
                single_button.click(analyse_single_post, inputs=single_text, outputs=[single_result, single_note])

            with gr.Tab("AI assistant"):
                gr.Markdown("The optional Gemini assistant is separate from VADER. Without an API key, the assistant returns a basic summary of the loaded dashboard only.")
                chatbot = gr.Chatbot(label="PolicyPulse assistant", height=360)
                with gr.Row():
                    chat_input = gr.Textbox(label="Ask a question", placeholder="Explain the sentiment distribution...", scale=5)
                    chat_send = gr.Button("Send", variant="primary", scale=1)
                with gr.Row():
                    gemini_key = gr.Textbox(label="Optional Gemini API key", type="password", placeholder="Paste your API key here; do not share it")
                    gemini_model = gr.Textbox(label="Gemini model", value=GEMINI_DEFAULT_MODEL)
                chat_inputs = [chat_input, chatbot, analysis_state, gemini_key, gemini_model]
                chat_outputs = [chatbot, chat_input]
                chat_send.click(chat_reply, inputs=chat_inputs, outputs=chat_outputs)
                chat_input.submit(chat_reply, inputs=chat_inputs, outputs=chat_outputs)

            with gr.Tab("Methodology & X integration"):
                gr.Markdown(
                    """
### Methodology
- The system analyses a dataset uploaded by the user in CSV, XLSX or XLS format. For Excel workbooks, the first worksheet is read.
- VADER is a lexicon- and rule-based sentiment analyser; it is not trained as part of this application.
- The VADER compound score is classified as **Positive (≥ 0.05)**, **Negative (≤ −0.05)** or **Neutral (> −0.05 and < 0.05)**.
- URL strings and excess whitespace are removed. Punctuation, capitalisation, emojis, hashtags and negation are retained because they can carry sentiment information.
- Rows with missing or empty post text are skipped and counted. The system does not claim to remove duplicate posts.

### Scope and limitations
- There is **no direct X/Twitter API connection** in this version. It does not continuously collect live posts.
- “Real-time” in this prototype means rapid processing after an uploaded dataset is submitted.
- Frequent terms are simple word-frequency indicators from the uploaded file, not live X trends or semantic topic modelling.
- Results describe only the uploaded dataset. They do not automatically represent all Nigerians or public opinion nationwide.
- VADER may misinterpret Nigerian Pidgin, local slang, sarcasm, irony and context. Human review is recommended for high-stakes interpretation.
- Data are held in the running application session; this build does not provide a permanent database or long-term record storage.
- The synthetic demonstration dataset is fictional and must not be reported as actual tweets or research findings.

### Research reference
Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *Proceedings of the International AAAI Conference on Web and Social Media, 8*(1), 216–225. https://doi.org/10.1609/icwsm.v8i1.14550
                    """
                )

        dashboard_outputs = [status, analysis_state, kpis, distribution_plot, timeline_plot,
                             category_plot, terms, preview]
        analyse_button.click(handle_upload, inputs=[upload, text_column], outputs=dashboard_outputs)
        demo_button.click(load_demo, inputs=None, outputs=dashboard_outputs)
    return demo


demo = build_app()

if __name__ == "__main__":
    # Colab can show a public share link; Hugging Face Spaces will manage its own server.
    demo.queue(default_concurrency_limit=2).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")), show_error=True, max_file_size="200mb", theme=gr.themes.Soft(primary_hue="teal", secondary_hue="cyan", neutral_hue="slate"), css=CUSTOM_CSS)
