import os
import re
import json
import glob
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# -----------------------------
# Config
# -----------------------------
# Use absolute path relative to this script file to avoid CWD issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "yt_blocks")

YOUTUBE_OVERRIDES_BY_ID = {
    "NGj6gDZoRTo": "https://www.youtube.com/watch?v=NGj6gDZoRTo",
    "MaBgmPcxpuE": "https://www.youtube.com/watch?v=MaBgmPcxpuE",
}

VIDEO_LINKS_JSON = os.path.join(BASE_DIR, "data", "video_links.json")

MAX_MATCHES_PER_VIDEO = 2


# -----------------------------
# Helpers
# -----------------------------
def normalize_title_key(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("\\", "/")
    s = os.path.basename(s)
    s = re.sub(r"\.(mp3|wav|m4a|flac|aac)$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\[[A-Za-z0-9_-]{6,}\]", "", s)  # remove [YouTubeId] etc.
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_youtube_id(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip()

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", t):
        return t

    m = re.search(r"v=([A-Za-z0-9_-]{11})", t)
    if m:
        return m.group(1)

    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", t)
    if m:
        return m.group(1)

    m = re.search(r"/embed/([A-Za-z0-9_-]{11})", t)
    if m:
        return m.group(1)

    m = re.search(r"\[([A-Za-z0-9_-]{11})\]", t)
    if m:
        return m.group(1)

    return None


def extract_year(text: str) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"(20\d{2})", text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def youtube_embed_html(url: str, start_seconds: int, autoplay: bool = True) -> str:
    ytid = extract_youtube_id(url) or ""
    if not ytid:
        return f"""
        <div style="padding:12px;border:1px solid #333;border-radius:10px;">
          <div style="font-size:14px;margin-bottom:8px;">No valid YouTube ID found.</div>
          <a href="{url}" target="_blank">{url}</a>
        </div>
        """

    embed = (
        f"https://www.youtube.com/embed/{ytid}"
        f"?start={int(start_seconds)}&autoplay={1 if autoplay else 0}"
        f"&mute=0&playsinline=1&rel=0&modestbranding=1"
    )

    return f"""
    <div>
      <iframe
        width="100%"
        height="420"
        src="{embed}"
        title="YouTube video player"
        frameborder="0"
        allow="autoplay; encrypted-media; picture-in-picture"
        allowfullscreen
      ></iframe>
      <div style="font-size:12px;opacity:0.75;margin-top:6px;">
        If autoplay is blocked, click the play button in the video.
      </div>
    </div>
    """


@dataclass
class Match:
    score: float
    start: float
    end: float
    text: str
    video_title: str
    year: Optional[int]
    url: Optional[str]
    audio_file: Optional[str]


@dataclass
class VideoResult:
    video_key: str
    title: str
    year: Optional[int]
    url: Optional[str]
    best_score: float
    matches: List[Match]


# -----------------------------
# Loading (Optimized)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_video_links_map() -> Dict[str, str]:
    out: Dict[str, str] = {}
    if os.path.exists(VIDEO_LINKS_JSON):
        try:
            raw = json.loads(open(VIDEO_LINKS_JSON, "r", encoding="utf-8").read())
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if isinstance(k, str) and isinstance(v, str):
                        out[normalize_title_key(k)] = v
        except Exception:
            pass
    return out


def pick_semantic_index_file() -> str:
    # Try the standard data directory first
    candidates = glob.glob(os.path.join(DATA_DIR, "*.semantic.pkl"))
    
    # Fallback: check the current directory (GitHub root upload case)
    if not candidates:
        candidates = glob.glob(os.path.join(BASE_DIR, "*.semantic.pkl"))
    
    if not candidates:
        raise FileNotFoundError(f"No .semantic.pkl found in {DATA_DIR} or {BASE_DIR}")

    # prefer final
    candidates_sorted = sorted(
        candidates,
        key=lambda p: (("final" not in os.path.basename(p).lower()), os.path.basename(p).lower()),
    )
    return candidates_sorted[0]


@st.cache_resource(show_spinner=True)
def load_index_data(pkl_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]], str]:
    """
    Loads the pickle and prepares the matrix in one go.
    Cached as a resource to avoid hashing large objects on every rerun.
    """
    with open(pkl_path, "rb") as f:
        index_obj = pickle.load(f)
    
    if not isinstance(index_obj, dict):
        raise ValueError("Semantic index pkl is not a dict.")

    model_name = (
        index_obj.get("model_name")
        if isinstance(index_obj.get("model_name"), str)
        else index_obj.get("model")
        if isinstance(index_obj.get("model"), str)
        else "sentence-transformers/all-MiniLM-L6-v2"
    )

    blocks = _get_first_present(index_obj, ["blocks", "items", "data"])
    if not isinstance(blocks, list):
        raise ValueError("Semantic index missing blocks list.")

    emb = _get_first_present(index_obj, ["embeddings", "X", "matrix"])
    if emb is None:
        raise ValueError("Semantic index missing embeddings matrix.")

    E = np.array(emb, dtype=np.float32)
    if E.ndim != 2:
        raise ValueError("Embeddings matrix must be 2D.")

    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    E = E / norms
    
    return E, blocks, model_name


@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _get_first_present(obj: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in obj:
            return obj[k]
    return None


# -----------------------------
# Metadata resolution & Search
# -----------------------------
# (Reusing existing helper functions for resolution, omitted for brevity if unchanged, 
# but ensuring search_index uses the new loader)

def resolve_video_title(block: Dict[str, Any]) -> str:
    for key in ["video_title", "title", "file_title"]:
        v = block.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    af = block.get("audio_file")
    if isinstance(af, str) and af.strip():
        return os.path.basename(af)
    return "Unknown"


def resolve_year(block: Dict[str, Any]) -> Optional[int]:
    y = block.get("year")
    if isinstance(y, int):
        return y
    t = resolve_video_title(block)
    y2 = extract_year(t)
    if y2:
        return y2
    af = block.get("audio_file")
    if isinstance(af, str):
        y3 = extract_year(af)
        if y3:
            return y3
    return None


def resolve_video_url(block: Dict[str, Any], links_map: Dict[str, str]) -> Optional[str]:
    # 1) direct block urls
    for key in ["webpage_url", "youtube_url", "url", "video_url"]:
        v = block.get(key)
        if isinstance(v, str) and "youtu" in v:
            return v

    # 2) video_id
    vid = block.get("video_id")
    if isinstance(vid, str):
        if vid in YOUTUBE_OVERRIDES_BY_ID:
            return YOUTUBE_OVERRIDES_BY_ID[vid]
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid):
            return f"https://www.youtube.com/watch?v={vid}"

    # 3) parse from title/audio filename
    audio_file = block.get("audio_file") or ""
    title = resolve_video_title(block)
    for src in [str(audio_file), str(title)]:
        ytid = extract_youtube_id(src)
        if ytid:
            if ytid in YOUTUBE_OVERRIDES_BY_ID:
                return YOUTUBE_OVERRIDES_BY_ID[ytid]
            return f"https://www.youtube.com/watch?v={ytid}"

    # 4) mapping file
    if isinstance(audio_file, str) and audio_file:
        k = normalize_title_key(audio_file)
        if k in links_map:
            return links_map[k]
    if isinstance(title, str) and title:
        k = normalize_title_key(title)
        if k in links_map:
            return links_map[k]

    return None


def block_duration(block: Dict[str, Any]) -> float:
    try:
        s = float(block.get("start", 0.0))
        e = float(block.get("end", 0.0))
        return e - s
    except Exception:
        return 0.0


@st.cache_data(show_spinner=False)
def search_index(
    pkl_path: str,
    query: str,
    mandatory_keyword: str,
    min_duration_s: int,
    year_filter: str,
) -> List[VideoResult]:
    # Use the optimized loader
    E, blocks, model_name = load_index_data(pkl_path)
    embedder = load_embedder(model_name)

    q = (query or "").strip()
    if not q:
        return []

    qv = embedder.encode([q], normalize_embeddings=True)
    qv = np.array(qv, dtype=np.float32)[0]
    sims = E @ qv

    kw = (mandatory_keyword or "").strip().lower()
    links_map = load_video_links_map()

    candidates: List[Tuple[int, float]] = []
    for i, b in enumerate(blocks):
        if min_duration_s and block_duration(b) < float(min_duration_s):
            continue
        if kw:
            txt = str(b.get("text", "")).lower()
            if kw not in txt:
                continue
        by = resolve_year(b)
        if year_filter != "All":
            try:
                y_int = int(year_filter)
                if by != y_int:
                    continue
            except Exception:
                pass
        candidates.append((i, float(sims[i])))

    candidates.sort(key=lambda t: t[1], reverse=True)

    per_video: Dict[str, List[Match]] = {}
    meta: Dict[str, Tuple[str, Optional[int], Optional[str]]] = {}

    for idx, score in candidates:
        b = blocks[idx]
        title = resolve_video_title(b)
        year = resolve_year(b)
        url = resolve_video_url(b, links_map)

        af = b.get("audio_file") or ""
        video_key = normalize_title_key(af) if isinstance(af, str) and af.strip() else normalize_title_key(title)

        if video_key not in meta:
            meta[video_key] = (title, year, url)

        if video_key not in per_video:
            per_video[video_key] = []
        if len(per_video[video_key]) >= MAX_MATCHES_PER_VIDEO:
            continue

        try:
            s = float(b.get("start", 0.0))
            e = float(b.get("end", 0.0))
        except Exception:
            s, e = 0.0, 0.0

        per_video[video_key].append(
            Match(
                score=score,
                start=s,
                end=e,
                text=str(b.get("text", "")).strip(),
                video_title=title,
                year=year,
                url=url,
                audio_file=str(af) if isinstance(af, str) else None,
            )
        )

    video_results: List[VideoResult] = []
    for video_key, matches in per_video.items():
        title, year, url = meta.get(video_key, ("Unknown", None, None))
        best = max(m.score for m in matches) if matches else -1.0
        video_results.append(
            VideoResult(
                video_key=video_key,
                title=title,
                year=year,
                url=url,
                best_score=best,
                matches=matches,
            )
        )

    video_results.sort(key=lambda vr: vr.best_score, reverse=True)
    return video_results


# -----------------------------
# UI
# -----------------------------
def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Main container styling */
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }

        /* Headings */
        h1, h2, h3 {
            font-weight: 600 !important;
            letter-spacing: -0.5px;
        }
        
        /* Inputs */
        .stTextInput > div > div > input {
            background-color: #262730;
            color: #ffffff;
            border: 1px solid #4a4a4a;
            border-radius: 8px;
            padding: 10px 12px;
        }
        .stTextInput > div > div > input:focus {
            border-color: #ff4b4b;
            box-shadow: 0 0 0 1px #ff4b4b;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            padding: 0.5rem 1rem;
            transition: all 0.2s;
        }
        
        /* Primary Search Button */
        div[data-testid="stSidebar"] button[kind="primary"] {
            width: 100%;
            background-color: #ff4b4b;
            border: none;
            color: white;
            padding: 12px;
            margin-top: 10px;
        }
        div[data-testid="stSidebar"] button[kind="primary"]:hover {
            background-color: #ff3333;
            box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
        }

        /* Video Result Cards (simulated via markdown) */
        hr {
            margin: 2em 0;
            opacity: 0.2;
        }

        /* Expander */
        .streamlit-expanderHeader {
            font-weight: 600;
            background-color: #1f1f1f;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Kapil Gupta QnA — Semantic Search", layout="wide", page_icon="🌲")
    inject_custom_css()

    try:
        pkl_path = pick_semantic_index_file()
    except Exception as e:
        st.error(f"Cannot find semantic index: {e}")
        st.stop()

    # Load data efficiently
    _, blocks, _ = load_index_data(pkl_path)
    
    # Calculate years
    years = sorted({y for y in (resolve_year(b) for b in blocks) if isinstance(y, int)})
    year_options = ["All"] + [str(y) for y in years]

    # --- URL PARAM HANDLING ---
    # Check if 'q' is in the URL query params
    # streamlit >= 1.30 uses st.query_params as a dict-like object
    query_params = st.query_params
    url_q = query_params.get("q", "")
    
    # Initialize session state from URL if present and not already set
    if "q" not in st.session_state:
        st.session_state["q"] = url_q
    
    # If we have a URL query but haven't run yet, mark it to run
    if url_q and "results" not in st.session_state:
        auto_run = True
    else:
        auto_run = False

    with st.sidebar:
        st.title("🔍 Search")
        st.caption("Explore the wisdom of Kapil Gupta.")
        
        q = st.text_input("Question", value=st.session_state.get("q", ""), placeholder="Enter your question")
        
        with st.expander("Advanced Options"):
            mandatory_keyword = st.text_input(
                "Mandatory keyword",
                value=st.session_state.get("mandatory_keyword", ""),
                placeholder="e.g. anxiety",
                help="Only show results containing this exact word."
            )

            min_duration_s = st.number_input(
                "Min duration (seconds)",
                min_value=0,
                max_value=60 * 60,
                value=int(st.session_state.get("min_duration_s", 0)),
                step=5,
            )

            year_filter = st.selectbox(
                "Year Filter",
                options=year_options,
                index=year_options.index(st.session_state.get("year_filter", "All"))
                if st.session_state.get("year_filter", "All") in year_options
                else 0,
            )

        run = st.button("Search", type="primary")

    # Sync UI back to session state
    st.session_state["q"] = q
    st.session_state["mandatory_keyword"] = mandatory_keyword
    st.session_state["min_duration_s"] = int(min_duration_s)
    st.session_state["year_filter"] = year_filter
    
    # Update URL param when searching
    if q:
        st.query_params["q"] = q
    else:
        # clear valid param if empty
        if "q" in st.query_params:
            del st.query_params["q"]

    # Trigger search if button clicked OR auto-run from URL is true
    if run or auto_run:
        if not q.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching transcripts..."):
                results = search_index(
                    pkl_path=pkl_path,
                    query=q,
                    mandatory_keyword=mandatory_keyword,
                    min_duration_s=int(min_duration_s),
                    year_filter=year_filter,
                )
            st.session_state["results"] = results
            st.session_state["video_idx"] = 0
            st.session_state["match_idx"] = 0

    results: List[VideoResult] = st.session_state.get("results", [])
    if not results:
        # Welcome Screen
        st.markdown(
            """
            <div style="text-align: center; padding: 4rem 0;">
                <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">Kapil Gupta QnA</h1>
                <p style="font-size: 1.1rem; opacity: 0.7; margin-bottom: 2rem; max-width: 600px; margin-left: auto; margin-right: auto;">
                    Transcripts from Kapil's Question and Answer sessions. <br>
                    The semantic search attempts to match your question with topics Kapil has addressed.
                </p>
            </div>
            """, unsafe_allow_html=True
        )
        return

    video_idx = int(st.session_state.get("video_idx", 0))
    match_idx = int(st.session_state.get("match_idx", 0))
    video_idx = max(0, min(video_idx, len(results) - 1))
    st.session_state["video_idx"] = video_idx

    vr = results[video_idx]
    match_idx = max(0, min(match_idx, len(vr.matches) - 1))
    st.session_state["match_idx"] = match_idx
    m = vr.matches[match_idx]

    # --- Top Navigation Bar ---
    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if st.button("← Prev Video", use_container_width=True):
            st.session_state["video_idx"] = max(0, video_idx - 1)
            st.session_state["match_idx"] = 0
            st.rerun()
    with c2:
        st.markdown(f"<div style='text-align: center; font-weight: 600; padding-top: 8px;'>Result {video_idx+1} of {len(results)}</div>", unsafe_allow_html=True)
    with c3:
        if st.button("Next Video →", use_container_width=True):
            st.session_state["video_idx"] = min(len(results) - 1, video_idx + 1)
            st.session_state["match_idx"] = 0
            st.rerun()

    st.markdown("---")

    # --- Main Content Area ---
    # Video details
    st.markdown(f"### {vr.title}")
    st.caption(f"Year: {vr.year or 'Unknown'} • Relevance Score: {vr.best_score:.2f}")

    if m.url:
        components.html(
            youtube_embed_html(m.url, start_seconds=int(m.start), autoplay=True),
            height=480,
        )
    else:
        st.warning("Video URL not available.")

    # Match Navigation
    st.markdown("")
    m_col1, m_col2, m_col3 = st.columns([1, 2, 1])
    with m_col1:
         if st.button("Previous Match", key="pm", disabled=(match_idx == 0)):
            st.session_state["match_idx"] = max(0, match_idx - 1)
            st.rerun()
    with m_col2:
        st.markdown(f"<div style='text-align: center; opacity: 0.7; font-size: 0.9em; padding-top: 5px;'>Segment {match_idx+1} / {len(vr.matches)}</div>", unsafe_allow_html=True)
    with m_col3:
        if st.button("Next Match", key="nm", disabled=(match_idx >= len(vr.matches) - 1)):
            st.session_state["match_idx"] = min(len(vr.matches) - 1, match_idx + 1)
            st.rerun()

    # Transcript Box
    st.markdown("#### Transcript Segment")
    
    # Highlight the relevant text if possible? For now just show standard code block but styled
    st.markdown(
        f"""
        <div style="background-color: #262730; padding: 20px; border-radius: 10px; border: 1px solid #444; font-family: monospace; white-space: pre-wrap; line-height: 1.5;">
            {m.text}
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption(f"Time: {int(m.start)}s - {int(m.end)}s")


if __name__ == "__main__":
    main()
