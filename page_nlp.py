"""
Doc Tools — merged NLP Tools + Dashboard + RAG Q&A with citations.
"""
import os
import io
import re
import json
import uuid
from collections import Counter

import streamlit as st
import pandas as pd
import plotly.express as px

import ui
from i18n import t
from ml import load_table_with_notes
from nlp import (clean_text, tokenize, key_phrases, summarize_extractive,
                 summarize_abstractive, translate, fix_grammar,
                 cluster_documents, search_documents, STOP_WORDS)
from llm import ask


# ============================================================
# LAZY LOADERS
# ============================================================
@st.cache_resource(show_spinner=False)
def _load_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        def encode(texts):
            return model.encode(texts, normalize_embeddings=True,
                                 show_progress_bar=False)
        return {"kind": "st", "encode": encode, "dim": 384}
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        def encode(texts):
            vec = TfidfVectorizer(stop_words="english",
                                   ngram_range=(1, 2), sublinear_tf=True)
            X = vec.fit_transform(texts).toarray()
            norms = (X ** 2).sum(axis=1) ** 0.5
            norms[norms == 0] = 1
            return X / norms[:, None]
        return {"kind": "tfidf", "encode": encode, "dim": None}


def _read_plain_text(uploaded):
    try:
        raw = uploaded.getvalue()
    except Exception:
        return None, "Could not read the file."
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc), None
        except UnicodeDecodeError:
            continue
    return None, "Could not decode the file as text."


def _read_pdf(uploaded):
    try:
        from pypdf import PdfReader
    except ImportError:
        return None, "Install `pypdf` to read PDFs."
    try:
        reader = PdfReader(io.BytesIO(uploaded.getvalue()))
        pages = []
        for i, p in enumerate(reader.pages, 1):
            try:
                txt = p.extract_text() or ""
            except Exception:
                txt = ""
            if txt.strip():
                pages.append((i, txt.strip()))
        if not pages:
            return None, "That PDF has no extractable text."
        return pages, None
    except Exception as e:
        return None, f"Could not read PDF: {type(e).__name__}: {e}"


def _read_docx(uploaded):
    try:
        import docx
    except ImportError:
        return None, "Install `python-docx` to read Word files."
    try:
        doc = docx.Document(io.BytesIO(uploaded.getvalue()))
        paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        if not paras:
            return None, "That Word document has no text."
        pages = [(i // 20 + 1, "\n".join(paras[i:i + 20]))
                 for i in range(0, len(paras), 20)]
        return pages, None
    except Exception as e:
        return None, f"Could not read DOCX: {type(e).__name__}: {e}"


# ============================================================
# ANIMATED SVG BOOK
# ============================================================
def _render_book_svg():
    theme = st.session_state.get("theme", "light")
    if theme == "dark":
        cover_dark = "#1F2937"; cover_light = "#374151"
        pages = "#E5E7EB"; spine = "#00F0FF"; word_color = "#00F0FF"
        shadow = "rgba(0, 240, 255, 0.35)"
    else:
        cover_dark = "#78350F"; cover_light = "#B45309"
        pages = "#FFFFFF"; spine = "#D4AF37"; word_color = "#B8860B"
        shadow = "rgba(212, 175, 55, 0.35)"

    st.markdown(f"""
    <div style="display:flex; justify-content:center; padding:1.5rem 0 2.5rem 0;">
    <svg width="360" height="220" viewBox="0 0 360 220"
         xmlns="http://www.w3.org/2000/svg" aria-label="Animated book">
      <defs>
        <filter id="bookShadow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur in="SourceAlpha" stdDeviation="6" />
          <feOffset dx="0" dy="4" result="offsetblur" />
          <feComponentTransfer><feFuncA type="linear" slope="0.35"/></feComponentTransfer>
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <linearGradient id="coverGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="{cover_light}"/>
          <stop offset="100%" stop-color="{cover_dark}"/>
        </linearGradient>
        <linearGradient id="pageGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="{pages}" stop-opacity="1"/>
          <stop offset="100%" stop-color="{pages}" stop-opacity="0.85"/>
        </linearGradient>
      </defs>
      <ellipse cx="180" cy="200" rx="140" ry="8" fill="{shadow}" opacity="0.35">
        <animate attributeName="rx" values="140;130;140" dur="4s" repeatCount="indefinite"/>
        <animate attributeName="opacity" values="0.35;0.2;0.35" dur="4s" repeatCount="indefinite"/>
      </ellipse>
      <g filter="url(#bookShadow)">
        <path d="M 50 60 Q 90 50 175 55 L 175 185 Q 90 180 50 190 Z"
              fill="url(#coverGrad)" stroke="{spine}" stroke-width="1.5"/>
        <path d="M 175 55 Q 260 50 310 60 L 310 190 Q 260 180 175 185 Z"
              fill="url(#coverGrad)" stroke="{spine}" stroke-width="1.5"/>
        <line x1="175" y1="55" x2="175" y2="185"
              stroke="{spine}" stroke-width="2.5" opacity="0.7"/>
      </g>
      <g opacity="0.9">
        <path d="M 58 66 Q 96 58 172 62 L 172 180 Q 96 176 58 182 Z"
              fill="url(#pageGrad)" stroke="#9CA3AF" stroke-width="0.6"/>
        <path d="M 62 72 Q 98 64 170 68 L 170 176 Q 98 172 62 178 Z"
              fill="url(#pageGrad)" stroke="#9CA3AF" stroke-width="0.6"/>
      </g>
      <g opacity="0.55">
        <rect x="80" y="90" width="70" height="3" rx="1.5" fill="#6B7280"/>
        <rect x="80" y="102" width="60" height="3" rx="1.5" fill="#6B7280"/>
        <rect x="80" y="114" width="72" height="3" rx="1.5" fill="#6B7280"/>
        <rect x="80" y="126" width="52" height="3" rx="1.5" fill="#6B7280"/>
      </g>
      <g>
        <path d="M 180 62 Q 258 58 302 66 L 302 184 Q 258 176 180 180 Z"
              fill="url(#pageGrad)" stroke="#9CA3AF" stroke-width="0.6"
              transform-origin="180 121">
          <animateTransform attributeName="transform" type="rotate"
                            values="0 180 121; 0 180 121; -180 180 121; -180 180 121; 0 180 121"
                            keyTimes="0; 0.15; 0.45; 0.85; 1"
                            dur="5s" repeatCount="indefinite"/>
        </path>
        <path d="M 182 68 Q 256 64 298 72 L 298 178 Q 256 172 182 176 Z"
              fill="url(#pageGrad)" stroke="#9CA3AF" stroke-width="0.6"
              transform-origin="182 122">
          <animateTransform attributeName="transform" type="rotate"
                            values="0 182 122; 0 182 122; -180 182 122; -180 182 122; 0 182 122"
                            keyTimes="0; 0.3; 0.6; 0.95; 1"
                            dur="5s" repeatCount="indefinite"/>
        </path>
      </g>
      <g font-family="Plus Jakarta Sans, sans-serif" font-weight="700"
         font-size="11" fill="{word_color}" text-anchor="middle">
        <text x="120" y="80" opacity="0">words
          <animate attributeName="opacity" values="0;1;0"
                   keyTimes="0;0.5;1" dur="3.5s" begin="0s" repeatCount="indefinite"/>
          <animateTransform attributeName="transform" type="translate"
                            values="0 0; -8 -60" dur="3.5s" begin="0s"
                            repeatCount="indefinite"/>
        </text>
        <text x="180" y="80" opacity="0">ideas
          <animate attributeName="opacity" values="0;1;0"
                   keyTimes="0;0.5;1" dur="3.5s" begin="0.9s" repeatCount="indefinite"/>
          <animateTransform attributeName="transform" type="translate"
                            values="0 0; 4 -70" dur="3.5s" begin="0.9s"
                            repeatCount="indefinite"/>
        </text>
        <text x="240" y="80" opacity="0">insight
          <animate attributeName="opacity" values="0;1;0"
                   keyTimes="0;0.5;1" dur="3.5s" begin="1.8s" repeatCount="indefinite"/>
          <animateTransform attributeName="transform" type="translate"
                            values="0 0; 10 -55" dur="3.5s" begin="1.8s"
                            repeatCount="indefinite"/>
        </text>
        <text x="200" y="80" opacity="0">meaning
          <animate attributeName="opacity" values="0;1;0"
                   keyTimes="0;0.5;1" dur="3.5s" begin="2.6s" repeatCount="indefinite"/>
          <animateTransform attributeName="transform" type="translate"
                            values="0 0; -6 -65" dur="3.5s" begin="2.6s"
                            repeatCount="indefinite"/>
        </text>
      </g>
    </svg>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# RAG ENGINE
# ============================================================
def _chunk_document(filename, pages, chunk_size=500, overlap=100):
    chunks = []; cid = 0
    for page_num, page_text in pages:
        words = page_text.split()
        if not words: continue
        if len(words) <= chunk_size:
            chunks.append({"chunk_id": cid, "source": filename,
                           "page": page_num, "text": page_text}); cid += 1
            continue
        step = max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            piece = words[i:i + chunk_size]
            if not piece: break
            chunks.append({"chunk_id": cid, "source": filename,
                           "page": page_num, "text": " ".join(piece)})
            cid += 1
            if i + chunk_size >= len(words): break
    return chunks


def _build_index(chunks):
    embedder = _load_embedder()
    texts = [c["text"] for c in chunks]
    if not texts: return embedder, None
    vectors = embedder["encode"](texts)
    return embedder, vectors


def _retrieve(query, chunks, embedder, vectors, top_k=4):
    if vectors is None or len(chunks) == 0: return []
    q_vec = embedder["encode"]([query])[0]
    scores = vectors @ q_vec
    order = scores.argsort()[::-1][:top_k]
    return [{**chunks[int(i)], "score": float(scores[int(i)])}
            for i in order if scores[i] > 0]


def _answer_with_citations(query, retrieved):
    if not retrieved:
        return {"answer": "No relevant passages found in the loaded documents.",
                "citations": []}
    context = "\n\n".join(
        f"[Source: {r['source']}, Page {r['page']}]\n{r['text']}"
        for r in retrieved)
    prompt = ("You are a precise research assistant. Answer the user's "
              "question using ONLY the passages below. For every claim, cite "
              "the source using the exact format [Source: filename, Page N]. "
              "If the passages do not contain the answer, say so plainly — do "
              "not guess.\n\n"
              f"--- PASSAGES ---\n{context}\n--- END ---\n\n"
              f"Question: {query}\n\nAnswer (with inline citations):")
    try:
        answer = (ask(prompt, temperature=0.1) or "").strip()
    except Exception as e:
        return {"answer": f"(LLM unavailable: {type(e).__name__})", "citations": []}
    citations = re.findall(r"\[Source:\s*([^,\]]+),\s*Page\s*(\d+)\]", answer)
    citations = [{"source": s.strip(), "page": int(p)} for s, p in citations]
    return {"answer": answer, "citations": citations}


# ============================================================
# STATE
# ============================================================
NL_KEYS = ("doc_input", "doc_pages", "doc_filename", "doc_chunks",
           "doc_embedder", "doc_vectors", "doc_cleaned", "doc_clean_stats",
           "doc_tokens", "doc_extractive", "doc_abstractive",
           "doc_phrases", "doc_clusters", "doc_search_results",
           "doc_rag_history", "doc_api_key", "_doc_clean_hash",
           "_doc_last_sig")


def _init_state():
    for k in NL_KEYS:
        if k not in st.session_state:
            st.session_state[k] = None
    if st.session_state.get("doc_pages") is None:
        st.session_state["doc_pages"] = []
    if st.session_state.get("doc_chunks") is None:
        st.session_state["doc_chunks"] = []
    if st.session_state.get("doc_rag_history") is None:
        st.session_state["doc_rag_history"] = []


def _reset_derived():
    for k in ("doc_cleaned", "doc_clean_stats", "doc_tokens",
              "doc_extractive", "doc_abstractive", "doc_phrases",
              "doc_clusters", "doc_search_results", "_doc_clean_hash"):
        st.session_state[k] = None


def _ingest_file(up):
    if up is None: return
    sig = (up.name, getattr(up, "size", None))
    if st.session_state.get("_doc_last_sig") == sig: return
    st.session_state["_doc_last_sig"] = sig

    name = up.name.lower()
    pages = None; err = None

    if name.endswith(".pdf"):
        pages, err = _read_pdf(up)
    elif name.endswith(".docx"):
        pages, err = _read_docx(up)
    elif name.endswith(".csv"):
        os.makedirs("outputs", exist_ok=True)
        tmp = os.path.join("outputs", f"_doc_{uuid.uuid4().hex[:6]}.csv")
        with open(tmp, "wb") as fh: fh.write(up.getbuffer())
        try:
            df, _ = load_table_with_notes(tmp)
            text_cols = [c for c in df.columns
                         if df[c].dtype == object
                         and df[c].astype(str).str.len().mean() > 20]
            if not text_cols:
                st.error("No text-like column found in the CSV."); return
            col = text_cols[0]
            rows = df[col].dropna().astype(str).tolist()
            pages = [(i + 1, txt) for i, txt in enumerate(rows) if txt.strip()]
        except Exception as e:
            st.error(f"Could not read CSV: {type(e).__name__}: {e}"); return
    else:
        text, err = _read_plain_text(up)
        if not err:
            pages = [(1, text)] if text.strip() else []

    if err:
        st.error(err); return
    if not pages:
        st.warning("No usable text found in the file."); return

    chunks = _chunk_document(up.name, pages)
    with st.spinner("Indexing document…"):
        embedder, vectors = _build_index(chunks)

    st.session_state["doc_pages"] = pages
    st.session_state["doc_filename"] = up.name
    st.session_state["doc_chunks"] = chunks
    st.session_state["doc_embedder"] = embedder
    st.session_state["doc_vectors"] = vectors
    _reset_derived()
    st.success(f"Indexed {len(pages)} pages → {len(chunks)} chunks from {up.name}")


def _render_export():
    st.divider()
    st.markdown(t("doc_export_title"))
    cleaned = (st.session_state.get("doc_cleaned")
               or "\n\n".join(p for _, p in st.session_state.get("doc_pages") or []))
    ex = (st.session_state.get("doc_extractive") or {}).get("summary", "")
    ab = (st.session_state.get("doc_abstractive") or {}).get("summary", "")
    stats = st.session_state.get("doc_clean_stats") or {}

    fmt = st.radio(t("doc_export_format"),
                    [t("doc_export_excel"), t("doc_export_sql"),
                     t("doc_export_txt")],
                    horizontal=True, key="doc_export_fmt")

    row = {"filename": st.session_state.get("doc_filename", "document"),
           "original_chars": sum(len(p) for _, p in st.session_state.get("doc_pages") or []),
           "cleaned_chars": len(cleaned or ""),
           "reduction_pct": stats.get("reduction_pct", 0),
           "extractive_summary": ex,
           "abstractive_summary": ab,
           "cleaned_text": (cleaned or "")[:50000]}

    if fmt == t("doc_export_excel"):
        try:
            buf = io.BytesIO()
            pd.DataFrame([row]).to_excel(buf, index=False, engine="openpyxl")
            st.download_button(t("doc_export_download_xlsx"),
                                data=buf.getvalue(),
                                file_name="doc_archive.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="doc_dl_xlsx")
        except ImportError:
            st.warning("Install `openpyxl` for Excel. Falling back to CSV.")
            st.download_button(t("doc_export_download_csv"),
                                data=pd.DataFrame([row]).to_csv(index=False).encode("utf-8"),
                                file_name="doc_archive.csv", mime="text/csv",
                                key="doc_dl_csv")
    elif fmt == t("doc_export_sql"):
        def esc(s): return str(s).replace("'", "''") if s else ""
        sql = ("CREATE TABLE IF NOT EXISTS documents (\n"
               "  id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
               "  filename TEXT, original_chars INTEGER, cleaned_chars INTEGER,\n"
               "  reduction_pct REAL, extractive_summary TEXT,\n"
               "  abstractive_summary TEXT, cleaned_text TEXT\n);\n\n"
               f"INSERT INTO documents VALUES (NULL, '{esc(row['filename'])}', "
               f"{row['original_chars']}, {row['cleaned_chars']}, {row['reduction_pct']}, "
               f"'{esc(row['extractive_summary'])}', '{esc(row['abstractive_summary'])}', "
               f"'{esc(row['cleaned_text'])}');")
        st.code(sql, language="sql")
        st.download_button(t("doc_export_download_sql"),
                            data=sql.encode("utf-8"),
                            file_name="doc_archive.sql", mime="application/sql",
                            key="doc_dl_sql")
    else:
        bundle = (f"=== CLEANED TEXT ===\n{cleaned or ''}\n\n"
                  f"=== EXTRACTIVE SUMMARY ===\n{ex}\n\n"
                  f"=== ABSTRACTIVE SUMMARY ===\n{ab}\n")
        st.download_button(t("doc_export_download_txt"),
                            data=bundle.encode("utf-8"),
                            file_name="doc_archive.txt", mime="text/plain",
                            key="doc_dl_txt")


# ============================================================
# MAIN
# ============================================================
def render_tools():
    _init_state()

    ui.hero(t("doc_tools_hero_title"), t("doc_tools_hero_sub"),
            pills=[t("doc_tools_pill_rag"), t("doc_tools_pill_cited"),
                   t("doc_tools_pill_clean"), t("doc_tools_pill_summarize"),
                   t("doc_tools_pill_cluster")])

    st.markdown(t("doc_upload_title"))
    st.caption(t("doc_upload_hint"))
    up = st.file_uploader(
        "Upload document",
        type=["pdf", "docx", "txt", "md", "html", "htm", "csv"],
        label_visibility="collapsed", key="doc_uploader")
    _ingest_file(up)

    pages = st.session_state.get("doc_pages") or []
    if not pages:
        st.info(t("doc_upload_waiting"))
        return

    _render_book_svg()

    total_chars = sum(len(p) for _, p in pages)
    full_text = "\n\n".join(p for _, p in pages)
    st.caption(t("doc_indexed",
                  name=st.session_state.get("doc_filename"),
                  pages=len(pages), chars=total_chars,
                  chunks=len(st.session_state.get("doc_chunks") or [])))

    # ---- RAG ----
    st.divider()
    st.markdown(t("doc_rag_title"))
    st.caption(t("doc_rag_sub"))

    with st.expander(t("doc_rag_settings"), expanded=False):
        st.info("🔒 Cloud embedding with your own API key is coming soon. "
                 "The current version indexes locally using "
                 "sentence-transformers (or TF-IDF as a fallback) — no API "
                 "key needed.")
        st.slider(t("doc_rag_top_k"), 2, 8, 4, key="rag_top_k")
        if st.button(t("doc_rag_clear"), key="rag_clear_btn"):
            st.session_state["doc_rag_history"] = []
            st.rerun()

    for turn in st.session_state["doc_rag_history"]:
        if turn["role"] == "user":
            st.markdown(
                f'<div class="card" style="border-left:3px solid #00F0FF;">'
                f'<p style="font-weight:600;margin-bottom:0.3rem;">'
                f'{t("doc_rag_you")}</p>'
                f'<p>{turn["text"]}</p></div>', unsafe_allow_html=True)
        else:
            cites = turn.get("citations", [])
            cite_html = ""
            if cites:
                pills = "".join(
                    f'<span class="pill">📎 {c["source"]} · page {c["page"]}</span>'
                    for c in cites[:6])
                cite_html = f'<div style="margin-top:0.5rem;">{pills}</div>'
            st.markdown(
                f'<div class="card" style="border-left:3px solid #D4AF37;">'
                f'<p style="font-weight:600;margin-bottom:0.3rem;">'
                f'{t("doc_rag_answer")}</p>'
                f'<p>{turn["text"].replace(chr(10), "<br>")}</p>'
                f'{cite_html}</div>', unsafe_allow_html=True)

    with st.form("rag_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            q = st.text_input(t("doc_rag_ask_btn"),
                              placeholder=t("doc_rag_input_placeholder"),
                              label_visibility="collapsed", key="rag_query")
        with c2:
            sent = st.form_submit_button(t("doc_rag_ask_btn"),
                                          use_container_width=True)

    if sent and q.strip():
        chunks = st.session_state.get("doc_chunks") or []
        embedder = st.session_state.get("doc_embedder")
        vectors = st.session_state.get("doc_vectors")
        if not chunks or embedder is None:
            st.warning(t("doc_rag_not_indexed"))
        else:
            with st.spinner(t("doc_rag_retrieving")):
                retrieved = _retrieve(q.strip(), chunks, embedder, vectors,
                                       top_k=st.session_state.get("rag_top_k", 4))
            with st.spinner(t("doc_rag_writing")):
                result = _answer_with_citations(q.strip(), retrieved)
            st.session_state["doc_rag_history"].append(
                {"role": "user", "text": q.strip()})
            st.session_state["doc_rag_history"].append(
                {"role": "assistant", "text": result["answer"],
                 "citations": result["citations"]})
            st.rerun()

    # ---- Clean & tokenize ----
    st.divider()
    st.markdown(t("doc_clean_title"))
    with st.expander(t("doc_clean_options"), expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            opt_lower = st.checkbox("Lowercase", value=True, key="doc_opt_lower")
            opt_html = st.checkbox("Strip HTML", value=True, key="doc_opt_html")
            opt_urls = st.checkbox("Strip URLs", value=True, key="doc_opt_urls")
            opt_emails = st.checkbox("Strip emails", value=True, key="doc_opt_emails")
        with c2:
            opt_punct = st.checkbox("Strip punctuation", value=False, key="doc_opt_punct")
            opt_digits = st.checkbox("Strip numbers", value=False, key="doc_opt_digits")
            opt_ws = st.checkbox("Collapse whitespace", value=True, key="doc_opt_ws")
            opt_stops = st.checkbox("Remove stop-words", value=False, key="doc_opt_stops")

    if st.button(t("doc_clean_btn"), type="primary", key="doc_clean_btn"):
        cleaned, stats = clean_text(
            full_text, lowercase=opt_lower, strip_html=opt_html,
            strip_urls=opt_urls, strip_emails=opt_emails,
            strip_punctuation=opt_punct, strip_digits=opt_digits,
            collapse_whitespace=opt_ws, remove_stopwords=opt_stops)
        st.session_state["doc_cleaned"] = cleaned
        st.session_state["doc_clean_stats"] = stats
        st.session_state["doc_tokens"] = None

    cleaned = st.session_state.get("doc_cleaned")
    if cleaned:
        stats = st.session_state.get("doc_clean_stats") or {}
        cols = st.columns(4)
        with cols[0]: ui.stat(stats.get("original_chars", 0), t("doc_clean_original"))
        with cols[1]: ui.stat(stats.get("cleaned_chars", 0), t("doc_clean_cleaned"))
        with cols[2]: ui.stat(f"{stats.get('reduction_pct', 0)}%", t("doc_clean_reduced"))
        with cols[3]: ui.stat(stats.get("stopwords_removed", 0), t("doc_clean_stopwords"))

        if st.button(t("doc_tokenize_btn"), key="doc_tokenize_btn"):
            st.session_state["doc_tokens"] = tokenize(
                cleaned, keep_stopwords=not opt_stops)
        tokens = st.session_state.get("doc_tokens")
        if tokens:
            st.metric(t("doc_tokenize_count"), f"{len(tokens):,}")
            counts = Counter(tokens).most_common(25)
            if counts:
                dfc = pd.DataFrame(counts, columns=["token", "count"])
                fig = px.bar(dfc, x="count", y="token", orientation="h",
                             color="count",
                             color_continuous_scale=[[0, "#1c2540"], [1, "#22d3ee"]])
                fig.update_layout(height=max(300, 22 * len(dfc)),
                                  showlegend=False, margin=dict(t=10, b=10),
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="#A1A1AA"))
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)

    # ---- Summarize ----
    st.divider()
    st.markdown(t("doc_summarize_title"))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(t("doc_summarize_extractive"))
        n_sent = st.slider(t("doc_summarize_sentences"), 1, 10, 3, key="doc_n_ex")
        if st.button(t("doc_summarize_extract_btn"), key="doc_extract_btn"):
            src = st.session_state.get("doc_cleaned") or full_text
            st.session_state["doc_extractive"] = summarize_extractive(src, n=n_sent)
        ex = st.session_state.get("doc_extractive")
        if ex:
            st.text_area("Summary", ex["summary"], height=140,
                         key="doc_view_ex", label_visibility="collapsed")
    with c2:
        st.markdown(t("doc_summarize_abstractive"))
        target_words = st.slider(t("doc_summarize_target_words"),
                                  30, 250, 80, 10, key="doc_n_ab")
        if st.button(t("doc_summarize_generate_btn"), key="doc_abstract_btn"):
            src = st.session_state.get("doc_cleaned") or full_text
            with st.spinner(t("doc_summarize_summarizing")):
                st.session_state["doc_abstractive"] = summarize_abstractive(
                    src, target_words=target_words)
        ab = st.session_state.get("doc_abstractive")
        if ab:
            st.text_area("Summary", ab["summary"], height=140,
                         key="doc_view_ab", label_visibility="collapsed")

    # ---- Key phrases ----
    st.divider()
    st.markdown(t("doc_phrases_title"))
    if st.button(t("doc_phrases_btn"), key="doc_phrases_btn"):
        src = st.session_state.get("doc_cleaned") or full_text
        st.session_state["doc_phrases"] = key_phrases(src, n=15)
    phrases = st.session_state.get("doc_phrases")
    if phrases:
        cols = st.columns(3)
        for i, item in enumerate(phrases):
            with cols[i % 3]:
                st.markdown(
                    f'<div class="card" style="padding:0.6rem 0.9rem;'
                    f'display:flex;justify-content:space-between;">'
                    f'<span>{item["phrase"]}</span>'
                    f'<span style="opacity:0.6;">×{item["count"]}</span></div>',
                    unsafe_allow_html=True)

    # ---- Clustering ----
    st.divider()
    st.markdown(t("doc_cluster_title"))
    if len(pages) >= 3:
        k = st.slider(t("doc_cluster_topics"), 2, min(10, len(pages) - 1),
                      min(4, max(2, len(pages) // 5)), key="doc_k")
        if st.button(t("doc_cluster_btn"), key="doc_cluster_btn"):
            texts = [p for _, p in pages]
            with st.spinner(t("doc_cluster_clustering")):
                st.session_state["doc_clusters"] = cluster_documents(texts, n_clusters=k)
        res = st.session_state.get("doc_clusters")
        if res:
            st.markdown(f"**{res['n_clusters']} topics** · "
                        f"silhouette **{res['silhouette']}**")
            for i in range(res["n_clusters"]):
                st.markdown(
                    f'<div class="card" style="padding:0.7rem 1rem;">'
                    f'<h4>{t("doc_cluster_topic_n", i=i+1)} '
                    f'<span class="pill">{t("doc_cluster_docs", n=res["sizes"][i])}</span></h4>'
                    f'<p>{", ".join(res["top_terms"][i][:6])}</p></div>',
                    unsafe_allow_html=True)
    else:
        st.info(t("doc_cluster_need_three"))

    # ---- Search ----
    st.divider()
    st.markdown(t("doc_search_title"))
    q = st.text_input(t("doc_search_title"),
                       placeholder=t("doc_search_placeholder"),
                       key="doc_search_q", label_visibility="collapsed")
    if q and st.button(t("doc_search_btn"), key="doc_search_btn"):
        vec = mat = None
        if st.session_state.get("doc_clusters"):
            vec = st.session_state["doc_clusters"].get("vectorizer")
            mat = st.session_state["doc_clusters"].get("matrix")
        results = search_documents(q, [p for _, p in pages], top_k=5,
                                     vectorizer=vec, matrix=mat)
        st.session_state["doc_search_results"] = results
    results = st.session_state.get("doc_search_results")
    if results:
        for r in results:
            st.markdown(
                f'<div class="card" style="padding:0.8rem 1rem;">'
                f'<span class="pill">score {r["score"]}</span>'
                f'<p style="margin-top:0.5rem;">{r["text"]}</p></div>',
                unsafe_allow_html=True)

    # ---- Translate & grammar ----
    st.divider()
    st.markdown(t("doc_translate_title"))
    c1, c2 = st.columns(2)
    with c1:
        lang = st.selectbox(t("doc_translate_lang"),
                             ["Spanish", "French", "German", "Italian",
                              "Portuguese", "Arabic", "Chinese (Simplified)",
                              "Japanese", "Hindi", "Russian"],
                             key="doc_lang")
        if st.button(t("doc_translate_btn"), key="doc_translate_btn"):
            with st.spinner(t("doc_translate_translating", lang=lang)):
                st.session_state["doc_translation"] = translate(
                    st.session_state.get("doc_cleaned") or full_text, lang)
    with c2:
        if st.button(t("doc_grammar_btn"), key="doc_grammar_btn"):
            with st.spinner(t("doc_grammar_correcting")):
                st.session_state["doc_grammar"] = fix_grammar(
                    st.session_state.get("doc_cleaned") or full_text)

    tr = st.session_state.get("doc_translation")
    if tr:
        st.text_area("Translation", tr["translation"], height=160,
                     key="doc_view_tr", label_visibility="collapsed")
    gr = st.session_state.get("doc_grammar")
    if gr:
        st.text_area("Corrected", gr["fixed"], height=160,
                     key="doc_view_gr", label_visibility="collapsed")

    _render_export()


render_dashboard = render_tools