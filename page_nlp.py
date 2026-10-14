"""
NLP pages: Tools (document processing) and Dashboard (corpus analytics).
"""
import os
import uuid
import json
import re
import io
import datetime
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


# ============================================================ FILE READERS
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
        return None, "Reading PDF files needs `pypdf`. Run: `pip install pypdf`."
    try:
        reader = PdfReader(io.BytesIO(uploaded.getvalue()))
        pages = []
        for p in reader.pages:
            try:
                txt = p.extract_text() or ""
            except Exception:
                txt = ""
            if txt.strip():
                pages.append(txt.strip())
        if not pages:
            return None, "That PDF has no extractable text (may be scanned)."
        return "\n\n".join(pages), None
    except Exception as e:
        return None, f"Could not read PDF: {type(e).__name__}: {e}"


def _read_docx(uploaded):
    try:
        import docx
    except ImportError:
        return None, "Reading Word files needs `python-docx`. Run: `pip install python-docx`."
    try:
        doc = docx.Document(io.BytesIO(uploaded.getvalue()))
        paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        if not paras:
            return None, "That Word document has no text paragraphs."
        return "\n\n".join(paras), None
    except Exception as e:
        return None, f"Could not read DOCX: {type(e).__name__}: {e}"


# ============================================================ NLP TOOLS
NL_KEYS = ("nlp_input", "nlp_docs", "nlp_docs_cleaned", "nlp_cleaned",
           "nlp_clean_stats", "nlp_tokens", "nlp_extractive",
           "nlp_abstractive", "nlp_translation", "nlp_grammar",
           "nlp_phrases", "nlp_clusters", "nlp_search_results",
           "_clean_hash", "_nlp_last_file_sig")


def _nlp_init():
    for k in NL_KEYS:
        if k not in st.session_state:
            st.session_state[k] = None
    if st.session_state.get("nlp_input") is None:
        st.session_state["nlp_input"] = ""
    if st.session_state.get("nlp_docs") is None:
        st.session_state["nlp_docs"] = []


def _nlp_reset_derived():
    for k in ("nlp_cleaned", "nlp_clean_stats", "nlp_tokens",
              "nlp_extractive", "nlp_abstractive", "nlp_translation",
              "nlp_grammar", "nlp_phrases", "nlp_clusters",
              "nlp_search_results", "nlp_docs_cleaned", "_clean_hash"):
        st.session_state[k] = None


def _render_export():
    st.divider()
    st.markdown(t("nlp_10_title"))
    st.caption(t("nlp_10_caption"))
    cleaned = st.session_state.get("nlp_cleaned") or st.session_state.get("nlp_input")
    ex = (st.session_state.get("nlp_extractive") or {}).get("summary", "")
    ab = (st.session_state.get("nlp_abstractive") or {}).get("summary", "")
    stats = st.session_state.get("nlp_clean_stats") or {}
    fmt = st.radio(t("nlp_export_fmt"),
                   [t("nlp_export_excel"), t("nlp_export_sql"),
                    t("nlp_export_txt")],
                   horizontal=True, key="nlp_export_fmt")
    row = {
        "original_chars": len(st.session_state.get("nlp_input") or ""),
        "cleaned_chars": len(cleaned or ""),
        "reduction_pct": stats.get("reduction_pct", 0),
        "extractive_summary": ex,
        "abstractive_summary": ab,
        "cleaned_text": (cleaned or "")[:50000],
    }
    if fmt == t("nlp_export_excel"):
        try:
            buf = io.BytesIO()
            pd.DataFrame([row]).to_excel(buf, index=False, engine="openpyxl")
            st.download_button(t("nlp_download_xlsx"), data=buf.getvalue(),
                                file_name="archive.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except ImportError:
            st.warning(t("nlp_openpyxl_missing"))
            st.download_button(t("nlp_download_csv"),
                                data=pd.DataFrame([row]).to_csv(index=False).encode("utf-8"),
                                file_name="archive.csv", mime="text/csv")
    elif fmt == t("nlp_export_sql"):
        def esc(s): return str(s).replace("'", "''") if s else ""
        sql = (
            "CREATE TABLE IF NOT EXISTS documents (\n"
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
            "  original_chars INTEGER, cleaned_chars INTEGER, reduction_pct REAL,\n"
            "  extractive_summary TEXT, abstractive_summary TEXT, cleaned_text TEXT\n"
            ");\n\n"
            f"INSERT INTO documents VALUES (NULL, {row['original_chars']}, "
            f"{row['cleaned_chars']}, {row['reduction_pct']}, "
            f"'{esc(row['extractive_summary'])}', '{esc(row['abstractive_summary'])}', "
            f"'{esc(row['cleaned_text'])}');"
        )
        st.code(sql, language="sql")
        st.download_button(t("nlp_download_sql"), data=sql.encode("utf-8"),
                            file_name="archive.sql", mime="application/sql")
    else:
        bundle = (f"=== CLEANED TEXT ===\n{cleaned or ''}\n\n"
                  f"=== EXTRACTIVE SUMMARY ===\n{ex}\n\n"
                  f"=== ABSTRACTIVE SUMMARY ===\n{ab}\n")
        st.download_button(t("nlp_download_txt"), data=bundle.encode("utf-8"),
                            file_name="archive_bundle.txt", mime="text/plain")


def _resolve_cluster_docs(docs):
    if len(docs) >= 3:
        return docs
    if len(docs) == 1:
        with st.expander(t("nlp_cluster_split"), expanded=True):
            method = st.radio(
                t("nlp_cluster_split_label"),
                [t("nlp_cluster_split_blank"),
                 t("nlp_cluster_split_sent"),
                 t("nlp_cluster_split_words")],
                horizontal=False, key="split_method")
            if method == t("nlp_cluster_split_blank"):
                parts = [p.strip() for p in re.split(r"\n\s*\n", docs[0]) if p.strip()]
            elif method == t("nlp_cluster_split_sent"):
                parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", docs[0])
                         if len(s.strip()) >= 20]
            else:
                n = st.slider(t("nlp_cluster_chunk"), 30, 300, 120, 10,
                              key="chunk_size")
                ws = docs[0].split()
                parts = [" ".join(ws[i:i + n]) for i in range(0, len(ws), n)]
            if len(parts) >= 3:
                st.success(t("nlp_cluster_split_ok").format(n=len(parts)))
                return parts
            st.info(t("nlp_cluster_split_no").format(n=len(parts)))
    return None


def _handle_upload(up):
    """Process an uploaded file ONLY when it actually changes."""
    if up is None:
        return
    sig = (up.name, getattr(up, "size", None))
    if st.session_state.get("_nlp_last_file_sig") == sig:
        return
    st.session_state["_nlp_last_file_sig"] = sig

    name = up.name.lower()
    if name.endswith(".csv"):
        os.makedirs("outputs", exist_ok=True)
        tmp = os.path.join("outputs", f"_nlp_{uuid.uuid4().hex[:6]}.csv")
        with open(tmp, "wb") as fh:
            fh.write(up.getbuffer())
        try:
            df, _ = load_table_with_notes(tmp)
            text_cols = [c for c in df.columns if df[c].dtype == object]
            if not text_cols:
                st.error("No text-like column found in the CSV.")
                return
            col = st.selectbox("Which column holds the text?",
                                text_cols, key="nlp_csv_col")
            docs = df[col].dropna().astype(str).tolist()
            st.session_state["nlp_input"] = "\n\n".join(docs)
            st.session_state["nlp_docs"] = docs
            _nlp_reset_derived()
            st.success(f"Loaded {len(docs)} documents from column '{col}'")
        except Exception as e:
            st.error(f"Could not read CSV: {type(e).__name__}: {e}")
    elif name.endswith(".pdf"):
        text, err = _read_pdf(up)
        if err:
            st.error(err)
        else:
            st.session_state["nlp_input"] = text
            st.session_state["nlp_docs"] = [text]
            _nlp_reset_derived()
            st.success(f"Loaded {len(text):,} chars from {up.name}")
    elif name.endswith(".docx"):
        text, err = _read_docx(up)
        if err:
            st.error(err)
        else:
            st.session_state["nlp_input"] = text
            st.session_state["nlp_docs"] = [text]
            _nlp_reset_derived()
            st.success(f"Loaded {len(text):,} chars from {up.name}")
    else:
        text, err = _read_plain_text(up)
        if err:
            st.error(err)
        else:
            st.session_state["nlp_input"] = text
            st.session_state["nlp_docs"] = [text] if text.strip() else []
            _nlp_reset_derived()
            st.success(f"Loaded {len(text):,} chars from {up.name}")


def render_tools():
    _nlp_init()
    ui.hero(
        t("nlp_tools_hero_title"),
        t("nlp_tools_hero_sub"),
        pills=[t("nlp_tools_pill_1"), t("nlp_tools_pill_2"),
               t("nlp_tools_pill_3"), t("nlp_tools_pill_4"),
               t("nlp_tools_pill_5")],
    )

    # ---- 1. INPUT
    st.markdown(t("nlp_1_title"))
    tab_paste, tab_upload = st.tabs([t("nlp_paste_tab"), t("nlp_upload_tab")])
    with tab_paste:
        pasted = st.text_area("Paste the document",
                               value=st.session_state.get("nlp_input") or "",
                               height=220, key="nlp_paste_input",
                               label_visibility="collapsed",
                               placeholder=t("nlp_paste_placeholder"))
        if pasted != (st.session_state.get("nlp_input") or ""):
            st.session_state["nlp_input"] = pasted
            st.session_state["nlp_docs"] = [pasted] if pasted.strip() else []
            st.session_state["_nlp_last_file_sig"] = None
            _nlp_reset_derived()
    with tab_upload:
        up = st.file_uploader(
            t("nlp_upload_label"),
            type=["txt", "md", "html", "htm", "pdf", "docx", "csv",
                  "json", "log", "py", "rst"],
            label_visibility="collapsed", key="nlp_upload")
        _handle_upload(up)

    text = st.session_state.get("nlp_input") or ""
    docs = st.session_state.get("nlp_docs") or []
    if not text.strip():
        st.info(t("nlp_no_input"))
        return
    st.caption(t("nlp_input_caption").format(
        chars=len(text), docs=len(docs) if len(docs) > 1 else 1))

    # ---- 2. CLEAN
    st.divider()
    st.markdown(t("nlp_2_title"))
    st.caption(t("nlp_2_caption"))
    with st.expander(t("nlp_clean_options"), expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            opt_lower = st.checkbox(t("nlp_opt_lower"), value=True, key="opt_lower")
            opt_html = st.checkbox(t("nlp_opt_html"), value=True, key="opt_html")
            opt_urls = st.checkbox(t("nlp_opt_urls"), value=True, key="opt_urls")
            opt_emails = st.checkbox(t("nlp_opt_emails"), value=True, key="opt_emails")
        with c2:
            opt_punct = st.checkbox(t("nlp_opt_punct"), value=False, key="opt_punct")
            opt_digits = st.checkbox(t("nlp_opt_digits"), value=False, key="opt_digits")
            opt_ws = st.checkbox(t("nlp_opt_ws"), value=True, key="opt_ws")
            opt_stops = st.checkbox(t("nlp_opt_stops"), value=False, key="opt_stops")

    current_opts = (opt_lower, opt_html, opt_urls, opt_emails,
                    opt_punct, opt_digits, opt_ws, opt_stops)
    current_hash = hash(current_opts)

    stored_hash = st.session_state.get("_clean_hash")
    if st.session_state.get("nlp_cleaned") and stored_hash is not None \
            and stored_hash != current_hash:
        st.warning(t("nlp_opt_changed"))

    if st.button(t("nlp_clean_btn"), type="primary", key="btn_clean"):
        cleaned, stats = clean_text(
            text, lowercase=opt_lower, strip_html=opt_html,
            strip_urls=opt_urls, strip_emails=opt_emails,
            strip_punctuation=opt_punct, strip_digits=opt_digits,
            collapse_whitespace=opt_ws, remove_stopwords=opt_stops)
        st.session_state["nlp_cleaned"] = cleaned
        st.session_state["nlp_clean_stats"] = stats
        st.session_state["_clean_hash"] = current_hash
        for k in ("nlp_tokens", "nlp_extractive", "nlp_abstractive",
                  "nlp_phrases", "nlp_clusters", "nlp_search_results",
                  "nlp_docs_cleaned"):
            st.session_state[k] = None
        if len(docs) > 1:
            cleaned_docs = [
                clean_text(d, lowercase=opt_lower, strip_html=opt_html,
                           strip_urls=opt_urls, strip_emails=opt_emails,
                           strip_punctuation=opt_punct, strip_digits=opt_digits,
                           collapse_whitespace=opt_ws,
                           remove_stopwords=opt_stops)[0]
                for d in docs]
            st.session_state["nlp_docs_cleaned"] = cleaned_docs

    cleaned = st.session_state.get("nlp_cleaned")
    if cleaned:
        stats = st.session_state.get("nlp_clean_stats") or {}
        counters = [
            ("html_tags_removed", "HTML tags removed"),
            ("urls_removed", "URLs removed"),
            ("emails_removed", "Emails removed"),
            ("control_chars_removed", "Control chars removed"),
            ("punctuation_removed", "Punctuation marks removed"),
            ("digit_runs_removed", "Number runs removed"),
            ("stopwords_removed", "Stop-words removed"),
        ]
        active = [(k, l) for k, l in counters if stats.get(k, 0)]
        cols = st.columns(4)
        col = 0
        for k, l in active:
            with cols[col % 4]:
                ui.stat(stats[k], l)
            col += 1
        with cols[col % 4]:
            ui.stat(f"{stats.get('reduction_pct', 0)}%", t("nlp_reduction"))
        with st.expander(t("nlp_cleaned_title"), expanded=True):
            st.text_area(t("nlp_cleaned_output"), value=cleaned, height=200,
                         key=f"nlp_cleaned_view_{st.session_state.get('_clean_hash')}",
                         label_visibility="collapsed")

    # ---- 3. TOKENIZE
    st.divider()
    st.markdown(t("nlp_3_title"))
    st.caption(t("nlp_3_caption"))
    c1, c2 = st.columns(2)
    with c1:
        tok_keep_stops = st.checkbox(t("nlp_tok_keep_stops"), value=True,
                                      key="tok_stops")
        if st.button(t("nlp_tokenize_btn"), key="btn_tokenize"):
            src = st.session_state.get("nlp_cleaned") or text
            toks = tokenize(src, keep_stopwords=tok_keep_stops)
            st.session_state["nlp_tokens"] = toks
    tokens = st.session_state.get("nlp_tokens")
    if tokens:
        with c2:
            st.metric(t("nlp_tokens"), f"{len(tokens):,}")
        counts = Counter(tokens).most_common(30)
        if counts:
            dfc = pd.DataFrame(counts, columns=["token", "count"])
            fig = px.bar(dfc, x="count", y="token", orientation="h",
                         color="count",
                         color_continuous_scale=[[0, "#1c2540"], [1, "#22d3ee"]])
            fig.update_layout(height=max(300, 22 * len(dfc)),
                              showlegend=False, margin=dict(t=10, b=10),
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#b6c2d4"))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        with st.expander(t("nlp_all_tokens"), expanded=False):
            st.write(" ".join(tokens[:500]) + (" …" if len(tokens) > 500 else ""))
            st.caption(t("nlp_all_tokens_caption").format(n=len(tokens)))

    # ---- 4. KEY PHRASES
    st.divider()
    st.markdown(t("nlp_4_title"))
    if st.button(t("nlp_phrases_btn"), key="btn_phrases"):
        src = st.session_state.get("nlp_cleaned") or text
        st.session_state["nlp_phrases"] = key_phrases(src, n=15)
    phrases = st.session_state.get("nlp_phrases")
    if phrases:
        cols = st.columns(3)
        for i, item in enumerate(phrases):
            with cols[i % 3]:
                st.markdown(
                    f'<div class="card" style="padding:0.6rem 0.9rem;'
                    f'display:flex;justify-content:space-between;">'
                    f'<span style="color:#7ee0c0;">{item["phrase"]}</span>'
                    f'<span style="color:#7f8b99;">×{item["count"]}</span></div>',
                    unsafe_allow_html=True)

    # ---- 5. SUMMARIZE
    st.divider()
    st.markdown(t("nlp_5_title"))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(t("nlp_extractive_title"))
        n_sent = st.slider(t("nlp_sentences"), 1, 10, 3, key="n_ex")
        if st.button(t("nlp_extract_btn"), key="btn_ex"):
            src = st.session_state.get("nlp_cleaned") or text
            st.session_state["nlp_extractive"] = summarize_extractive(src, n=n_sent)
        ex = st.session_state.get("nlp_extractive")
        if ex:
            st.text_area(t("nlp_extractive_output"), ex["summary"], height=140,
                         key="view_ex", label_visibility="collapsed")
            st.caption(f"_{ex['note']}_")
    with c2:
        st.markdown(t("nlp_abstractive_title"))
        target_words = st.slider(t("nlp_approx_words"), 30, 200, 80, 10, key="n_ab")
        if st.button(t("nlp_abstract_btn"), key="btn_ab"):
            src = st.session_state.get("nlp_cleaned") or text
            with st.spinner(t("nlp_summarizing")):
                st.session_state["nlp_abstractive"] = summarize_abstractive(
                    src, target_words=target_words)
        ab = st.session_state.get("nlp_abstractive")
        if ab:
            st.text_area(t("nlp_abstractive_output"), ab["summary"], height=140,
                         key="view_ab", label_visibility="collapsed")
            st.caption(f"_{ab['note']}_")

    # ---- 6. CLUSTERING
    st.divider()
    st.markdown(t("nlp_6_title"))
    st.caption(t("nlp_6_caption"))
    cluster_src = _resolve_cluster_docs(docs)
    if cluster_src is None or len(cluster_src) < 3:
        st.info(t("nlp_cluster_need_3"))
    else:
        cc1, cc2 = st.columns([1, 2])
        with cc1:
            k = st.slider(t("nlp_topics_count"), 2, min(10, len(cluster_src) - 1),
                          min(4, max(2, len(cluster_src) // 5)), key="n_clusters")
            if st.button(t("nlp_cluster_btn"), key="btn_cluster"):
                src_docs = st.session_state.get("nlp_docs_cleaned") or cluster_src
                with st.spinner(t("nlp_clustering")):
                    st.session_state["nlp_clusters"] = cluster_documents(
                        src_docs, n_clusters=k)
        with cc2:
            res = st.session_state.get("nlp_clusters")
            if res:
                st.markdown(t("nlp_topics_prefix").format(
                    n=res["n_clusters"], sil=res["silhouette"]))
                for i in range(res["n_clusters"]):
                    st.markdown(
                        f'<div class="card" style="padding:0.7rem 1rem;">'
                        f'<h4>{t("nlp_topic").format(i=i+1)} '
                        f'<span class="pill">{t("nlp_topic_docs").format(n=res["sizes"][i])}</span></h4>'
                        f'<p style="color:#7ee0c0;">{", ".join(res["top_terms"][i][:6])}</p></div>',
                        unsafe_allow_html=True)

    # ---- 7. SEARCH
    st.divider()
    st.markdown(t("nlp_7_title"))
    if len(docs) >= 1 or cluster_src:
        q = st.text_input(t("nlp_search_results"),
                           placeholder=t("nlp_search_placeholder"),
                           key="nlp_search_query")
        c1, c2 = st.columns([1, 4])
        with c1:
            top_k = st.slider(t("nlp_search_results"), 1, 10, 5, key="n_search")
        with c2:
            if st.button(t("nlp_search_btn"), key="btn_search"):
                search_pool = cluster_src or docs
                src_docs = st.session_state.get("nlp_docs_cleaned") or search_pool
                vec = mat = None
                if st.session_state.get("nlp_clusters"):
                    vec = st.session_state["nlp_clusters"].get("vectorizer")
                    mat = st.session_state["nlp_clusters"].get("matrix")
                st.session_state["nlp_search_results"] = search_documents(
                    q, src_docs, top_k=top_k, vectorizer=vec, matrix=mat)
        results = st.session_state.get("nlp_search_results")
        if results:
            for r in results:
                st.markdown(
                    f'<div class="card" style="padding:0.8rem 1rem;">'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<span class="pill">{t("nlp_search_score").format(s=r["score"])}</span>'
                    f'<span style="color:#7f8b99;font-size:0.8rem;">'
                    f'{t("nlp_search_doc").format(i=r["index"])}</span></div>'
                    f'<p style="margin-top:0.5rem;">{r["text"]}</p></div>',
                    unsafe_allow_html=True)
        elif results is not None:
            st.info(t("nlp_search_none"))

    # ---- 8. TRANSLATE
    st.divider()
    st.markdown(t("nlp_8_title"))
    c1, c2 = st.columns([1, 2])
    with c1:
        lang = st.selectbox(t("nlp_lang_label"),
                             ["Spanish", "French", "German", "Italian", "Portuguese",
                              "Arabic", "Chinese (Simplified)", "Japanese", "Hindi",
                              "Urdu", "Russian"], key="nlp_lang")
        if st.button(t("nlp_translate_btn"), key="btn_tr"):
            with st.spinner(t("nlp_translating").format(lang=lang)):
                st.session_state["nlp_translation"] = translate(text, lang)
            st.rerun()
    with c2:
        tr = st.session_state.get("nlp_translation")
        if tr:
            st.text_area(t("nlp_translation_output"), tr["translation"],
                         height=180, key="view_tr", label_visibility="collapsed")
            st.caption(f"_{tr['note']}_")

    # ---- 9. GRAMMAR
    st.divider()
    st.markdown(t("nlp_9_title"))
    if st.button(t("nlp_grammar_btn"), key="btn_gr"):
        with st.spinner(t("nlp_correcting")):
            st.session_state["nlp_grammar"] = fix_grammar(text)
    gr = st.session_state.get("nlp_grammar")
    if gr:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(t("nlp_original"))
            st.text_area(t("nlp_original"), text, height=220,
                         key="view_gr_orig", label_visibility="collapsed")
        with c2:
            st.markdown(t("nlp_corrected"))
            st.text_area(t("nlp_corrected"), gr["fixed"], height=220,
                         key="view_gr_fixed", label_visibility="collapsed")
        st.caption(f"_{gr['note']}_")

    _render_export()


# ============================================================ DASHBOARD
def _load_any_document(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        os.makedirs("outputs", exist_ok=True)
        tmp = os.path.join("outputs", f"_dash_{uuid.uuid4().hex[:6]}.csv")
        with open(tmp, "wb") as fh:
            fh.write(uploaded.getbuffer())
        try:
            df, _ = load_table_with_notes(tmp)
            return df, "csv", None
        except Exception as e:
            return None, None, f"CSV: {type(e).__name__}: {e}"
    if name.endswith(".pdf"):
        text, err = _read_pdf(uploaded)
        if err:
            return None, None, err
        return text, "single", None
    if name.endswith(".docx"):
        text, err = _read_docx(uploaded)
        if err:
            return None, None, err
        return text, "single", None
    text, err = _read_plain_text(uploaded)
    if err:
        return None, None, err
    return text, "single", None


def _word_cloud_png(text):
    try:
        from wordcloud import WordCloud
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        wc = WordCloud(width=900, height=420, background_color="#0a0e14",
                       colormap="cool", max_words=120, stopwords=STOP_WORDS,
                       prefer_horizontal=0.9).generate(text or "")
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="#0a0e14")
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#0a0e14")
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        return None


def _top_ngrams(texts, n=15, ngram=2):
    tokens_per_doc = []
    for txt in texts:
        words = re.findall(r"[a-zA-Z']+", (txt or "").lower())
        words = [w for w in words if len(w) > 2 and w not in STOP_WORDS]
        tokens_per_doc.append(words)
    counts = Counter()
    for tokens in tokens_per_doc:
        for i in range(len(tokens) - ngram + 1):
            counts[" ".join(tokens[i:i + ngram])] += 1
    return [{"ngram": k, "count": v} for k, v in counts.most_common(n)]


def _split_into_docs_single(text, method, chunk_size=200):
    if method.startswith("paragraphs") or method.startswith("فقرات"):
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 40]
    elif method.startswith("sentences") or method.startswith("جمل"):
        parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
                 if len(s.strip()) >= 40]
    else:
        ws = text.split()
        parts = [" ".join(ws[i:i + chunk_size])
                 for i in range(0, len(ws), chunk_size) if ws[i:i + chunk_size]]
    return parts


def render_dashboard():
    from llm import ask
    ui.hero(
        t("nlp_dash_hero_title"),
        t("nlp_dash_hero_sub"),
        pills=[t("nlp_dash_pill_1"), t("nlp_dash_pill_2"),
               t("nlp_dash_pill_3")],
    )

    up = st.file_uploader(t("nlp_dash_upload"),
                           type=["csv", "pdf", "docx", "txt", "md", "html", "htm"],
                           key="dash_upload", label_visibility="collapsed")

    if up is None:
        if st.session_state.get("nlp_docs"):
            st.info(t("nlp_dash_use_loaded"))
            if st.button(t("nlp_dash_use_loaded_btn"), type="primary"):
                st.session_state["dash_docs"] = st.session_state["nlp_docs"]
                st.rerun()
        return

    data, kind, err = _load_any_document(up)
    if err:
        st.error(f"Could not read the file: {err}")
        return

    if kind == "csv":
        df = data
        text_cols = [c for c in df.columns
                     if df[c].dtype == object
                     and df[c].astype(str).str.len().mean() > 20]
        if not text_cols:
            st.error("No text-like column found in the CSV.")
            return
        col = st.selectbox(t("nlp_dash_csv_pick"), text_cols,
                           key="dash_col_pick")
        docs = [d for d in df[col].dropna().astype(str).tolist() if d.strip()]
    else:
        text = data
        st.caption(f"Loaded {len(text):,} characters from {up.name}.")
        method = st.radio(
            t("nlp_dash_split_label"),
            [t("nlp_dash_split_1"), t("nlp_dash_split_2"), t("nlp_dash_split_3")],
            horizontal=True, key="dash_split")
        chunk = 200
        if method == t("nlp_dash_split_3"):
            chunk = st.slider(t("nlp_dash_chunk"), 50, 500, 200, 10,
                              key="dash_chunk")
        docs = _split_into_docs_single(text, method, chunk)
        st.caption(t("nlp_dash_split_ok").format(n=len(docs)))

    if not docs:
        st.warning("Nothing usable found in that file.")
        return

    st.caption(f"{len(docs)} documents · {sum(len(d) for d in docs):,} characters.")

    st.divider()
    lengths = [len(d.split()) for d in docs]
    vocab = set()
    for d in docs:
        for w in re.findall(r"[a-zA-Z']+", d.lower()):
            if w not in STOP_WORDS and len(w) > 2:
                vocab.add(w)
    c1, c2, c3, c4 = st.columns(4)
    with c1: ui.stat(len(docs), t("nlp_dash_stat_docs"))
    with c2: ui.stat(f"{sum(lengths):,}", t("nlp_dash_stat_words"))
    with c3: ui.stat(int(sum(lengths) / len(lengths)), t("nlp_dash_stat_avg"))
    with c4: ui.stat(f"{len(vocab):,}", t("nlp_dash_stat_vocab"))

    st.divider()
    st.markdown(t("nlp_dash_wordcloud"))
    png = _word_cloud_png(" ".join(docs))
    if png:
        st.image(png, use_container_width=True)
    else:
        st.info(t("nlp_dash_wc_missing"))

    st.divider()
    st.markdown(t("nlp_dash_sentiment"))
    st.caption(t("nlp_dash_sentiment_caption"))
    with st.spinner(t("nlp_dash_scoring")):
        try:
            sample = docs[:15]
            joined = "\n".join(f"{i+1}. {txt[:400]}" for i, txt in enumerate(sample))
            reply = (ask("For each numbered text below give -1 (negative), 0 (neutral) "
                          "or +1 (positive). Return ONLY a JSON array of integers.\n\n"
                          + joined, temperature=0.1) or "").strip()
            s, e = reply.find("["), reply.rfind("]")
            scores = json.loads(reply[s:e+1])[:len(sample)]
            method = "LLM"
        except Exception:
            scores = None
            method = "unavailable"
    if scores is None:
        st.info(t("nlp_dash_llm_unavail"))
    else:
        label_map = {-1: "negative", 0: "neutral", 1: "positive"}
        label_counts = Counter(label_map.get(int(s), "neutral") for s in scores)
        colors = {"positive": "#4ade80", "neutral": "#94a3b8", "negative": "#f43f5e"}
        fig = px.bar(x=list(label_counts.keys()), y=list(label_counts.values()),
                     color=list(label_counts.keys()), color_discrete_map=colors,
                     text=list(label_counts.values()))
        fig.update_layout(height=340, showlegend=False,
                          xaxis_title="sentiment", yaxis_title="documents",
                          margin=dict(t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#b6c2d4"))
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(t("nlp_dash_sent_method").format(m=method, n=len(scores)))

    st.divider()
    st.markdown(t("nlp_dash_length"))
    fig = px.histogram(x=lengths, nbins=30, color_discrete_sequence=["#22d3ee"],
                       labels={"x": t("nlp_dash_length_x"),
                               "y": t("nlp_dash_length_y")})
    fig.update_layout(height=340, showlegend=False, margin=dict(t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#b6c2d4"))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown(t("nlp_dash_bigrams"))
    bigrams = _top_ngrams(docs, n=15, ngram=2)
    if bigrams:
        dfb = pd.DataFrame(bigrams)
        fig = px.bar(dfb, x="count", y="ngram", orientation="h", color="count",
                     color_continuous_scale=[[0, "#1c2540"], [1, "#e879f9"]])
        fig.update_layout(height=max(340, 24 * len(dfb)), showlegend=False,
                          margin=dict(t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#b6c2d4"))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown(t("nlp_dash_vocab_growth"))
    seen = set(); vocab_by_doc = []
    for d in docs:
        for w in re.findall(r"[a-zA-Z']+", d.lower()):
            if w not in STOP_WORDS and len(w) > 2:
                seen.add(w)
        vocab_by_doc.append(len(seen))
    fig = px.line(x=list(range(1, len(docs) + 1)), y=vocab_by_doc,
                  labels={"x": t("nlp_dash_vocab_x"),
                          "y": t("nlp_dash_vocab_y")},
                  color_discrete_sequence=["#22d3ee"])
    fig.update_layout(height=320, margin=dict(t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#b6c2d4"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(t("nlp_dash_vocab_caption"))