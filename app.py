"""
ML in a Box — main driver.

Run as: streamlit run app.py
"""
import html
import streamlit as st

st.set_page_config(page_title="ML in a Box", page_icon="🤖", layout="wide",
                   initial_sidebar_state="expanded")

import ui
ui.inject()

import auth
from i18n import t
from page_ml import render_home, render_profile, render_pipeline, render_tuning
from page_nlp import render_tools, render_dashboard
from page_cv import render_cv
from page_info import (render_tutorial, render_glossary, render_chat,
                        render_about)
from llm import ask


# ============================================================ STATE
def init_state():
    defaults = {
        "page": "home",
        "log": [],
        "chat_history": [],
        "sidebar_chat": [],
        "retried": False,
        "lang": "en",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_run():
    for k in ("df", "source_path", "source_name", "load_notes", "target",
              "task", "features", "mode", "model_keys", "proposed_plan",
              "clean_plan", "warnings", "results", "extras", "notes",
              "advice", "importance", "output_folder", "plan_approved",
              "retried", "profile", "column_descriptions", "chat_history",
              "modality", "learning_mode", "learning_mode_radio",
              "agent_target_suggestion", "metric_preference",
              "interaction_proposals", "interaction_specs", "_pdf_path",
              "prof_chat", "manual_params", "_col_desc_cache",
              "nlp_input", "nlp_docs", "nlp_cleaned", "nlp_clean_stats",
              "nlp_tokens", "nlp_extractive", "nlp_abstractive",
              "nlp_translation", "nlp_grammar", "nlp_phrases",
              "nlp_clusters", "nlp_search_results", "_clean_hash"):
        st.session_state.pop(k, None)
    st.session_state.log = []
    st.session_state.retried = False


init_state()


# ============================================================ LOGIN GATE
if not auth.is_logged_in():
    auth.render_login()
    st.stop()

USER = auth.current_user()


# ============================================================ ACCESS
def _can_access(page_key: str) -> bool:
    if page_key in ("home", "about", "tutorial", "glossary", "chat", "profile"):
        return True
    if page_key == "run":
        return any(auth.can_use(m) for m in ("ml", "dl", "nlp", "cv"))
    if page_key == "tuning":
        return any(auth.can_use(m) for m in ("ml", "dl"))
    if page_key in ("nlp_tools", "nlp_dash"):
        return auth.can_use("nlp")
    if page_key == "cv":
        return auth.can_use("cv")
    return True


# ============================================================ SIDEBAR ASK
def _build_short_context():
    parts = []
    if st.session_state.get("source_name"):
        df = st.session_state.df
        parts.append(f"Dataset: {st.session_state.source_name}, "
                     f"{len(df)} rows, {len(df.columns)} columns. "
                     f"Columns: {list(df.columns)}")
    if st.session_state.get("target"):
        parts.append(f"Target: {st.session_state.target} "
                     f"({st.session_state.task}).")
    if st.session_state.get("results") is not None:
        parts.append("Leaderboard:\n" +
                     st.session_state.results.to_string(index=False))
    return "\n".join(parts) if parts else "No dataset loaded."


def _sidebar_answer(question):
    ctx = _build_short_context()
    prompt = ("You are the inline assistant inside an ML demo. Answer the user's "
              "question using ONLY the facts below. If not covered, say so. "
              "2-3 short sentences.\n\n"
              f"--- facts ---\n{ctx}\n--- end ---\n\nQuestion: {question}")
    try:
        return (ask(prompt, temperature=0.2) or "").strip()
    except Exception as e:
        return f"(Assistant unavailable: {type(e).__name__})"


def _sidebar_ask():
    st.markdown(
        '<div style="font-size:0.85rem;font-weight:600;color:#22d3ee;'
        'margin-top:0.4rem;">💬 Ask anything</div>'
        '<div style="font-size:0.75rem;color:#94a3b8;margin-bottom:0.4rem;">'
        'Uses the current session facts.</div>',
        unsafe_allow_html=True)
    with st.form("sidebar_ask_form", clear_on_submit=True):
        q = st.text_input("Ask", placeholder="why did the best model win?",
                          label_visibility="collapsed",
                          key="sidebar_ask_input")
        submitted = st.form_submit_button("Send", use_container_width=True)
    if submitted and q.strip():
        a = _sidebar_answer(q.strip())
        st.session_state.sidebar_chat.append(("user", q.strip()))
        st.session_state.sidebar_chat.append(("assistant", a))
        st.rerun()
    if st.session_state.sidebar_chat:
        last_q = last_a = None
        for role, text in reversed(st.session_state.sidebar_chat):
            if role == "assistant" and last_a is None:
                last_a = text
            elif role == "user" and last_q is None:
                last_q = text
            if last_q is not None and last_a is not None:
                break
        if last_q:
            st.markdown(
                f'<div class="card" style="padding:0.6rem 0.8rem;">'
                f'<p style="font-size:0.78rem;color:#67e8f9;margin:0;">'
                f'<b>You:</b> {html.escape(last_q)}</p></div>',
                unsafe_allow_html=True)
        if last_a:
            st.markdown(
                f'<div class="card" style="padding:0.6rem 0.8rem;">'
                f'<p style="font-size:0.78rem;color:#b6c2d4;margin:0;">'
                f'<b>Assistant:</b> {html.escape(last_a)}</p></div>',
                unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Full chat", use_container_width=True, key="s2c"):
                st.session_state.page = "chat"
                st.rerun()
        with c2:
            if st.button("Clear", use_container_width=True, key="sclr"):
                st.session_state.sidebar_chat = []
                st.rerun()


# ============================================================ NAV
PAGES = [
    ("home",      "🏠", "Home",             True),
    ("about",     "ℹ️", "About",            True),
    ("tutorial",  "📚", "Tutorial",         True),
    ("profile",   "🔎", "Data Profiler",    True),
    ("run",       "⚡", "Run Pipeline",     None),
    ("cv",        "🖼️", "Computer Vision", None),
    ("tuning",    "⚙️", "Tuning",           None),
    ("nlp_tools", "📄", "NLP Tools",        None),
    ("nlp_dash",  "📊", "NLP Dashboard",    None),
    ("chat",      "💬", "Ask",              True),
    ("glossary",  "📖", "Glossary",         True),
]

with st.sidebar:
    # --- language switcher (top of sidebar)
    st.radio(
        t("language"),
        options=["en", "ar"],
        format_func=lambda x: "English" if x == "en" else "العربية",
        horizontal=True,
        key="lang",
        label_visibility="collapsed",
    )
    st.divider()

    modality_pills = " ".join(
        f'<span class="pill{" purple" if m in ("nlp", "cv") else ""}">'
        f'{m.upper()}</span>'
        for m in USER["modalities"])
    st.markdown(
        f'<div class="card" style="padding:0.9rem 1.1rem;margin-bottom:0.9rem;">'
        f'<p style="margin:0;font-size:0.72rem;color:#94a3b8;'
        f'text-transform:uppercase;letter-spacing:0.08em;font-weight:700;">'
        f'{t("signed_in")}</p>'
        f'<p style="margin:0.3rem 0 0.2rem 0;font-size:1rem;font-weight:700;'
        f'color:#e8eef7;">{html.escape(USER["display_name"])}</p>'
        f'<p style="margin:0;font-size:0.75rem;color:#94a3b8;">'
        f'@{html.escape(USER["username"])} · {html.escape(USER["role"])}</p>'
        f'<div style="margin-top:0.6rem;">{modality_pills}</div>'
        f'</div>',
        unsafe_allow_html=True)

    st.markdown(
        '<div style="font-size:1.1rem;font-weight:700;margin:0.2rem 0 0.6rem 0;'
        'color:#67e8f9;">🤖 ML in a Box</div>',
        unsafe_allow_html=True)

    for key, icon, label, always in PAGES:
        allowed = always if always is not None else _can_access(key)
        
        # Lock the Run Pipeline page if no dataset is loaded
        if key == "run" and not st.session_state.get("source_name"):
            allowed = False
            
        localized = t(key)
        if allowed:
            if st.button(f"{icon}  {localized}", key=f"nav_{key}",
                         use_container_width=True):
                st.session_state.page = key
                st.rerun()
        else:
            st.button(f"🔒  {localized}", key=f"nav_{key}_locked",
                      use_container_width=True, disabled=True)

    st.divider()

    if st.session_state.get("source_name"):
        st.markdown(
            f'<div class="card" style="padding:0.8rem 1rem;">'
            f'<h4 style="font-size:0.85rem;">📄 Loaded</h4>'
            f'<p style="font-size:0.82rem;">'
            f'{st.session_state.source_name}</p></div>',
            unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button(f"↺ {t('reset')}", use_container_width=True):
            reset_run()
            st.rerun()
    with c2:
        if st.button(f"⎋ {t('sign_out')}", use_container_width=True):
            auth.logout()
            reset_run()
            st.rerun()

    if st.session_state.get("log"):
        with st.expander("Agent trace", expanded=False):
            for line in st.session_state.log[-20:]:
                ui.trace_line(line)

    st.divider()
    _sidebar_ask()


# ============================================================ ROUTE
PAGE_RENDERERS = {
    "home":      render_home,
    "about":     render_about,
    "tutorial":  render_tutorial,
    "profile":   render_profile,
    "run":       render_pipeline,
    "tuning":    render_tuning,
    "nlp_tools": render_tools,
    "nlp_dash":  render_dashboard,
    "cv":        render_cv,
    "chat":      render_chat,
    "glossary":  render_glossary,
}

current = st.session_state.get("page", "home")
if not _can_access(current):
    st.session_state.page = "home"
    current = "home"

PAGE_RENDERERS[current]()

ui.watermark()