"""
ML in a Box — main driver.
Run as: streamlit run app.py
"""
import os
import streamlit as st

st.set_page_config(page_title="ML in a Box", page_icon="🤖",
                   layout="wide", initial_sidebar_state="collapsed")

import ui
import auth
from i18n import t
from page_ml import render_home, render_profile, render_pipeline, render_tuning
from page_nlp import render_tools
from page_info import (render_tutorial, render_glossary, render_chat,
                        render_about, render_agent_builder)


# ============================================================ STATE
def init_state():
    try:
        url_lang = st.query_params.get("lang")
        url_theme = st.query_params.get("theme")
        url_page = st.query_params.get("page")
    except Exception:
        url_lang = url_theme = url_page = None

    defaults = {
        "page": url_page if url_page else "home",
        "theme": url_theme if url_theme in ("light", "dark") else "light",
        "lang": url_lang if url_lang in ("en", "ar") else "en",
        "log": [], "chat_history": [], "retried": False,
        "byte_chat": [], "byte_chat_open": False,
        "_byte_tour_done": True, # Disabled tour
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def sync_url():
    try:
        if st.query_params.get("lang") != st.session_state.lang:
            st.query_params["lang"] = st.session_state.lang
        if st.query_params.get("theme") != st.session_state.theme:
            st.query_params["theme"] = st.session_state.theme
    except Exception:
        pass


init_state()

# ---------------------------------------------------------- URL LOGOUT
if st.query_params.get("logout") == "1":
    auth.logout()
    st.query_params.clear()
    st.rerun()


# ---------------------------------------------------------- RESET
def reset_run():
    """Clear all per-user data state. Keeps the user signed in."""
    keys = (
        # ---- ML pipeline
        "df", "source_path", "source_name", "load_notes", "target",
        "task", "features", "mode", "model_keys", "proposed_plan",
        "clean_plan", "warnings", "results", "extras", "notes",
        "advice", "importance", "output_folder", "plan_approved",
        "retried", "profile", "column_descriptions",
        "modality", "learning_mode", "agent_target_suggestion",
        "metric_preference", "interaction_proposals",
        "interaction_specs", "_pdf_path", "prof_chat",
        "manual_params", "_col_desc_cache", "enhancement_specs",
        "_tune_result", "_agent_suggested_hp", "use_smote",
        "_show_enhance_ui",
        # ---- Doc tools
        "doc_pages", "doc_filenames", "doc_filename", "doc_chunks",
        "doc_embedder", "doc_vectors", "doc_cleaned", "doc_clean_stats",
        "doc_tokens", "doc_extractive", "doc_abstractive",
        "doc_phrases", "doc_clusters", "doc_search_results",
        "doc_rag_history", "doc_api_key", "_doc_clean_hash",
        "_doc_last_sig", "doc_input", "doc_translation", "doc_grammar",
        # ---- Loaded model (Profiler upload)
        "_loaded_model", "_loaded_model_meta",
        # ---- Computer Vision
        "cv_task", "cv_extracted_root", "cv_workdir", "cv_class_names",
        "cv_n_train", "cv_n_val", "cv_model", "cv_model_label",
        "cv_model_classes", "cv_input_size", "cv_backbone_key",
        "cv_custom_layers", "cv_model_saved", "cv_loaded_slot",
        "cv_history", "cv_eval", "cv_layer_spec", "cv_upl_n", "cv_save_n",
        "cv_confirm_del", "cv_flash",
        # ---- YOLOv8 fine-tune
        "_yolo_zip_sig", "_yolo_workdir", "_yolo_yaml", "_yolo_names",
        "_yolo_weights",
        # ---- Chat + Byte
        "chat_history", "byte_chat", "_byte_modal", "_byte_tour_done",
    )
    for k in keys:
        st.session_state.pop(k, None)
    st.session_state.log = []
    st.session_state.retried = False
    st.session_state.page = "home"
    try:
        del st.query_params["page"]
    except Exception:
        pass


ui.inject()

# ============================================================ LOGIN GATE
if not auth.is_logged_in():
    auth.render_login()
    ui.render_byte("Hi! I'm Byte.")
    ui.render_byte_chat("User is on the login screen.",
                         show_tour_if_new=False)
    ui.watermark()
    st.stop()

USER = auth.current_user()

NAV_PAGES = [
    ("home",          "home"),
    ("profile",       "profile"),
    ("run",           "run"),
    ("doc_tools",     "doc_tools"),
    ("agent_builder", "agent_builder"),
    ("knowledge",     "knowledge"),
    ("about",         "about"),
]


def can_access(page_key: str) -> bool:
    if page_key in ("home", "about", "knowledge", "tutorial"):
        return True
    u = auth.current_user()
    if u is None:
        return False
    if u.get("role") == "visitor":
        return page_key in ("home", "about", "knowledge", "tutorial")
    if page_key == "profile":
        return True
    if page_key == "run":
        return any(auth.can_use(m) for m in ("ml", "dl", "nlp", "cv"))
    if page_key == "doc_tools":
        return auth.can_use("nlp")
    if page_key == "agent_builder":
        return True
    return True


def inject_nav_css():
    theme = st.session_state.theme
    if theme == "dark":
        bg = "rgba(15, 15, 15, 0.78)"
        border = "rgba(0, 240, 255, 0.28)"
        shadow = "0 8px 40px rgba(0, 240, 255, 0.18)"
        text = "#A1A1AA"; text_hover = "#00F0FF"
        hover_bg = "rgba(0, 240, 255, 0.12)"
        active_text = "#00F0FF"; active_bg = "rgba(0, 240, 255, 0.15)"
    else:
        bg = "rgba(255, 255, 255, 0.88)"
        border = "rgba(212, 175, 55, 0.4)"
        shadow = "0 8px 40px rgba(212, 175, 55, 0.22)"
        text = "#6B7280"; text_hover = "#059669"
        hover_bg = "rgba(52, 211, 153, 0.12)"
        active_text = "#059669"; active_bg = "rgba(52, 211, 153, 0.18)"

    st.markdown(f"""
    <style>
        .st-key-nav_buttons {{
            position: fixed !important;
            top: 20px !important;
            left: 50% !important;
            transform: translateX(-50%) !important;
            width: auto !important;
            min-width: 1000px !important;
            max-width: 98vw !important;
            background: {bg} !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border: 1px solid {border} !important;
            border-radius: 50px !important;
            padding: 0.35rem 0.7rem !important;
            z-index: 99999 !important;
            box-shadow: {shadow} !important;
        }}
        .st-key-nav_buttons button {{
            background: transparent !important;
            border: none !important;
            color: {text} !important;
            font-weight: 600 !important;
            padding: 0.4rem 0.4rem !important;
            box-shadow: none !important;
            font-size: 0.8rem !important;
            border-radius: 20px !important;
            white-space: nowrap !important;
            transition: all 0.2s ease !important;
            min-height: 0 !important;
            line-height: 1.2 !important;
            width: 100% !important;
        }}
        .st-key-nav_buttons button p,
        .st-key-nav_buttons button div {{
            color: inherit !important;
            font-size: inherit !important;
            font-weight: inherit !important;
            font-family: inherit !important;
        }}
        .st-key-nav_buttons button:hover {{
            color: {text_hover} !important;
            background: {hover_bg} !important;
            transform: none !important;
        }}
        .st-key-nav_buttons button:focus,
        .st-key-nav_buttons button:active {{
            box-shadow: none !important;
            outline: none !important;
        }}
        .st-key-nav_buttons [data-testid="stHorizontalBlock"] {{
            gap: 0 !important;
            align-items: center !important;
        }}
        .st-key-nav_buttons [data-testid="column"] {{
            padding: 0 !important;
        }}
    </style>
    """, unsafe_allow_html=True)

    cur = st.session_state.page
    if any(cur == k for k, _ in NAV_PAGES):
        st.markdown(
            f'<style>.st-key-nav_buttons .st-key-nav_{cur} button {{'
            f'color: {active_text} !important;'
            f'background: {active_bg} !important;'
            f'font-weight: 700 !important;}}</style>',
            unsafe_allow_html=True)

    lang_now = st.session_state.get("lang", "en")
    active_lang = "lang_en" if lang_now == "en" else "lang_ar"
    st.markdown(
        f'<style>.st-key-nav_buttons .st-key-nav_{active_lang} button {{'
        f'color: {active_text} !important;'
        f'background: {active_bg} !important;'
        f'font-weight: 700 !important;}}</style>',
        unsafe_allow_html=True)


def render_pill_nav():
    inject_nav_css()
    with st.container(key="nav_buttons"):
        # Removed logo -> now 13 columns
        cols = st.columns([0.3, 0.6, 1.1, 1.0, 0.8, 1.1, 1.0, 0.6, 0.4, 0.4, 0.4, 0.7, 0.8])

        with cols[0]:
            if st.button("?", key="nav_help", use_container_width=True):
                st.session_state["_byte_modal"] = None
                st.session_state.page = "tutorial"; st.rerun()

        for i, (k, label_key) in enumerate(NAV_PAGES):
            with cols[i + 1]:
                if st.button(t(label_key), key=f"nav_{k}",
                             use_container_width=True):
                    st.session_state["_byte_modal"] = None
                    st.session_state.page = k; st.rerun()

        with cols[8]:
            icon = "🌙" if st.session_state.theme == "light" else "☀️"
            if st.button(icon, key="nav_theme", use_container_width=True):
                st.session_state["_byte_modal"] = None
                st.session_state.theme = (
                    "dark" if st.session_state.theme == "light" else "light")
                sync_url()
                st.rerun()

        with cols[9]:
            if st.button("EN", key="nav_lang_en", use_container_width=True):
                st.session_state["_byte_modal"] = None
                st.session_state.lang = "en"
                sync_url()
                st.rerun()

        with cols[10]:
            if st.button("AR", key="nav_lang_ar", use_container_width=True):
                st.session_state["_byte_modal"] = None
                st.session_state.lang = "ar"
                sync_url()
                st.rerun()

        with cols[11]:
            if st.button("Reset", key="nav_reset", use_container_width=True):
                reset_run()
                st.rerun()

        with cols[12]:
            if st.button("Logout", key="nav_logout", use_container_width=True):
                auth.logout()
                reset_run()
                st.rerun()


sync_url()
render_pill_nav()

ui.render_byte(f"Hi {USER['display_name'].split()[0]}! I'm Byte.")

_PAGE_HINTS = {
    "home": "the Home page.",
    "profile": "the Data Profiler page.",
    "run": "the Run Pipeline page.",
    "doc_tools": "the Doc Tools page.",
    "agent_builder": "the Agent Builder page.",
    "knowledge": "the Knowledge Tree page.",
    "about": "the About page.",
    "tutorial": "the Tutorial page.",
    "chat": "the Ask page.",
    "tuning": "the Tuning page.",
}
_ctx = [
    f"Current page: {_PAGE_HINTS.get(st.session_state.page, st.session_state.page)}",
    f"Theme: {st.session_state.theme}.",
    f"Language: {'Arabic' if st.session_state.lang == 'ar' else 'English'}.",
]
if st.session_state.get("source_name"):
    _ctx.append(f"Dataset loaded: {st.session_state.source_name}.")
if st.session_state.get("results") is not None:
    _ctx.append("A trained model exists in this session.")

ui.render_byte_chat(context_hint=" ".join(_ctx))

PAGE_RENDERERS = {
    "home": render_home, "profile": render_profile, "run": render_pipeline,
    "tuning": render_tuning, "doc_tools": render_tools,
    "agent_builder": render_agent_builder, "knowledge": render_glossary,
    "about": render_about, "tutorial": render_tutorial, "chat": render_chat,
}

current = st.session_state.page
if not can_access(current):
    st.session_state.page = "home"
    current = "home"

renderer = PAGE_RENDERERS.get(current, render_home)
try:
    renderer()
except Exception as e:
    if type(e).__name__ in (
            "RerunException", "StopException", "ScriptControlException"):
        raise
    st.error(f"Page error: {type(e).__name__}: {e}")
    import traceback
    with st.expander("Traceback"):
        st.code(traceback.format_exc())

ui.watermark()