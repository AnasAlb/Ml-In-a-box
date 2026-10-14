"""
ML pages: Home, Data Profiler, Run Pipeline, Tuning.
"""
import os
import uuid

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

import ui
import auth
from i18n import t
from ml import (load_table_with_notes, profile_df, column_kind, conversion_note,
                engineering_summary, get_plan, validate_plan, restrict_plan,
                run as train_run, task_evidence, select_features,
                catalog, REASON, export_all, suggest_features,
                torch_status, TORCH_AVAILABLE, DEFAULT_NN_CONFIG,
                recommend_target, agent_propose_interactions,
                load_history, hyperparams, prune_features,
                auto_detect_modality, agent_tune_hyperparams) # Added imports
from nlp import nlp_catalog
from llm import ask


# ============================================================ HOME BODY
def _render_home_body():
    st.markdown(t("home_intro"))

    st.markdown(f"### {t('home_decisions_title')}")
    st.caption(t("home_decisions_sub"))

    decisions = t("home_decisions") or []
    for i in range(0, len(decisions), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(decisions):
                icon, title, desc = decisions[i + j]
                with col:
                    with st.container(border=True):
                        st.markdown(f"#### {icon} {title}")
                        st.caption(desc)

    st.info(t("home_fallback"))

    st.markdown(f"### {t('home_code_title')}")
    st.markdown(t("home_code_body"))

    st.markdown(f"### {t('home_output_title')}")
    outputs = [t("home_output_1"), t("home_output_2"), t("home_output_3")]
    cols = st.columns(3)
    for col, item in zip(cols, outputs):
        icon, title, desc = item
        with col:
            with st.container(border=True):
                st.markdown(f"#### {icon} {title}")
                st.caption(desc)


# ============================================================ HOME
def render_home():
    user = auth.current_user()

    ui.hero(
        t("hero_title"),
        t("hero_sub"),
        pills=["agentic", "explainable", "end-to-end"],
    )
    st.markdown(
        f'<p style="font-style:italic;opacity:0.55;color:#94a3b8;'
        f'font-size:0.95rem;margin:-1.2rem 0 1.6rem 0;padding-left:2.9rem;">'
        f'{t("hero_quote")}</p>',
        unsafe_allow_html=True,
    )

    _render_home_body()

    st.markdown("###")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button(f"📚  {t('tutorial_btn')}", use_container_width=True):
            st.session_state.page = "tutorial"; st.rerun()
    with c2:
        if st.button(f"🔎  {t('profile_btn')}", use_container_width=True,
                     type="primary"):
            st.session_state.page = "profile"; st.rerun()
    with c3:
        if st.button(f"⚙️  {t('tune_btn')}", use_container_width=True):
            st.session_state.page = "tuning"; st.rerun()

    status = torch_status()
    if status["cuda"]:
        st.success(t("home_gpu").format(name=status["device_name"]))
    elif status["torch"]:
        st.info(t("home_cpu"))
    else:
        st.info(t("home_no_torch"))

    if user:
        mods = ", ".join(m.upper() for m in user["modalities"])
        st.markdown(
            f'<div class="card" style="border-left:3px solid #22d3ee;">'
            f'<h4>{t("home_signed_in").format(name=user["display_name"])}</h4>'
            f'<p>{t("home_modules").format(mods=mods)}</p></div>',
            unsafe_allow_html=True)

    ui.sisyphus()


# ============================================================ PROFILE
def _kind_label(kind):
    key = f"prof_kind_{kind}"
    label = t(key)
    color = {
        "num": "blue", "numunit": "blue",
        "cat": "purple", "topcat": "purple", "bool": "purple",
        "date": "amber", "list": "amber", "text": "amber",
        "id": "red", "empty": "red",
    }.get(kind, "blue")
    return label, color


def _sample_block(df, sample_size=5):
    lines = []
    for c in df.columns:
        samples = df[c].dropna().astype(str).head(sample_size).tolist()
        lines.append(f"- column name: {c}\n  dtype: {df[c].dtype}\n  samples: {samples}")
    return "\n".join(lines)


def _describe_all_columns(df):
    source = st.session_state.get("source_name", "")
    cache_key = f"_col_desc::{source}::{tuple(df.columns)}"
    cache = st.session_state.setdefault("_col_desc_cache", {})
    if cache_key in cache:
        return cache[cache_key]
    prompt = (
        "For each column of a dataset, give ONE short sentence (max 20 words) "
        "saying what the column likely represents. Use only the name, dtype and "
        "sample values. If unclear, write 'unclear'.\n\n"
        "Return ONLY a JSON object mapping column name -> one-sentence description.\n\n"
        "Columns:\n" + _sample_block(df)
    )
    result = {c: "unclear" for c in df.columns}
    try:
        import json
        reply = (ask(prompt) or "").strip()
        start, end = reply.find("{"), reply.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(reply[start:end + 1])
            for c in df.columns:
                v = data.get(c)
                if isinstance(v, str) and 0 < len(v) <= 200:
                    result[c] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    cache[cache_key] = result
    return result


def _render_column_card(name, series, info, description, kind):
    label, color = _kind_label(kind)
    conversion = conversion_note(series, name)
    missing_pct = info.get("missing_pct", 0)
    unique = info.get("unique", 0)
    if missing_pct < 1:
        miss_pill = f'<span class="pill">{t("prof_missing_none")}</span>'
    elif missing_pct < 30:
        miss_pill = f'<span class="pill amber">{t("prof_missing_some").format(p=missing_pct)}</span>'
    else:
        miss_pill = f'<span class="pill red">{t("prof_missing_some").format(p=missing_pct)}</span>'
    conversions_html = (
        f'<p style="color:#94a3b8;font-size:0.85rem;margin-top:0.5rem;">'
        f'<b>{t("prof_auto_conversion")}</b> {conversion}</p>'
        if conversion else ""
    )
    st.markdown(
        f'<div class="card"><h4>{name} '
        f'<span class="pill {color}">{label}</span>{miss_pill}'
        f'<span class="pill">{t("prof_dtype").format(t=info["type"])}</span>'
        f'<span class="pill">{t("prof_distinct").format(n=unique)}</span></h4>'
        f'<p><b>{t("prof_likely_meaning")}</b> {description}</p>{conversions_html}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- DATASET CHAT
def _dataset_chat_ctx(df, prof, descs):
    lines = [f"{len(df)} rows, {len(df.columns)} columns."]
    for c in df.columns:
        s = df[c]
        parts = [f"column '{c}'"]
        parts.append(f"dtype {s.dtype}")
        parts.append(f"{s.nunique()} distinct values")
        miss = round(s.isna().mean() * 100, 1)
        if miss:
            parts.append(f"{miss}% missing")
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            try:
                parts.append(f"range {s.min():.3g} to {s.max():.3g}, mean {s.mean():.3g}")
            except Exception:
                pass
        desc = descs.get(c, "")
        if desc and desc != "unclear":
            parts.append(f"likely meaning: {desc}")
        samples = s.dropna().astype(str).head(3).tolist()
        if samples:
            parts.append(f"samples: {samples}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _dataset_answer(question, ctx):
    prompt = (
        "You are a data analyst. Answer the user's question about the dataset "
        "below in 3-6 sentences. Use only the facts given. If the answer is not "
        "in the facts, say so plainly. Be specific — mention column names and "
        "numbers where useful.\n\n"
        f"--- DATASET PROFILE ---\n{ctx}\n--- END ---\n\n"
        f"Question: {question}"
    )
    try:
        return (ask(prompt, temperature=0.2) or "").strip()
    except Exception as e:
        return f"(The assistant is unavailable: {type(e).__name__})"


def _render_dataset_chat():
    st.divider()
    st.markdown(f"### {t('prof_chat_title')}")
    st.caption(t("prof_chat_caption"))

    df = st.session_state.df
    prof = st.session_state.profile
    descs = _describe_all_columns(df)
    ctx = _dataset_chat_ctx(df, prof, descs)

    if "prof_chat" not in st.session_state:
        st.session_state.prof_chat = []

    suggestions = t("prof_chat_suggested")
    if not isinstance(suggestions, list) or not suggestions:
        suggestions = []
    if suggestions:
        st.markdown(t("chat_try_one"))
        cols = st.columns(len(suggestions))
        for i, q in enumerate(suggestions):
            with cols[i % len(suggestions)]:
                if st.button(q, key=f"profq_{i}", use_container_width=True):
                    st.session_state.prof_chat.append(("user", q))
                    with st.spinner(t("chat_thinking")):
                        a = _dataset_answer(q, ctx)
                    st.session_state.prof_chat.append(("assistant", a))
                    st.rerun()

    for role, text in st.session_state.prof_chat:
        safe = text.replace("\n", "<br>")
        if role == "user":
            st.markdown(
                f'<div class="card" style="border-left:3px solid #22d3ee;">'
                f'<p style="color:#67e8f9;font-weight:600;margin-bottom:0.3rem;">'
                f'{t("prof_you")}</p>'
                f'<p style="color:#e8eef7;">{safe}</p></div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="card" style="border-left:3px solid #e879f9;">'
                f'<p style="color:#f0a6fc;font-weight:600;margin-bottom:0.3rem;">'
                f'{t("prof_assistant")}</p>'
                f'<p style="color:#e8eef7;">{safe}</p></div>',
                unsafe_allow_html=True)

    with st.form("prof_chat_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            q = st.text_input(t("prof_chat_input"),
                              placeholder=t("prof_chat_placeholder"),
                              label_visibility="collapsed",
                              key="prof_chat_input")
        with c2:
            submitted = st.form_submit_button(t("prof_chat_send"),
                                               use_container_width=True)
    if submitted and q.strip():
        st.session_state.prof_chat.append(("user", q.strip()))
        with st.spinner(t("chat_thinking")):
            a = _dataset_answer(q.strip(), ctx)
        st.session_state.prof_chat.append(("assistant", a))
        st.rerun()

    if st.session_state.prof_chat:
        if st.button(t("prof_chat_clear"), key="prof_chat_clear"):
            st.session_state.prof_chat = []
            st.rerun()


def _render_uploaded_profile():
    df = st.session_state.df
    info = st.session_state.profile
    c1, c2, c3, c4 = st.columns(4)
    with c1: ui.stat(len(df), t("prof_stat_rows"))
    with c2: ui.stat(len(df.columns), t("prof_stat_cols"))
    with c3: ui.stat(int(df.isna().sum().sum()), t("prof_stat_missing"))
    with c4: ui.stat(info.get("duplicate_rows", 0), t("prof_stat_dups"))
    for n in st.session_state.load_notes:
        st.info(n)
    st.markdown(f"### {t('prof_columns_title')}")
    with st.spinner(t("prof_describe_spinner")):
        descs = _describe_all_columns(df)
    kinds = {c: column_kind(df[c], c) for c in df.columns}
    with st.expander(t("prof_preview"), expanded=False):
        st.dataframe(df.head(20), use_container_width=True)
    for c in df.columns:
        _render_column_card(c, df[c], info["columns"][c], descs.get(c, "unclear"), kinds[c])

    st.divider()
    st.markdown(f"### {t('prof_feature_eng_title')}")
    plan = engineering_summary(df)
    if not plan:
        st.info(t("prof_no_conversions"))
    else:
        for line in plan:
            st.markdown(
                f'<div class="card" style="padding:0.7rem 1rem;">'
                f'<p style="margin:0;color:#b6c2d4;">🧪 {line}</p></div>',
                unsafe_allow_html=True)

    _render_dataset_chat()

    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button(t("prof_continue_btn"), type="primary",
                     use_container_width=True):
            st.session_state.page = "run"; st.rerun()
    with c2:
        if st.button(t("prof_reupload_btn"), use_container_width=True):
            for k in ("df", "source_path", "source_name", "load_notes",
                      "profile", "_col_desc_cache", "target", "task",
                      "features", "mode", "model_keys",
                      "proposed_plan", "clean_plan", "warnings",
                      "plan_approved", "results", "extras", "notes",
                      "advice", "importance", "output_folder", "retried",
                      "agent_target_suggestion", "learning_mode",
                      "learning_mode_radio", "modality",
                      "metric_preference", "interaction_proposals",
                      "interaction_specs", "_pdf_path", "prof_chat",
                      "manual_params"):
                st.session_state.pop(k, None)
            st.session_state.log = []
            st.rerun()


def render_profile():
    ui.hero(t("prof_hero_title"), t("prof_hero_sub"),
            pills=[t("prof_pill_1"), t("prof_pill_2"), t("prof_pill_3")])
    if st.session_state.get("df") is not None:
        _render_uploaded_profile()
        return
    uploaded = st.file_uploader(t("prof_upload_label"),
                                 type=["csv", "xlsx", "xls"],
                                 label_visibility="collapsed")
    st.caption(t("prof_try_hint"))
    if uploaded is None:
        return
    os.makedirs("outputs", exist_ok=True)
    suffix = os.path.splitext(uploaded.name)[1]
    tmp = os.path.join("outputs", f"_upload_{uuid.uuid4().hex[:6]}{suffix}")
    with open(tmp, "wb") as f:
        f.write(uploaded.getbuffer())
    try:
        df, notes = load_table_with_notes(tmp)
    except Exception as e:
        st.error(f"Could not read the file: {type(e).__name__}: {e}")
        return
    st.session_state.df = df
    st.session_state.source_path = tmp
    st.session_state.source_name = uploaded.name
    st.session_state.load_notes = notes
    st.session_state.profile = profile_df(df)
    st.session_state.log = [f"Ingested {len(df)} rows × {len(df.columns)} cols from {uploaded.name}"]
    st.rerun()


# ============================================================ ADVANCED NN
def _advanced_nn_panel():
    cfg = st.session_state.setdefault("nn_config", dict(DEFAULT_NN_CONFIG))
    status = torch_status()
    if status["cuda"]:
        st.success(f"🟢 GPU ready · {status['device_name']} · torch {status['torch_version']}")
    elif status["torch"]:
        st.warning("🟡 PyTorch installed, no CUDA GPU. Training runs on CPU.")
    else:
        st.info("⚪ PyTorch not installed — using the scikit-learn MLP. Dropout and "
                "optimizer choice need PyTorch. Your other settings still apply.")
    if status["torch"]:
        framework_opts = ["sklearn (CPU, tuned for small tables)",
                          "pytorch (deep learning, GPU-capable)"]
        framework = st.radio(
            "Framework", framework_opts,
            index=0 if cfg.get("framework", "sklearn") == "sklearn" else 1,
            horizontal=True, key="nn_framework")
        cfg["framework"] = "sklearn" if framework.startswith("sklearn") else "torch"
    else:
        cfg["framework"] = "sklearn"
    use_torch = cfg["framework"] == "torch" and status["torch"]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Architecture**")
        layers_str = st.text_input(
            "Hidden layers (comma-separated)",
            value=", ".join(str(x) for x in cfg.get("hidden_layers", [64, 32])),
            key="nn_layers")
        try:
            cfg["hidden_layers"] = [max(4, int(x.strip()))
                                     for x in layers_str.split(",") if x.strip()]
            if not cfg["hidden_layers"]:
                cfg["hidden_layers"] = [64, 32]
        except ValueError:
            st.warning("Layers must be numbers separated by commas.")
        cfg["activation"] = st.selectbox(
            "Activation", ["relu", "tanh", "logistic", "identity"],
            index=["relu", "tanh", "logistic", "identity"].index(cfg.get("activation", "relu")),
            key="nn_act")
        if use_torch:
            cfg["dropout"] = st.slider(
                "Dropout", 0.0, 0.5, float(cfg.get("dropout", 0.0)), 0.05,
                key="nn_drop")
        else:
            st.caption("_Dropout needs PyTorch._")
    with c2:
        st.markdown("**Training**")
        if use_torch:
            cfg["optimizer"] = st.selectbox(
                "Optimizer", ["adam", "sgd", "adamw", "rmsprop"],
                index=["adam", "sgd", "adamw", "rmsprop"].index(cfg.get("optimizer", "adam")),
                key="nn_opt")
        else:
            cfg["optimizer"] = "adam"
            st.caption("_Optimizer needs PyTorch._")
        cfg["learning_rate"] = st.number_input(
            "Learning rate", min_value=1e-5, max_value=1.0,
            value=float(cfg.get("learning_rate", 0.001)),
            step=0.0001, format="%.4f", key="nn_lr")
        cfg["weight_decay"] = st.number_input(
            "L2 regularization", min_value=0.0, max_value=1.0,
            value=float(cfg.get("weight_decay", 0.0001)),
            step=0.0001, format="%.4f", key="nn_wd")
        if use_torch:
            cfg["batch_size"] = st.number_input(
                "Batch size", min_value=4, max_value=512,
                value=int(cfg.get("batch_size", 32)), step=4, key="nn_bs")
        else:
            cfg["batch_size"] = 32
        cfg["max_epochs"] = st.number_input(
            "Max epochs", min_value=10, max_value=2000,
            value=int(cfg.get("max_epochs", 100)), step=10, key="nn_ep")
        cfg["early_stopping"] = st.checkbox(
            "Early stopping", value=bool(cfg.get("early_stopping", True)), key="nn_es")

    layers = cfg["hidden_layers"]
    act = cfg["activation"]
    preview = ["Input"]
    for h in layers:
        preview.append(f"Dense({h})")
        preview.append(act.capitalize())
        if use_torch and cfg.get("dropout", 0) > 0:
            preview.append(f"Dropout({cfg['dropout']})")
    preview.append("Output")
    st.markdown(
        f'<div class="card" style="padding:0.9rem 1.3rem;background:linear-gradient(135deg,'
        f'rgba(34,211,238,0.06),rgba(232,121,249,0.06));">'
        f'<p style="margin:0;color:#94a3b8;font-size:0.72rem;text-transform:uppercase;'
        f'letter-spacing:0.1em;font-weight:700;">Architecture preview</p>'
        f'<p style="margin:0.5rem 0 0 0;color:#67e8f9;'
        f'font-family:Consolas,monospace;font-size:1rem;">'
        f'{" → ".join(preview)}</p></div>',
        unsafe_allow_html=True)
    st.session_state.nn_config = cfg


# ============================================================ MANUAL HYPERPARAMS
def _render_manual_hyperparams(model_keys, task):
    if not model_keys:
        return {}
    st.markdown(
        '<div class="card" style="border-left:3px solid #fbbf24;">'
        '<h4>⚙️ Hyperparameters</h4>'
        '<p>You picked these models by hand, so you can tune each one. Leave any '
        'value alone to use its default. The neural network has its own panel '
        'above — it is not shown here.</p></div>',
        unsafe_allow_html=True)

    cat = catalog(task, 10000)
    overrides = {}
    for key in model_keys:
        if key not in cat or key == "mlp":
            continue
        params = hyperparams(key, task)
        if not params:
            continue
        with st.expander(f"**{cat[key]['name']}**", expanded=False):
            overrides[key] = {}
            cols = st.columns(2)
            for i, p in enumerate(params):
                with cols[i % 2]:
                    widget_key = f"hp_{key}_{p['name']}"
                    if p["kind"] == "int":
                        v = st.slider(p["name"], min_value=int(p["min"]),
                                      max_value=int(p["max"]),
                                      value=int(p["default"]),
                                      step=int(p["step"]),
                                      help=p.get("help", ""),
                                      key=widget_key)
                    elif p["kind"] == "float":
                        v = st.number_input(p["name"], min_value=float(p["min"]),
                                            max_value=float(p["max"]),
                                            value=float(p["default"]),
                                            step=float(p["step"]),
                                            help=p.get("help", ""),
                                            key=widget_key)
                    elif p["kind"] == "choice":
                        v = st.selectbox(p["name"], p["choices"],
                                         index=p["choices"].index(p["default"])
                                         if p["default"] in p["choices"] else 0,
                                         help=p.get("help", ""),
                                         key=widget_key)
                    else:
                        continue
                    overrides[key][p["name"]] = v
    return overrides


# ============================================================ RUN helpers
def _reset_stages():
    for k in ("plan_approved", "proposed_plan", "clean_plan", "warnings",
              "results", "extras", "notes", "advice", "importance",
              "output_folder", "retried", "_pdf_path"):
        st.session_state.pop(k, None)


def _reset_modality():
    _reset_stages()
    for k in ("target", "task", "features", "mode", "model_keys",
              "learning_mode", "metric_preference",
              "interaction_proposals", "interaction_specs", "manual_params"):
        st.session_state.pop(k, None)


def _text_stats_block(df):
    from nlp import is_text_column
    text_cols = [c for c in df.columns if is_text_column(df[c], c)]
    if not text_cols:
        st.warning(t("setup_text_no_text"))
        return
    st.markdown(t("setup_text_stats_title"))
    for c in text_cols:
        s = df[c].dropna().astype(str)
        lengths = s.str.split().str.len()
        vocab = len(set(" ".join(s.head(3000)).lower().split()))
        st.markdown(f"**Column: `{c}`**")
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1: ui.stat(int(lengths.mean()), t("setup_text_avg_words"))
        with cc2: ui.stat(int(lengths.max()), t("setup_text_max_words"))
        with cc3: ui.stat(int(lengths.min()), t("setup_text_min_words"))
        with cc4: ui.stat(f"{vocab:,}", t("setup_text_vocab"))
    with st.expander(t("setup_text_preview")):
        for c in text_cols[:2]:
            st.markdown(f"**`{c}`**")
            for i, txt in enumerate(df[c].dropna().astype(str).head(5).tolist(), 1):
                truncated = txt[:280] + ("…" if len(txt) > 280 else "")
                st.markdown(
                    f'<div class="card" style="padding:0.6rem 0.9rem;">'
                    f'<p style="margin:0;color:#b6c2d4;font-size:0.85rem;">'
                    f'<b>#{i}</b> {truncated}</p></div>',
                    unsafe_allow_html=True)


# ============================================================ SETUP
def _stage_setup():
    df = st.session_state.df
    modality = st.session_state.get("modality", "tabular")
    can_ml = auth.can_use("ml")
    can_dl = auth.can_use("dl")

    if modality == "nlp":
        st.markdown(t("setup_nlp_step1"))
        _text_stats_block(df)
        st.divider()
    else:
        st.markdown(t("setup_ml_step1"))
        options = []
        if can_ml:
            options.append(t("setup_lm_ml"))
        if can_dl:
            options.append(t("setup_lm_dl"))
        if not options:
            st.error("Your account has no ML/DL access. Contact an admin.")
            return
        default_idx = 0
        current_lm = st.session_state.get("learning_mode")
        if current_lm == "dl" and can_dl and len(options) > 1:
            default_idx = 1
        choice = st.radio(
            t("setup_lm_pick"),
            options, index=default_idx, horizontal=False,
            key="learning_mode_radio",
            help=t("setup_lm_help") if len(options) > 1 else t("setup_lm_only_one"))
        st.session_state.learning_mode = "ml" if choice == t("setup_lm_ml") else "dl"

        mode_title = (t("setup_ml_mode_title") if st.session_state.learning_mode == "ml"
                      else t("setup_dl_mode_title"))
        mode_body = (t("setup_ml_mode_body") if st.session_state.learning_mode == "ml"
                     else t("setup_dl_mode_body"))
        border = "#22d3ee" if st.session_state.learning_mode == "ml" else "#fbbf24"
        st.markdown(
            f'<div class="card" style="border-left:3px solid {border};">'
            f'<h4>{mode_title}</h4><p>{mode_body}</p></div>',
            unsafe_allow_html=True)
        if st.session_state.learning_mode == "dl":
            with st.expander(t("setup_nn_expander"), expanded=True):
                _advanced_nn_panel()
        st.divider()
        st.markdown(t("setup_step2"))

    st.markdown(
        f'<div class="card" style="border-left:3px solid #22d3ee;">'
        f'<h4>{t("setup_agent_card_title")}</h4>'
        f'<p>{t("setup_agent_card_body")}</p></div>',
        unsafe_allow_html=True)
    if st.button(t("setup_agent_btn")):
        with st.spinner(t("setup_agent_reading")):
            rec = recommend_target(df)
        if rec:
            st.session_state["agent_target_suggestion"] = rec
            st.session_state.log.append(
                f"Agent suggested target: '{rec['target']}' ({rec['task']}) — {rec['reason']}")
        else:
            st.warning(t("setup_agent_fail"))
        st.rerun()
    suggestion = st.session_state.get("agent_target_suggestion")
    if suggestion:
        st.success(t("setup_agent_suggests").format(
            t=suggestion["target"], task=suggestion["task"],
            reason=suggestion["reason"]))

    with st.expander(t("setup_preview").format(
            name=st.session_state.source_name,
            rows=len(df), cols=len(df.columns))):
        st.dataframe(df.head(8), use_container_width=True)

    options = list(df.columns)
    default_idx = 0
    if suggestion and suggestion["target"] in options:
        default_idx = options.index(suggestion["target"])
    target = st.selectbox(t("setup_target_label"), options=options,
                          index=default_idx, key="target_pick")
    guess, conf, reason = task_evidence(df[target])
    st.info(t("setup_reading").format(task=guess, conf=conf, reason=reason))

    usable, dropped = select_features(df, target)
    features = st.multiselect(t("setup_features_label"),
                               options=[c for c in df.columns if c != target],
                               default=usable)
    for c, why in dropped:
        if c in features:
            st.warning(t("setup_feature_warn").format(c=c, why=why))

    text_chosen = [c for c in features if column_kind(df[c], c) == "text"]
    if modality == "nlp":
        if not text_chosen:
            st.error(t("setup_nlp_no_text"))
    else:
        if text_chosen and guess == "classification":
            st.info(t("setup_nlp_switch"))

    metric_preference = "balanced"
    if guess == "classification":
        st.markdown(t("setup_step3"))
        choice = st.radio(
            t("setup_metric_label"),
            [t("setup_metric_balanced"),
             t("setup_metric_precision"),
             t("setup_metric_recall")],
            index=0, key="metric_preference_radio",
            help=t("setup_metric_help"))
        if choice == t("setup_metric_precision"):
            metric_preference = "precision"
        elif choice == t("setup_metric_recall"):
            metric_preference = "recall"
        else:
            metric_preference = "balanced"
        st.session_state.metric_preference = metric_preference

    st.markdown(t("setup_step4"))
    st.caption(t("setup_step4_caption"))
    if st.button(t("setup_inter_btn")):
        numeric_chosen = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_chosen) < 2:
            st.info(t("setup_inter_need_two"))
        else:
            with st.spinner(t("setup_inter_thinking")):
                proposals = agent_propose_interactions(df, features, target, max_n=3)
            st.session_state["interaction_proposals"] = proposals
            if not proposals:
                st.info(t("setup_inter_none"))
        st.rerun()

    proposals = st.session_state.get("interaction_proposals") or []
    selected_interactions = []
    if proposals:
        st.markdown(t("setup_inter_proposals"))
        for i, p in enumerate(proposals):
            keep = st.checkbox(
                f"`{p['name']}` = {p['a']} {p['op']} {p['b']}  —  {p.get('reason', '')}",
                value=True, key=f"ix_prop_{i}")
            if keep:
                selected_interactions.append(p)
    st.session_state.interaction_specs = selected_interactions

    st.markdown(t("setup_step5"))
    mode_label = st.radio(t("setup_mode_label"),
                          [t("setup_mode_auto"), t("setup_mode_manual")],
                          key="model_mode_radio")
    mode = "auto" if mode_label == t("setup_mode_auto") else "manual"
    model_keys = []
    manual_overrides = {}
    if mode == "manual":
        learning_mode = st.session_state.get("learning_mode", "ml") if modality == "tabular" else "all"
        nn_cfg = st.session_state.get("nn_config", DEFAULT_NN_CONFIG) if modality == "tabular" else None
        cat = (nlp_catalog(int(len(df) * 0.8)) if modality == "nlp"
               else catalog(guess, int(len(df) * 0.8), nn_config=nn_cfg, mode=learning_mode))
        labels = {k: v["name"] for k, v in cat.items()}
        model_keys = st.multiselect(t("setup_manual_label"),
                                     options=list(labels.keys()),
                                     format_func=lambda k: labels[k],
                                     default=list(labels.keys())[:3])
        if modality == "tabular":
            manual_overrides = _render_manual_hyperparams(model_keys, guess)

    st.session_state.manual_params = manual_overrides

    if not (target and features):
        st.info(t("setup_pick_hint"))
        return
    if st.button(t("setup_continue_btn"), type="primary"):
        _reset_stages()
        st.session_state.target = target
        st.session_state.task = guess
        st.session_state.features = features
        st.session_state.mode = mode
        st.session_state.model_keys = model_keys
        st.session_state.log.append(
            f"[{modality.upper()}] target='{target}' features={len(features)} "
            f"metric={metric_preference}")
        st.rerun()


# ============================================================ PLAN
def _stage_plan():
    df = st.session_state.df
    st.markdown(t("plan_title"))
    st.caption(t("plan_caption"))
    if "proposed_plan" not in st.session_state:
        with st.spinner(t("plan_spinner")):
            proposed = get_plan(df)
        st.session_state.log.append("LLM proposed a cleaning plan")
        clean, warnings = validate_plan(proposed, df)
        restricted = restrict_plan(clean, st.session_state.features or list(df.columns))
        restricted["drop_duplicates"] = clean.get("drop_duplicates", False)
        st.session_state.proposed_plan = proposed
        st.session_state.clean_plan = restricted
        st.session_state.warnings = warnings
        st.session_state.log.append(f"Validated plan ({len(warnings)} warning(s))")
    restricted = st.session_state.clean_plan
    warnings = st.session_state.warnings
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(t("plan_will_happen"))
        yes, no = t("plan_yes"), t("plan_no")
        st.write(t("plan_drop_dupes").format(
            v=(yes if restricted["drop_duplicates"] else no)))
        imp = restricted.get("impute") or {}
        if imp:
            st.write(t("plan_fill_empty"))
            for c, m in imp.items():
                st.write(t("plan_fill_one").format(c=c, m=m))
        else:
            st.write(t("plan_fill_none"))
        out = restricted.get("outlier_columns") or []
        st.write(t("plan_trim").format(
            cols=(", ".join(out) if out else t("plan_nothing"))))
        dropped = restricted.get("drop_high_missing") or []
        st.write(t("plan_drop_empty").format(
            cols=(", ".join(dropped) if dropped else t("plan_nothing"))))
    with c2:
        st.markdown(t("plan_validator"))
        if warnings:
            for w in warnings:
                st.warning(w)
        else:
            st.success(t("plan_no_warnings"))
    with st.expander(t("plan_show_raw")):
        st.json(st.session_state.proposed_plan)
    st.info(t("plan_info"))
    c1, c2 = st.columns(2)
    if c1.button(t("plan_approve"), type="primary", use_container_width=True):
        st.session_state.log.append("Human decision: approved")
        st.session_state.plan_approved = True
        st.rerun()
    if c2.button(t("plan_reject"), use_container_width=True):
        st.session_state.log.append("Human decision: rejected")
        _reset_stages()
        st.rerun()


# ============================================================ TRAIN
def _stage_train():
    df = st.session_state.df
    features = st.session_state.features
    target = st.session_state.target
    task = st.session_state.task
    modality = st.session_state.get("modality", "tabular")
    learning_mode = st.session_state.get("learning_mode", "all") if modality == "tabular" else "all"
    df_run = df[features + [target]] if features else df
    nn_cfg = st.session_state.get("nn_config", DEFAULT_NN_CONFIG) if modality == "tabular" else None
    metric_preference = st.session_state.get("metric_preference", "balanced")
    interaction_specs = st.session_state.get("interaction_specs") or []
    source_name = st.session_state.get("source_name", "")
    manual_overrides = st.session_state.get("manual_params") or {}

    st.markdown(t("train_title"))
    prog = st.progress(0, text=t("train_starting"))
    with st.spinner(t("train_spinner")):
        prog.progress(15, text=t("train_fitting"))
        try:
            results, notes, extras = train_run(
                df_run, target, st.session_state.clean_plan, task,
                st.session_state.model_keys or None, self_check=True,
                nn_config=nn_cfg, mode=learning_mode,
                metric_preference=metric_preference,
                interaction_specs=interaction_specs,
                source_name=source_name,
                hyperparams_overrides=manual_overrides)
        except ValueError as e:
            prog.empty()
            st.error(t("train_stopped").format(err=str(e)))
            if st.button(t("train_back")):
                st.session_state.pop("plan_approved", None)
                st.rerun()
            return
        prog.progress(65, text=t("train_importance"))
        importance = None
        try:
            pipe = extras["best_pipeline"]
            X = df_run[extras["features"]]
            y = df_run[target]
            if extras.get("mode") != "nlp" and len(X) <= 5000:
                r = permutation_importance(pipe, X, y, n_repeats=5, random_state=42)
                importance = pd.DataFrame({
                    "feature": X.columns, "importance": r.importances_mean,
                }).sort_values("importance", ascending=False)
        except Exception:
            pass
        prog.progress(100, text=t("train_done"))
    st.session_state.results = results
    st.session_state.extras = extras
    st.session_state.notes = notes
    st.session_state.importance = importance
    st.session_state.log.append(f"Training done: best = {extras['best_name']}")
    best = results.iloc[0]
    col = "Test R2" if task == "regression" else "F1"
    score = float(best[col])
    if score < 0.3 and len(features) > 3 and not st.session_state.get("retried"):
        st.session_state.retried = True
        keep, _ranked = prune_features(df, features, target, task)
        with st.spinner(t("train_retry_spinner").format(
                n=len(keep), feats=", ".join(keep))):
            try:
                r2, n2, e2 = train_run(df[keep + [target]], target,
                                        st.session_state.clean_plan, task,
                                        st.session_state.model_keys or None,
                                        self_check=True, nn_config=nn_cfg,
                                        mode=learning_mode,
                                        metric_preference=metric_preference,
                                        interaction_specs=interaction_specs,
                                        source_name=source_name,
                                        hyperparams_overrides=manual_overrides)
                cvcol = r2.columns[1] if not r2.empty else None
                if cvcol and float(r2.iloc[0][cvcol]) > score:
                    st.session_state.features = keep
                    st.session_state.results = r2
                    st.session_state.notes = n2
                    st.session_state.extras = e2
                    st.session_state.log.append(f"Retry improved: {float(r2.iloc[0][cvcol]):.3f}")
            except Exception as e:
                st.session_state.log.append(f"Retry failed: {type(e).__name__}")
    st.rerun()


# ============================================================ PREDICT
def _predict_manual(pipeline, df_train, features, target, task, interactions=None):
    interactions = interactions or []
    ix_names = {s["name"] for s in interactions}
    base_features = [f for f in features if f not in ix_names]
    st.markdown(t("predict_manual_title"))
    values = {}
    cols = st.columns(2)
    for i, f in enumerate(base_features):
        series = df_train[f]
        with cols[i % 2]:
            kind = column_kind(series, f)
            if kind in ("num", "numunit"):
                v = pd.to_numeric(series, errors="coerce").dropna()
                if len(v):
                    mn, mx = float(v.min()), float(v.max())
                    med = float(v.median())
                    if mx > mn:
                        values[f] = st.number_input(f, value=med, key=f"pred_in_{f}")
                    else:
                        values[f] = med
                else:
                    values[f] = st.number_input(f, value=0.0, key=f"pred_in_{f}")
            elif kind in ("cat", "topcat", "bool"):
                uniq = series.dropna().astype(str).unique().tolist()[:200]
                values[f] = st.selectbox(f, uniq, key=f"pred_in_{f}")
            elif kind == "date":
                values[f] = st.date_input(f, key=f"pred_in_{f}")
            else:
                values[f] = st.text_area(f, "", height=80, key=f"pred_in_{f}")
    if st.button(t("predict_btn"), type="primary", use_container_width=False,
                 key="predict_manual_btn"):
        row = pd.DataFrame([values])
        for spec in interactions:
            a, b, op = spec["a"], spec["b"], spec["op"]
            if a in row.columns and b in row.columns:
                if op == "ratio":
                    row[spec["name"]] = row[a] / row[b].replace(0, np.nan)
                elif op == "product":
                    row[spec["name"]] = row[a] * row[b]
                else:
                    row[spec["name"]] = row[a] - row[b]
        try:
            row = row[features]
        except KeyError as e:
            st.error(t("predict_missing_feature").format(e=e))
            return
        try:
            pred = pipeline.predict(row)[0]
        except Exception as e:
            st.error(t("predict_failed").format(err=f"{type(e).__name__}: {e}"))
            return
        conf = None
        if hasattr(pipeline, "predict_proba"):
            try:
                conf = float(pipeline.predict_proba(row)[0].max())
            except Exception:
                pass
        st.markdown(
            f'<div class="card" style="border-left:3px solid #fbbf24;padding:1.2rem 1.5rem;">'
            f'<h4 style="color:#fcd34d;">{t("predict_result")}</h4>'
            f'<p style="font-size:1.6rem;color:#e8eef7;font-weight:700;margin:0.4rem 0;">'
            f'{target} = {pred}</p>'
            + (f'<p style="color:#94a3b8;margin:0;">'
               f'{t("predict_confidence").format(c=conf)}</p>'
               if conf is not None else '')
            + f'</div>',
            unsafe_allow_html=True)


def _predict_batch(pipeline, features, target, task, interactions=None):
    uploaded = st.file_uploader(t("predict_upload_hint"),
                                 type=["csv", "xlsx", "xls"], key="predict_upload")
    if uploaded is None:
        st.caption(t("predict_upload_caption"))
        return
    os.makedirs("outputs", exist_ok=True)
    suffix = os.path.splitext(uploaded.name)[1]
    tmp = os.path.join("outputs", f"_predict_{uuid.uuid4().hex[:6]}{suffix}")
    with open(tmp, "wb") as f:
        f.write(uploaded.getbuffer())
    try:
        df_new, _ = load_table_with_notes(tmp)
    except Exception as e:
        st.error(f"Could not read the file: {type(e).__name__}: {e}")
        return
    interactions = interactions or []
    for spec in interactions:
        a, b, op = spec["a"], spec["b"], spec["op"]
        if spec["name"] in df_new.columns:
            continue
        if a in df_new.columns and b in df_new.columns:
            try:
                if op == "ratio":
                    df_new[spec["name"]] = df_new[a] / df_new[b].replace(0, np.nan)
                elif op == "product":
                    df_new[spec["name"]] = df_new[a] * df_new[b]
                else:
                    df_new[spec["name"]] = df_new[a] - df_new[b]
            except Exception:
                pass
    missing = [c for c in features if c not in df_new.columns]
    if missing:
        st.error(t("predict_missing").format(cols=", ".join(missing)))
        return
    X_new = df_new[features]
    try:
        preds = pipeline.predict(X_new)
    except Exception as e:
        st.error(t("predict_failed").format(err=f"{type(e).__name__}: {e}"))
        return
    out = df_new.copy()
    out[f"predicted_{target}"] = preds
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X_new)
            out["confidence"] = proba.max(axis=1).round(3)
        except Exception:
            pass
    st.success(t("predict_predicted_n").format(n=len(out)))
    st.dataframe(out.head(20), use_container_width=True)
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button(t("predict_download"), data=csv,
                       file_name=f"predictions_{os.path.basename(tmp)}.csv",
                       mime="text/csv")


def _stage_predict():
    extras = st.session_state.extras
    df_train = st.session_state.df
    features = extras["features"]
    target = extras["target"]
    task = extras["task"]
    pipeline = extras["best_pipeline"]
    interactions = extras.get("interactions") or []
    st.divider()
    st.markdown(t("results_pred_title"))
    st.caption(t("results_pred_caption").format(
        m=extras["best_name"], target=target))
    mode = st.radio(t("results_pred_method"),
                    [t("results_pred_manual"), t("results_pred_upload")],
                    horizontal=True, key="predict_mode")
    if mode == t("results_pred_manual"):
        _predict_manual(pipeline, df_train, features, target, task, interactions)
    else:
        _predict_batch(pipeline, features, target, task, interactions)


# ============================================================ RESULTS
def _metric_explanation(task, best, base):
    if task == "regression":
        r2 = float(best["Test R2"])
        return t("results_metric_help").format(
            r2=r2, pct=max(r2, 0) * 100, base=float(base["Test R2"]))
    f1 = float(best["F1"])
    return t("results_metric_help_clf").format(
        f1=f1, base=float(base["F1"]))


def _why_these_models(results, task, mode, learning_mode):
    n_train = st.session_state.get("extras", {}).get("n_train", 0)
    nn_cfg = st.session_state.get("nn_config", DEFAULT_NN_CONFIG)
    cat = (nlp_catalog(n_train) if mode == "nlp"
           else catalog(task, n_train, nn_config=nn_cfg, mode=learning_mode))
    name_to_key = {v["name"]: k for k, v in cat.items()}
    user_manual = st.session_state.get("mode") == "manual"
    user_keys = st.session_state.get("model_keys") or []
    specific = (st.session_state.get("extras") or {}).get("specific_reasons") or {}

    st.markdown(t("results_why_title"))
    st.caption(t("results_why_caption"))
    for _, row in results.iterrows():
        name = row["Model"]
        if name.startswith("Baseline"):
            st.markdown(
                f'<div class="card"><h4>{t("results_baseline")}</h4>'
                f'<p>{t("results_baseline_body")}</p></div>',
                unsafe_allow_html=True)
            continue
        key = name_to_key.get(name)
        user_picked = user_manual and key in user_keys
        tag = t("results_tag_user") if user_picked else t("results_tag_agent")
        pill_class = "purple" if user_picked else ""
        if user_picked:
            reason = t("results_user_prefix") + (
                cat.get(key, {}).get("blurb", "") if mode == "nlp" else REASON.get(key, ""))
        elif name in specific:
            reason = specific[name]
        else:
            reason = (cat.get(key, {}).get("blurb", "chosen for text classification.")
                      if mode == "nlp" else REASON.get(key, "chosen by the agent."))
        st.markdown(
            f'<div class="card"><h4>{name} <span class="pill {pill_class}">{tag}</span></h4>'
            f'<p>{reason}</p></div>',
            unsafe_allow_html=True)


def _render_roc_pr(task, extras):
    pipeline = extras.get("best_pipeline")
    X_te = extras.get("X_test")
    y_te = extras.get("y_test")
    if pipeline is None or X_te is None or y_te is None:
        return
    if not hasattr(pipeline, "predict_proba"):
        return

    classes = sorted(pd.Series(y_te).unique())
    if len(classes) < 2 or len(classes) > 8:
        return

    try:
        y_prob = pipeline.predict_proba(X_te)
    except Exception:
        return

    y_arr = np.asarray(y_te)
    st.divider()
    st.markdown(t("results_roc_title"))
    st.caption(t("results_roc_caption"))

    c1, c2 = st.columns(2)

    if len(classes) == 2:
        try:
            pos_idx = list(pipeline.classes_).index(classes[1])
        except Exception:
            pos_idx = 1
        p = y_prob[:, pos_idx]
        y_bin = (y_arr == classes[1]).astype(int)

        fpr, tpr, _ = roc_curve(y_bin, p)
        roc_auc = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(y_bin, p)
        pr_auc = auc(rec, prec)

        with c1:
            fig = px.line(x=fpr, y=tpr,
                          labels={"x": t("plot_fpr"), "y": t("plot_tpr")})
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dash", color="#94a3b8"))
            fig.update_traces(line=dict(color="#22d3ee", width=3))
            fig.update_layout(height=340,
                              title=f"ROC · AUC = {roc_auc:.3f}",
                              margin=dict(t=40, b=10),
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#b6c2d4"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.line(x=rec, y=prec,
                          labels={"x": t("plot_recall"), "y": t("plot_precision")})
            fig.update_traces(line=dict(color="#e879f9", width=3))
            fig.update_layout(height=340,
                              title=f"Precision-Recall · AUC = {pr_auc:.3f}",
                              margin=dict(t=40, b=10),
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#b6c2d4"))
            st.plotly_chart(fig, use_container_width=True)
    else:
        y_bin = label_binarize(y_arr, classes=classes)
        palette = ["#22d3ee", "#e879f9", "#fbbf24", "#4ade80",
                   "#f43f5e", "#a78bfa", "#67e8f9", "#fcd34d"]

        with c1:
            fig = go.Figure()
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                a = auc(fpr, tpr)
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{cls} (AUC {a:.2f})",
                    line=dict(color=palette[i % len(palette)], width=2)))
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dash", color="#94a3b8"))
            fig.update_layout(height=380,
                              title=t("plot_roc_title"),
                              margin=dict(t=40, b=10),
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#b6c2d4"),
                              legend=dict(font=dict(size=10)),
                              xaxis_title=t("plot_fpr"),
                              yaxis_title=t("plot_tpr"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure()
            for i, cls in enumerate(classes):
                prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
                a = auc(rec, prec)
                fig.add_trace(go.Scatter(
                    x=rec, y=prec, mode="lines",
                    name=f"{cls} (AUC {a:.2f})",
                    line=dict(color=palette[i % len(palette)], width=2)))
            fig.update_layout(height=380,
                              title=t("plot_pr_title"),
                              margin=dict(t=40, b=10),
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#b6c2d4"),
                              legend=dict(font=dict(size=10)),
                              xaxis_title=t("plot_recall"),
                              yaxis_title=t("plot_precision"))
            st.plotly_chart(fig, use_container_width=True)


def _render_residual_plot(extras):
    pipeline = extras.get("best_pipeline")
    X_te = extras.get("X_test")
    y_te = extras.get("y_test")
    if pipeline is None or X_te is None or y_te is None:
        return
    try:
        y_pred = pipeline.predict(X_te)
    except Exception:
        return
    y_true = np.asarray(y_te)
    residuals = y_true - y_pred

    st.markdown(t("results_resid_title"))
    st.caption(t("results_resid_caption"))

    c1, c2 = st.columns(2)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    with c1:
        fig = px.scatter(x=y_pred, y=y_true,
                         labels={"x": t("plot_predicted"), "y": t("plot_actual")})
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                      line=dict(dash="dash", color="#94a3b8"))
        fig.update_traces(marker=dict(color="#22d3ee", size=6, opacity=0.6))
        fig.update_layout(height=340, margin=dict(t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#b6c2d4"))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(x=y_pred, y=residuals,
                         labels={"x": t("plot_predicted"), "y": t("plot_residual")})
        fig.add_shape(type="line", x0=lo, y0=0, x1=hi, y1=0,
                      line=dict(dash="dash", color="#94a3b8"))
        fig.update_traces(marker=dict(color="#e879f9", size=6, opacity=0.6))
        fig.update_layout(height=340, margin=dict(t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#b6c2d4"))
        st.plotly_chart(fig, use_container_width=True)

    mean_r = float(np.mean(residuals))
    std_r = float(np.std(residuals))
    mae_r = float(np.mean(np.abs(residuals)))
    c1, c2, c3 = st.columns(3)
    with c1: ui.stat(f"{mean_r:+.3f}", t("results_mean_resid"))
    with c2: ui.stat(f"{std_r:.3f}", t("results_std_resid"))
    with c3: ui.stat(f"{mae_r:.3f}", t("results_mae_resid"))


def _stage_results():
    results = st.session_state.results
    extras = st.session_state.extras
    notes = st.session_state.notes
    task = st.session_state.task
    target = st.session_state.target
    mode = extras.get("mode", "tabular")
    learning_mode = extras.get("learning_mode", "ml")

    best = results.iloc[0]
    base = results.iloc[-1]
    cv_col = "CV R2" if task == "regression" else "CV F1"
    test_col = "Test R2" if task == "regression" else "F1"

    st.markdown(t("results_title"))

    nn = extras.get("nn_info")
    if nn:
        st.markdown(
            f'<div class="card" style="border-left:3px solid #fbbf24;">'
            f'<h4 style="color:#fcd34d;">{t("results_nn_title")}</h4>'
            f'<p><b>Model:</b> {nn["model_name"]} · <b>Device:</b> {nn["device"]}'
            + (f' · <b>Epochs trained:</b> {nn["epochs_trained"]}' if nn.get("epochs_trained") else "")
            + '</p>'
            f'<p><b>Architecture:</b> {nn["hidden_layers"]} · '
            f'activation={nn["activation"]} · dropout={nn["dropout"]}</p>'
            f'<p><b>Optimizer:</b> {nn["optimizer"]} · lr={nn["learning_rate"]}</p>'
            f'</div>',
            unsafe_allow_html=True)

    interactions = extras.get("interactions") or []
    if interactions:
        st.markdown(
            f'<div class="card" style="border-left:3px solid #e879f9;">'
            f'<h4 style="color:#f0a6fc;">{t("results_inter_title")}</h4>'
            + "".join(f'<p><code>{s["name"]}</code> = {s["a"]} {s["op"]} {s["b"]} '
                      f'— {s.get("reason", "")}</p>' for s in interactions)
            + '</div>', unsafe_allow_html=True)

    if mode == "nlp":
        st.markdown(
            f'<div class="card" style="border-left:3px solid #e879f9;">'
            f'<h4 style="color:#f0a6fc;">{t("results_nlp_title")}</h4>'
            f'<p>{t("results_nlp_text_cols").format(cols=", ".join(extras.get("text_columns", [])))}</p></div>',
            unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: ui.stat(best["Model"], t("results_best_model"))
    with c2: ui.stat(f"{float(best[cv_col]):.3f}", t("results_cv"))
    with c3: ui.stat(f"{float(best[test_col]):.3f}", t("results_test"))

    st.markdown(
        f'<div class="card"><h4>{t("results_what_numbers_title")}</h4>'
        f'<p>{_metric_explanation(task, best, base)}</p>'
        f'<p style="color:#94a3b8;font-size:0.85rem;margin-top:0.6rem;">'
        f'{t("results_metric_pref").format(m=extras.get("metric_preference", "balanced"))}'
        f'</p></div>', unsafe_allow_html=True)

    st.markdown(t("results_leaderboard"))
    st.dataframe(results, use_container_width=True)

    with st.expander(t("results_feat_eng")):
        if mode == "nlp":
            st.markdown("**Text** → TF-IDF. **Numeric** → median impute + scale. "
                        "**Categorical** → most-frequent impute + one-hot.")
        else:
            plan = engineering_summary(st.session_state.df[
                st.session_state.features + [st.session_state.target]])
            if not plan:
                st.info(t("results_no_conversions"))
            else:
                for line in plan: st.markdown(f"• {line}")

    fig = px.bar(results, x="Model", y=cv_col, color=cv_col,
                 color_continuous_scale=[[0, "#1c2540"], [0.5, "#22d3ee"], [1, "#e879f9"]],
                 text=cv_col)
    fig.update_layout(height=340, showlegend=False, margin=dict(t=20, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#b6c2d4"))
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    _why_these_models(results, task, mode, learning_mode)

    if task == "classification" and extras.get("per_class") is not None:
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(t("results_per_class"))
            st.dataframe(extras["per_class"], use_container_width=True)
        with c2:
            if extras.get("confusion") is not None:
                st.markdown(t("results_confusion"))
                cm = extras["confusion"]
                fig = px.imshow(cm.values,
                                x=[c.replace("Predicted ", "") for c in cm.columns],
                                y=[r.replace("Actual ", "") for r in cm.index],
                                color_continuous_scale=[[0, "#131a2e"], [1, "#22d3ee"]],
                                aspect="auto", text_auto=True)
                fig.update_layout(height=420, margin=dict(t=10, b=10),
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="#b6c2d4"))
                st.plotly_chart(fig, use_container_width=True)

        _render_roc_pr(task, extras)

    if task == "regression":
        with st.expander(t("results_adv_diag"), expanded=False):
            _render_residual_plot(extras)

    top_words = extras.get("top_words")
    if top_words:
        st.divider()
        st.markdown(t("results_top_words"))
        for cls, words in top_words.items():
            if not words: continue
            st.markdown(t("results_class_prefix").format(c=cls))
            cols = st.columns(min(5, len(words)))
            for i, w in enumerate(words[:10]):
                with cols[i % min(5, len(words))]:
                    st.markdown(
                        f'<div class="card" style="padding:0.5rem 0.7rem;text-align:center;">'
                        f'<p style="margin:0;color:#22d3ee;font-weight:600;">'
                        f'{w}</p></div>',
                        unsafe_allow_html=True)

    top_bigrams = extras.get("top_bigrams")
    if top_bigrams:
        st.divider()
        st.markdown(t("results_top_bigrams"))
        for cls, phrases in top_bigrams.items():
            if not phrases: continue
            st.markdown(t("results_class_prefix").format(c=cls))
            cols = st.columns(min(4, len(phrases)))
            for i, w in enumerate(phrases[:8]):
                with cols[i % min(4, len(phrases))]:
                    st.markdown(
                        f'<div class="card" style="padding:0.5rem 0.7rem;text-align:center;">'
                        f'<p style="margin:0;color:#e879f9;font-weight:600;">'
                        f'{w}</p></div>',
                        unsafe_allow_html=True)

    imp = st.session_state.get("importance")
    if imp is not None and len(imp):
        st.divider()
        st.markdown(t("results_feat_importance"))
        fig = px.bar(imp.head(15), x="importance", y="feature", orientation="h",
                     color="importance",
                     color_continuous_scale=[[0, "#1c2540"], [1, "#22d3ee"]])
        fig.update_layout(height=max(280, 26 * min(15, len(imp))),
                          showlegend=False, margin=dict(t=20, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#b6c2d4"))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # HYPERPARAMETER TUNING SECTION
    # ---------------------------------------------------------
    st.divider()
    st.markdown("### ⚙️ Fine-tune Top Models")
    
    top_models = results[~results["Model"].str.startswith("Baseline")].head(2)
    if len(top_models) == 0:
        st.info("No models to tune.")
    else:
        if mode == "nlp":
            cat = nlp_catalog(len(st.session_state.df))
        else:
            cat = catalog(task, len(st.session_state.df), mode=learning_mode, nn_config=st.session_state.get("nn_config", DEFAULT_NN_CONFIG))
        name_to_key = {v["name"]: k for k, v in cat.items()}
        applied_hp = extras.get("applied_hyperparams", {})

        for _, row in top_models.iterrows():
            m_name = row["Model"]
            key = name_to_key.get(m_name)
            if not key:
                continue
                
            with st.expander(f"Tune **{m_name}** (Current Score: {row[test_col]:.3f})", expanded=False):
                current_p = applied_hp.get(m_name, {})
                st.markdown("**Current Parameters:**")
                st.json(current_p)
                
                tune_mode = st.radio("Tuning Mode", ["Auto (Agent picks)", "Manual (I choose)"], key=f"tune_mode_{key}", horizontal=True)
                params_to_apply = {}
                
                if tune_mode == "Auto (Agent picks)":
                    st.caption("The agent will suggest new hyperparameters and retrain.")
                    if st.button(f"🤖 Ask Agent to Tune {m_name}", key=f"btn_auto_{key}"):
                        with st.spinner("Agent is thinking..."):
                            suggested = agent_tune_hyperparams(key, task, current_p)
                            st.session_state[f"suggested_hp_{key}"] = suggested
                        st.rerun()
                    
                    suggested = st.session_state.get(f"suggested_hp_{key}")
                    if suggested:
                        st.success(f"Agent suggests: {suggested}")
                        params_to_apply = suggested
                else:
                    st.caption("Adjust the sliders below to set your own hyperparameters.")
                    p_defs = hyperparams(key, task)
                    if not p_defs:
                        st.info("This model has no tunable hyperparameters.")
                    for p in p_defs:
                        # Safely handle None values
                        val = current_p.get(p["name"], p["default"])
                        if val is None:
                            val = p["default"]
                            
                        min_v = p["min"] if p.get("min") is not None else 0
                        max_v = p["max"] if p.get("max") is not None else 100
                        step_v = p["step"] if p.get("step") is not None else 1
                        
                        widget_key = f"tune_{key}_{p['name']}"
                        if p["kind"] == "int":
                            params_to_apply[p["name"]] = st.slider(
                                p["name"], int(min_v), int(max_v), int(val), int(step_v), 
                                key=widget_key, help=p.get("help", "")
                            )
                        elif p["kind"] == "float":
                            params_to_apply[p["name"]] = st.number_input(
                                p["name"], float(min_v), float(max_v), float(val), float(step_v), 
                                key=widget_key, help=p.get("help", "")
                            )
                        elif p["kind"] == "choice":
                            choices = p.get("choices", [])
                            if val not in choices:
                                val = choices[0] if choices else ""
                            params_to_apply[p["name"]] = st.selectbox(
                                p["name"], choices, index=choices.index(val), 
                                key=widget_key, help=p.get("help", "")
                            )

                if st.button(f"🚀 Retrain {m_name} with these parameters", key=f"btn_retrain_{key}", type="primary"):
                    if not params_to_apply and tune_mode == "Manual (I choose)":
                        st.warning("No parameters to apply.")
                    else:
                        with st.spinner(f"Retraining {m_name}..."):
                            df_run_tune = st.session_state.df[st.session_state.features + [st.session_state.target]]
                            try:
                                r2, n2, e2 = train_run(
                                    df_run_tune, st.session_state.target, st.session_state.clean_plan, task,
                                    [key], self_check=False, nn_config=st.session_state.get("nn_config", DEFAULT_NN_CONFIG),
                                    mode=learning_mode, metric_preference=extras.get("metric_preference", "balanced"),
                                    interaction_specs=extras.get("interactions", []), source_name=st.session_state.get("source_name", ""),
                                    hyperparams_overrides={key: params_to_apply}
                                )
                                new_row = r2[r2["Model"] == m_name].iloc[0]
                                idx = st.session_state.results.index[st.session_state.results["Model"] == m_name][0]
                                for col in st.session_state.results.columns:
                                    st.session_state.results.at[idx, col] = new_row[col]
                                st.session_state.results = st.session_state.results.sort_values(
                                    st.session_state.results.columns[1], ascending=False
                                ).reset_index(drop=True)
                                st.session_state.log.append(f"Retrained {m_name} with new hyperparameters.")
                                st.success(f"{m_name} retrained successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Retrain failed: {type(e).__name__}: {e}")
    # ---------------------------------------------------------

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(t("results_self_check"))
        for line in extras.get("selfcheck", []): st.info(line)
    with c2:
        st.markdown(t("results_plain_words"))
        for line in extras.get("explanations", []): st.markdown(f"• {line}")

    _stage_predict()

    with st.expander(t("results_pipeline_notes")):
        for n in notes: st.write("• " + n)

    st.divider()
    st.markdown(t("results_take_further"))
    c1, c2 = st.columns([2, 3])
    with c1:
        if st.button(t("results_tune_btn"), type="primary",
                     use_container_width=True):
            st.session_state.page = "tuning"
            st.rerun()
        st.caption(t("results_tune_caption"))
    with c2:
        ui.deployment_locked()

    st.divider()
    st.markdown(t("results_save_export"))
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button(t("results_save_btn"), type="primary",
                     use_container_width=True):
            try:
                folder = export_all(
                    st.session_state.source_path, target, task,
                    st.session_state.clean_plan, st.session_state.warnings,
                    st.session_state.load_notes, results, notes, extras)
                st.session_state.output_folder = folder
                st.success(t("results_saved_to").format(folder=folder))
                try:
                    from export import export_pdf, export_model_card
                    pdf_path = export_pdf(
                        folder, st.session_state.source_path, target, task,
                        st.session_state.clean_plan, st.session_state.warnings,
                        st.session_state.load_notes, results, notes, extras)
                    st.session_state["_pdf_path"] = pdf_path
                    export_model_card(folder, target, task, results, extras,
                                       st.session_state.get("load_notes") or [])
                except Exception:
                    st.session_state["_pdf_path"] = None
                st.rerun()
            except Exception as e:
                st.error(t("results_save_fail").format(
                    err=f"{type(e).__name__}: {e}"))
    with c2:
        st.download_button(t("results_csv_btn"),
                            data=results.to_csv(index=False),
                            file_name=f"leaderboard_{target}.csv",
                            mime="text/csv",
                            use_container_width=True)
    with c3:
        folder = st.session_state.get("output_folder")
        card_path = os.path.join(folder, "model_card.pdf") if folder else None
        if card_path and os.path.exists(card_path):
            with open(card_path, "rb") as f:
                st.download_button(t("results_card_btn"), data=f.read(),
                                    file_name="model_card.pdf",
                                    mime="application/pdf",
                                    use_container_width=True)
        else:
            st.button(t("results_card_btn"), disabled=True,
                      use_container_width=True,
                      help=t("results_card_help"))

    pdf_path = st.session_state.get("_pdf_path")
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            st.download_button(t("results_pdf_btn"), data=f.read(),
                                file_name=os.path.basename(pdf_path),
                                mime="application/pdf",
                                use_container_width=True)

    if st.button(t("results_try_another"), use_container_width=True):
        _reset_stages()
        for k in ("target", "task", "features", "mode", "model_keys",
                  "agent_target_suggestion", "metric_preference",
                  "interaction_proposals", "interaction_specs", "manual_params"):
            st.session_state.pop(k, None)
        st.rerun()


# ============================================================ ROUTER
def render_pipeline():
    if not auth.is_logged_in():
        ui.hero("Sign in required",
                "The Run Pipeline page needs an authenticated session.",
                pills=["restart required"])
        return

    # AUTO-DETECT MODALITY (and reset if dataset changed)
    if st.session_state.get("df") is not None:
        detected_modality = auto_detect_modality(st.session_state.df)
        if st.session_state.get("modality") != detected_modality:
            st.session_state.modality = detected_modality
            _reset_modality()  # Clear old target/features/results
            st.rerun()
    else:
        ui.hero("Run Pipeline", "Load a dataset in the Data Profiler first.",
                pills=["plan", "approve", "train", "review"])
        if st.button("🔎  Go to Data Profiler", type="primary"):
            st.session_state.page = "profile"; st.rerun()
        return

    modality = st.session_state.get("modality")
    if modality == "tabular" and not (auth.can_use("ml") or auth.can_use("dl")):
        st.error("Your account does not have ML/DL access.")
        return
    if modality == "nlp" and not auth.can_use("nlp"):
        st.error("Your account does not have NLP access.")
        return

    has_target = bool(st.session_state.get("target"))
    has_plan = bool(st.session_state.get("plan_approved"))
    has_results = "results" in st.session_state
    if not has_target and (has_plan or has_results):
        _reset_stages()
    if has_target and has_results:
        extras_feats = (st.session_state.get("extras") or {}).get("features") or []
        curr_feats = st.session_state.get("features") or []
        if set(extras_feats) != set(curr_feats):
            _reset_stages()

    if not st.session_state.get("source_name"):
        ui.hero("Run Pipeline", "Load a dataset in the Data Profiler first.",
                pills=["plan", "approve", "train", "review"])
        if st.button("🔎  Go to Data Profiler", type="primary"):
            st.session_state.page = "profile"; st.rerun()
        return

    if not st.session_state.get("target"):
        _stage_setup()
        return

    c1, c2, c3 = st.columns(3)
    with c1: ui.stat(st.session_state.source_name, "source")
    with c2: ui.stat(st.session_state.target, "target")
    with c3: ui.stat(len(st.session_state.features or []), "input columns")
    st.divider()

    if not st.session_state.get("plan_approved"):
        _stage_plan()
        st.divider()
        if st.button("← Change target or columns"):
            _reset_stages()
            for k in ("target", "task", "features", "mode", "model_keys",
                      "metric_preference", "interaction_proposals",
                      "interaction_specs", "manual_params"):
                st.session_state.pop(k, None)
            st.rerun()
    elif "results" not in st.session_state:
        _stage_train()
    else:
        _stage_results()


# ============================================================ TUNING
def render_tuning():
    ui.hero(t("tune_hero_title"), t("tune_hero_sub"),
            pills=[t("tune_pill_1"), t("tune_pill_2"), t("tune_pill_3")])

    df = st.session_state.get("df")
    source = st.session_state.get("source_name")
    if df is None or source is None:
        st.info(t("tune_load_first"))
        if st.button(t("tune_go_profile"), type="primary"):
            st.session_state.page = "profile"; st.rerun()
        return

    history = load_history()
    runs = [h for h in history if h.get("source") == source]
    if not runs:
        st.info(t("tune_no_runs").format(source=source))
        if st.button(t("tune_go_run"), type="primary"):
            st.session_state.page = "run"; st.rerun()
        return

    st.markdown(t("tune_history_title"))
    hist_df = pd.DataFrame([{
        "when": r["timestamp"], "target": r["target"], "task": r["task"],
        "features": r["n_features"], "best model": r["best_model"],
        "score": round(r["best_score"], 3),
        "mode": r.get("learning_mode", "ml"),
        "metric": r.get("metric_preference", "balanced"),
    } for r in runs])
    st.dataframe(hist_df, use_container_width=True)

    best_ever = max(runs, key=lambda r: r["best_score"])
    st.markdown(
        f'<div class="card" style="border-left:3px solid #4ade80;">'
        f'<h4>{t("tune_best_ever")}</h4>'
        f'<p>{t("tune_best_ever_body").format(model=best_ever["best_model"], score=best_ever["best_score"], task=best_ever["task"], target=best_ever["target"])}</p>'
        f'</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown(t("tune_last_run"))
    last = runs[-1]
    st.caption(t("tune_last_caption").format(
        target=last["target"], task=last["task"],
        n=last["n_features"], score=last["best_score"]))

    target_options = list(df.columns)
    default_t_idx = target_options.index(last["target"]) if last["target"] in target_options else 0
    new_target = st.selectbox(t("tune_target"), options=target_options,
                              index=default_t_idx, key="tune_target")

    usable, dropped = select_features(df, new_target)
    saved_features = [f for f in last["features"] if f in usable]
    new_features = st.multiselect(t("tune_inputs"),
                                   options=[c for c in df.columns if c != new_target],
                                   default=saved_features or usable,
                                   key="tune_features")
    for c, why in dropped:
        if c in new_features:
            st.warning(t("setup_feature_warn").format(c=c, why=why))

    new_metric = st.radio(
        t("tune_metric"),
        ["balanced", "precision", "recall"],
        index=["balanced", "precision", "recall"].index(last.get("metric_preference", "balanced")),
        horizontal=True, key="tune_metric")

    new_mode = st.radio(t("tune_family"), ["ml", "dl"],
                        index=0 if last.get("learning_mode", "ml") == "ml" else 1,
                        horizontal=True, key="tune_mode")

    mode_label = st.radio(t("tune_selection"),
                          [t("setup_mode_auto"), t("setup_mode_manual")],
                          key="tune_model_mode")
    model_keys = []
    nn_cfg = st.session_state.get("nn_config", DEFAULT_NN_CONFIG)
    task_guess, _, _ = task_evidence(df[new_target])
    if mode_label == t("setup_mode_manual"):
        cat = catalog(task_guess, int(len(df) * 0.8), nn_config=nn_cfg, mode=new_mode)
        labels = {k: v["name"] for k, v in cat.items()}
        model_keys = st.multiselect(t("tune_models"), options=list(labels.keys()),
                                     format_func=lambda k: labels[k],
                                     default=[k for k in labels if labels[k] == last["best_model"]] or
                                             list(labels.keys())[:3],
                                     key="tune_models")

    if st.button(t("tune_retrain"), type="primary"):
        plan = st.session_state.get("clean_plan") or {
            "drop_duplicates": False, "impute": {}, "outlier_columns": []
        }
        if not st.session_state.get("clean_plan"):
            with st.spinner("No saved plan in this session; generating a new one…"):
                from ml import get_plan, validate_plan, restrict_plan
                proposed = get_plan(df)
                clean, warnings = validate_plan(proposed, df)
                plan = restrict_plan(clean, new_features)
        with st.spinner(t("tune_spinner")):
            try:
                results, notes, extras = train_run(
                    df[new_features + [new_target]], new_target, plan,
                    task_guess, model_keys or None, self_check=True,
                    nn_config=nn_cfg, mode=new_mode,
                    metric_preference=new_metric,
                    interaction_specs=st.session_state.get("interaction_specs") or [],
                    source_name=source)
                st.session_state.results = results
                st.session_state.extras = extras
                st.session_state.notes = notes
                st.session_state.target = new_target
                st.session_state.task = task_guess
                st.session_state.features = new_features
                st.session_state.importance = None
                st.session_state.clean_plan = plan
            except ValueError as e:
                st.error(t("tune_fail").format(err=str(e)))
                return
        new_score = float(results.iloc[0]["Test R2" if task_guess == "regression" else "F1"])
        delta = new_score - last["best_score"]
        verb = "improved" if delta > 0.005 else ("regressed" if delta < -0.005 else "matched")
        color = "#4ade80" if delta > 0.005 else ("#f43f5e" if delta < -0.005 else "#94a3b8")
        st.markdown(
            f'<div class="card" style="border-left:3px solid {color};">'
            f'<h4>{t("tune_result_title")}</h4>'
            f'<p>{t("tune_result_body").format(model=results.iloc[0]["Model"], score=new_score, verb=verb, delta=delta)}</p>'
            f'</div>', unsafe_allow_html=True)
        if st.button(t("tune_open_results")):
            st.session_state.plan_approved = True
            st.session_state.page = "run"
            st.rerun()