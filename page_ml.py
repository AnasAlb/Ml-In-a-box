"""
ML pages: Home, Data Profiler, Run Pipeline, Tuning.
"""
import os
import io
import json
import zipfile
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
                auto_detect_modality, agent_tune_hyperparams)
from nlp import nlp_catalog
from llm import ask

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ============================================================ HELPERS
def _theme_colors():
    theme = st.session_state.get("theme", "light")
    if theme == "dark":
        return {"accent": "#00F0FF", "border": "#27272A", "bg": "#111111",
                "text": "#FFFFFF", "text_muted": "#A1A1AA",
                "up": "#34D399", "down": "#F43F5E"}
    return {"accent": "#D4AF37", "border": "#D4AF37", "bg": "#FFFFFF",
            "text": "#111827", "text_muted": "#6B7280",
            "up": "#059669", "down": "#DC2626"}


# ============================================================ HOME
def render_home():
    user = auth.current_user()
    col_left, col_right = st.columns([1.4, 1], gap="large")

    with col_left:
        st.markdown(
            '<div class="hero" style="text-align:left; padding: 3rem 2.5rem; '
            'margin-bottom: 0; min-height: 320px;">'
            f'<h1 style="font-size: 3.2rem;">{t("hero_title")}</h1>'
            f'<p style="font-size: 1.05rem;">{t("hero_sub")}</p>'
            '<div style="margin-top: 1.5rem;">'
            '<span class="pill">agentic</span>'
            '<span class="pill purple">explainable</span>'
            '<span class="pill">end-to-end</span>'
            '</div>'
            '</div>', unsafe_allow_html=True)

    with col_right:
        ui.render_robot("")

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button(f"🔎  {t('profile_btn')}", use_container_width=True,
                     type="primary", key="home_cta_profile"):
            st.session_state.page = "profile"; st.rerun()
    with c2:
        if st.button(f"📄  {t('home_cta_doc')}", use_container_width=True,
                     key="home_cta_doc"):
            st.session_state.page = "doc_tools"; st.rerun()
    with c3:
        if st.button(f"🤖  {t('home_cta_agent')}", use_container_width=True,
                     key="home_cta_agent"):
            st.session_state.page = "agent_builder"; st.rerun()

    st.markdown("<div style='height: 3rem;'></div>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(
            f'<div class="card"><h4>{t("home_card_1_title")}</h4>'
            f'<p>{t("home_card_1_body")}</p></div>',
            unsafe_allow_html=True)
    with f2:
        st.markdown(
            f'<div class="card"><h4>{t("home_card_2_title")}</h4>'
            f'<p>{t("home_card_2_body")}</p></div>',
            unsafe_allow_html=True)
    with f3:
        st.markdown(
            f'<div class="card"><h4>{t("home_card_3_title")}</h4>'
            f'<p>{t("home_card_3_body")}</p></div>',
            unsafe_allow_html=True)

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    status = torch_status()
    status_color = ("#34D399" if status["cuda"]
                    else "#F59E0B" if status["torch"] else "#9CA3AF")
    status_text = (t("home_gpu", name=status["device_name"]) if status["cuda"]
                   else t("home_cpu") if status["torch"] else t("home_no_torch"))
    user_info = ""
    if user:
        mods = ", ".join(m.upper() for m in user["modalities"]) or t("home_read_only")
        user_info = f" &nbsp;·&nbsp; {t('home_signed_in', name=user['display_name'])} ({mods})"

    c = _theme_colors()
    st.markdown(
        f'<div style="background: {c["bg"]}; border: 1px solid {c["border"]}; '
        f'border-radius: 12px; padding: 1rem 1.5rem; display: flex; '
        f'align-items: center; gap: 10px;">'
        f'<div style="width: 10px; height: 10px; border-radius: 50%; '
        f'background: {status_color};"></div>'
        f'<span style="font-size: 0.9rem;">{status_text}{user_info}</span></div>',
        unsafe_allow_html=True)

    ui.sisyphus()


# ============================================================ MODEL ZIP
def _build_model_zip(extras, target, task, results, clean_plan):
    import joblib
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        pipe_buf = io.BytesIO()
        joblib.dump(extras["best_pipeline"], pipe_buf)
        zf.writestr("model.joblib", pipe_buf.getvalue())
        meta = {
            "app": "ML in a Box", "format_version": 1, "target": target,
            "task": task, "best_model": extras.get("best_name"),
            "features": extras.get("features", []),
            "n_train": extras.get("n_train"), "n_test": extras.get("n_test"),
            "metric_preference": extras.get("metric_preference", "balanced"),
            "cleaning_plan": clean_plan,
            "interactions": [
                {k: v for k, v in s.items() if k != "_"}
                for s in (extras.get("interactions") or [])],
            "leaderboard": results.to_dict(orient="records"),
        }
        zf.writestr("metadata.json", json.dumps(meta, indent=2, default=str))
        readme = (
            f"ML in a Box — Model Bundle\n"
            f"===========================\n\n"
            f"Model:      {extras.get('best_name')}\n"
            f"Target:     {target} ({task})\n"
            f"Features:   {', '.join(extras.get('features', []))}\n\n"
            f"How to use:\n"
            f"1. Open ML in a Box, go to Data Profiler → Upload Model.\n"
            f"2. Drop this .zip in.\n"
            f"3. Use the prediction panel.\n")
        zf.writestr("README.txt", readme)
    return buf.getvalue()


def _load_model_zip(zip_bytes):
    import joblib
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        if "model.joblib" not in names:
            raise ValueError("Zip is missing model.joblib")
        if "metadata.json" not in names:
            raise ValueError("Zip is missing metadata.json")
        pipe_bytes = zf.read("model.joblib")
        meta = json.loads(zf.read("metadata.json").decode("utf-8"))
        pipeline = joblib.load(io.BytesIO(pipe_bytes))
        return pipeline, meta


# ============================================================ PROFILER HELPERS
def _kind_label(kind):
    return t(f"prof_kind_{kind}"), {
        "num": "blue", "numunit": "blue",
        "cat": "purple", "topcat": "purple", "bool": "purple",
        "date": "amber", "list": "amber", "text": "amber",
        "id": "red", "empty": "red"}.get(kind, "blue")


def _sample_block(df, n=5):
    lines = []
    for c in df.columns:
        samples = df[c].dropna().astype(str).head(n).tolist()
        lines.append(f"- {c} (dtype {df[c].dtype})\n  samples: {samples}")
    return "\n".join(lines)


def _describe_all_columns(df):
    source = st.session_state.get("source_name", "")
    key = f"_col_desc::{source}::{tuple(df.columns)}"
    cache = st.session_state.setdefault("_col_desc_cache", {})
    if key in cache:
        return cache[key]
    prompt = (
        "For each column below, give ONE short sentence (max 20 words) "
        "saying what it likely represents. Use the name, dtype, samples only.\n"
        "Return ONLY a JSON object {column: description}.\n\n" + _sample_block(df))
    result = {c: "unclear" for c in df.columns}
    try:
        reply = (ask(prompt) or "").strip()
        s, e = reply.find("{"), reply.rfind("}")
        if s != -1 and e != -1:
            data = json.loads(reply[s:e + 1])
            for c in df.columns:
                v = data.get(c)
                if isinstance(v, str) and 0 < len(v) <= 200:
                    result[c] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    cache[key] = result
    return result


def _render_column_card(name, series, info, description, kind):
    label, color = _kind_label(kind)
    conversion = conversion_note(series, name)
    missing_pct = info.get("missing_pct", 0)
    unique = info.get("unique", 0)
    if missing_pct < 1:
        miss_pill = f'<span class="pill">{t("prof_missing_none")}</span>'
    elif missing_pct < 30:
        miss_pill = f'<span class="pill amber">{t("prof_missing_some", p=missing_pct)}</span>'
    else:
        miss_pill = f'<span class="pill red">{t("prof_missing_some", p=missing_pct)}</span>'
    conv = (f'<p style="font-size:0.85rem;margin-top:0.5rem;">'
            f'<b>{t("prof_auto_conversion")}</b> {conversion}</p>'
            if conversion else "")
    st.markdown(
        f'<div class="card"><h4>{name} '
        f'<span class="pill {color}">{label}</span>{miss_pill}'
        f'<span class="pill">{t("prof_dtype", t=info["type"])}</span>'
        f'<span class="pill">{t("prof_distinct", n=unique)}</span></h4>'
        f'<p><b>{t("prof_likely_meaning")}</b> {description}</p>{conv}</div>',
        unsafe_allow_html=True)


def _dataset_chat_ctx(df, descs):
    lines = [f"{len(df)} rows, {len(df.columns)} columns."]
    for c in df.columns:
        s = df[c]
        parts = [f"'{c}'", f"dtype {s.dtype}", f"{s.nunique()} distinct"]
        miss = round(s.isna().mean() * 100, 1)
        if miss:
            parts.append(f"{miss}% missing")
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            try:
                parts.append(f"range {s.min():.3g} to {s.max():.3g}")
            except Exception:
                pass
        d = descs.get(c, "")
        if d and d != "unclear":
            parts.append(f"meaning: {d}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _dataset_answer(question, ctx):
    prompt = ("You are a data analyst. Answer the user's question about the "
              "dataset in 3-6 sentences using ONLY the facts given.\n\n"
              f"--- PROFILE ---\n{ctx}\n--- END ---\n\nQuestion: {question}")
    try:
        return (ask(prompt, temperature=0.2) or "").strip()
    except Exception as e:
        return f"(Assistant unavailable: {type(e).__name__})"


def _render_dataset_chat():
    st.divider()
    st.markdown(f"### {t('prof_chat_title')}")
    st.caption(t("prof_chat_caption"))
    df = st.session_state.df
    descs = _describe_all_columns(df)
    ctx = _dataset_chat_ctx(df, descs)

    if "prof_chat" not in st.session_state:
        st.session_state.prof_chat = []

    suggestions = t("prof_chat_suggested")
    if isinstance(suggestions, list) and suggestions:
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
        border = "#00F0FF" if role == "user" else "#D4AF37"
        name = t("prof_you") if role == "user" else t("prof_assistant")
        st.markdown(
            f'<div class="card" style="border-left:3px solid {border};">'
            f'<p style="font-weight:600;margin-bottom:0.3rem;">{name}</p>'
            f'<p>{safe}</p></div>', unsafe_allow_html=True)

    with st.form("prof_chat_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            q = st.text_input(t("prof_chat_input"),
                              placeholder=t("prof_chat_placeholder"),
                              label_visibility="collapsed")
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
        _render_column_card(c, df[c], info["columns"][c],
                            descs.get(c, "unclear"), kinds[c])

    st.divider()
    st.markdown(f"### {t('prof_feature_eng_title')}")
    plan = engineering_summary(df)
    if not plan:
        st.info(t("prof_no_conversions"))
    else:
        for line in plan:
            st.markdown(
                f'<div class="card" style="padding:0.7rem 1rem;">'
                f'<p style="margin:0;">🧪 {line}</p></div>',
                unsafe_allow_html=True)

    _render_dataset_chat()

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button(t("prof_continue_btn"), type="primary",
                     use_container_width=True, key="prof_to_run"):
            st.session_state.page = "run"; st.rerun()
    with c2:
        if st.button(t("prof_reupload_btn"), use_container_width=True,
                     key="prof_reupload"):
            for k in ("df", "source_path", "source_name", "load_notes",
                      "profile", "_col_desc_cache", "target", "task",
                      "features", "mode", "model_keys", "proposed_plan",
                      "clean_plan", "warnings", "plan_approved", "results",
                      "extras", "notes", "advice", "importance",
                      "output_folder", "retried", "agent_target_suggestion",
                      "learning_mode", "metric_preference",
                      "interaction_proposals", "interaction_specs",
                      "_pdf_path", "prof_chat", "manual_params",
                      "enhancement_specs", "_show_enhance_ui"):
                st.session_state.pop(k, None)
            st.session_state.log = []
            st.rerun()


# ============================================================ LOADED MODEL
def _render_loaded_model_predictor(pipeline, meta):
    features = meta.get("features", [])
    target = meta.get("target", "target")
    c = _theme_colors()
    st.markdown(t("prof_model_predict_title"))
    st.caption(t("prof_model_predict_caption"))

    values = {}
    cols = st.columns(2)
    for i, f in enumerate(features):
        with cols[i % 2]:
            values[f] = st.text_input(f, value="", key=f"loaded_pred_{f}")

    cp1, cp2 = st.columns(2)
    with cp1:
        predict_clicked = st.button(t("prof_model_predict_btn"), type="primary",
                                     key="loaded_predict_btn",
                                     use_container_width=True)
    with cp2:
        explain_clicked = st.button(t("prof_model_explain_btn"), type="primary",
                                     key="loaded_explain_btn",
                                     use_container_width=True)

    if predict_clicked or explain_clicked:
        try:
            row = pd.DataFrame([values])
            for col in row.columns:
                try:
                    row[col] = pd.to_numeric(row[col])
                except Exception:
                    pass
            pred = pipeline.predict(row)[0]
            st.markdown(
                f'<div class="card" style="border-left:3px solid {c["accent"]};">'
                f'<h4>{t("prof_model_prediction")}</h4>'
                f'<p style="font-size:1.6rem;font-weight:800;color:{c["accent"]};">'
                f'{target} = {pred}</p></div>',
                unsafe_allow_html=True)
            if explain_clicked:
                _render_row_xai(pipeline, row, features)
        except Exception as e:
            st.error(t("predict_failed", err=f"{type(e).__name__}: {e}"))


# ============================================================ PROFILE
def render_profile():
    ui.hero(t("prof_hero_title"), t("prof_hero_sub"),
            pills=[t("prof_pill_1"), t("prof_pill_2"), t("prof_pill_3")])

    tab_data, tab_model = st.tabs([
        t("prof_upload_dataset_tab"), t("prof_upload_model_tab")])

    with tab_data:
        if st.session_state.get("df") is not None:
            _render_uploaded_profile()
            return
        uploaded = st.file_uploader(
            t("prof_upload_label"),
            type=["csv", "xlsx", "xls"],
            key="profile_dataset_uploader")
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
            st.error(f"Could not read: {type(e).__name__}: {e}")
            return
        st.session_state.df = df
        st.session_state.source_path = tmp
        st.session_state.source_name = uploaded.name
        st.session_state.load_notes = notes
        st.session_state.profile = profile_df(df)
        st.session_state.log = [
            f"Ingested {len(df)} rows × {len(df.columns)} cols from {uploaded.name}"]
        st.rerun()

    with tab_model:
        st.markdown(
            f'<div class="card">'
            f'<h4>{t("prof_model_intro_title")}</h4>'
            f'<p>{t("prof_model_intro_body")}</p>'
            f'</div>', unsafe_allow_html=True)

        up = st.file_uploader(
            t("prof_model_drop_label"),
            type=["zip"],
            key="profile_model_uploader_visible",
            help="Upload a .zip bundle downloaded from the Results page.")

        if up is None:
            st.info(t("prof_model_waiting"))
            return

        try:
            pipeline, meta = _load_model_zip(up.getvalue())
        except Exception as e:
            st.error(t("prof_model_read_fail", err=f"{type(e).__name__}: {e}"))
            st.caption(t("prof_model_hint"))
            return

        c1, c2, c3 = st.columns(3)
        with c1: ui.stat(meta.get("best_model", "?"), t("prof_model_best"))
        with c2: ui.stat(meta.get("target", "?"), t("prof_model_target"))
        with c3: ui.stat(meta.get("task", "?"), t("prof_model_task"))

        st.markdown(
            f'<div class="card"><h4>{t("prof_model_features_title")}</h4>'
            f'<p>{", ".join(meta.get("features", [])) or "—"}</p></div>',
            unsafe_allow_html=True)

        st.session_state["_loaded_model"] = pipeline
        st.session_state["_loaded_model_meta"] = meta
        st.success(t("prof_model_loaded"))
        _render_loaded_model_predictor(pipeline, meta)


# ============================================================ XAI — GLOBAL
def _pipeline_inner_model(pipeline):
    if hasattr(pipeline, "named_steps"):
        return pipeline.named_steps.get("model", pipeline)
    if hasattr(pipeline, "pipe_"):
        return pipeline.pipe_.named_steps.get("model", pipeline)
    return pipeline


def _pipeline_transform(pipeline, X):
    try:
        if hasattr(pipeline, "named_steps"):
            X2 = X.copy()
            for step in list(pipeline.named_steps.keys())[:-1]:
                X2 = pipeline.named_steps[step].transform(X2)
            return X2
        if hasattr(pipeline, "pipe_"):
            return pipeline.pipe_.named_steps["prep"].transform(X)
    except Exception:
        pass
    return X


def _pipeline_feature_names(pipeline, fallback):
    try:
        if hasattr(pipeline, "named_steps"):
            prep = pipeline.named_steps.get("prep")
            if prep is not None and hasattr(prep, "get_feature_names_out"):
                return list(prep.get_feature_names_out())
        if hasattr(pipeline, "pipe_"):
            prep = pipeline.pipe_.named_steps.get("prep")
            if prep is not None and hasattr(prep, "get_feature_names_out"):
                return list(prep.get_feature_names_out())
    except Exception:
        pass
    return list(fallback)


def _compute_shap_global(pipeline, X_test, fallback_features):
    if not SHAP_AVAILABLE:
        return None, None
    try:
        model = _pipeline_inner_model(pipeline)
        X_s = (X_test.sample(min(len(X_test), 200), random_state=42)
               if len(X_test) > 200 else X_test)
        X_t = _pipeline_transform(pipeline, X_s)
        sv = None
        if hasattr(model, "feature_importances_"):
            sv = shap.TreeExplainer(model).shap_values(X_t)
        elif hasattr(model, "coef_"):
            sv = shap.LinearExplainer(model, X_t).shap_values(X_t)
        if sv is None:
            return None, None
        arr = np.array(sv)
        if arr.ndim == 3:
            arr = np.mean(np.abs(arr), axis=0)
        if arr.ndim == 2:
            mean_abs = np.mean(np.abs(arr), axis=0)
        else:
            mean_abs = np.abs(arr)
        names = _pipeline_feature_names(pipeline, fallback_features)
        if len(names) != len(mean_abs):
            names = [f"feature_{i}" for i in range(len(mean_abs))]
        return names, mean_abs
    except Exception:
        return None, None


def _render_global_xai(extras, X_test):
    pipeline = extras.get("best_pipeline")
    if pipeline is None or X_test is None or len(X_test) == 0:
        return
    c = _theme_colors()
    accent = c["accent"]

    st.markdown(
        f'<div style="background: {c["bg"]}; border: 2px solid {accent}; '
        f'border-radius: 20px; padding: 2rem; margin: 2rem 0;">'
        f'<h2 style="color: {accent}; margin: 0 0 0.5rem 0;">'
        f'{t("xai_global_title")}</h2>'
        f'<p style="margin: 0; color: {c["text_muted"]}; font-size: 0.95rem;">'
        f'{t("xai_global_body")}</p></div>',
        unsafe_allow_html=True)

    names, mean_abs = _compute_shap_global(pipeline, X_test, extras["features"])

    if names is not None and mean_abs is not None:
        df_imp = pd.DataFrame({
            "feature": names, "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).head(15).iloc[::-1]
        fig = px.bar(df_imp, x="mean_abs_shap", y="feature",
                     orientation="h", color="mean_abs_shap",
                     color_continuous_scale=[[0, c["border"]], [1, accent]],
                     text="mean_abs_shap")
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(
            height=max(360, 28 * len(df_imp)), showlegend=False,
            coloraxis_showscale=False, margin=dict(t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=c["text_muted"], size=12),
            xaxis_title="Mean |SHAP value|")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(t("xai_global_caption"))
    else:
        st.caption(t("xai_shap_unavailable"))
        imp = st.session_state.get("importance")
        if imp is not None and len(imp):
            fig = px.bar(imp.head(15), x="importance", y="feature",
                         orientation="h", color="importance",
                         color_continuous_scale=[[0, c["border"]], [1, accent]])
            fig.update_layout(
                height=max(360, 28 * min(15, len(imp))),
                showlegend=False, coloraxis_showscale=False,
                margin=dict(t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=c["text_muted"]))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)


# ============================================================ XAI — ROW
def _render_row_xai(pipeline, X_row, features):
    c = _theme_colors()
    st.markdown(t("xai_row_title"))
    st.caption(t("xai_row_caption"))
    if not SHAP_AVAILABLE:
        st.warning(t("xai_row_no_shap"))
        return
    try:
        model = _pipeline_inner_model(pipeline)
        X_t = _pipeline_transform(pipeline, X_row)
        sv = None
        if hasattr(model, "feature_importances_"):
            sv = shap.TreeExplainer(model).shap_values(X_t)
        elif hasattr(model, "coef_"):
            sv = shap.LinearExplainer(model, X_t).shap_values(X_t)
        if sv is None:
            st.info(t("xai_row_unsupported"))
            return
        arr = np.array(sv)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim > 1:
            arr = arr[0]
        names = _pipeline_feature_names(pipeline, features)
        if len(names) != len(arr):
            names = [f"feature_{i}" for i in range(len(arr))]
        df = pd.DataFrame({"feature": names, "shap": arr})
        df = df.reindex(df["shap"].abs().sort_values(ascending=False).index).head(15).iloc[::-1]
        fig = px.bar(df, x="shap", y="feature", orientation="h", color="shap",
                     color_continuous_scale=[[0, c["down"]], [0.5, c["border"]],
                                              [1, c["up"]]])
        fig.update_layout(height=max(360, 28 * len(df)), showlegend=False,
                          coloraxis_showscale=False, margin=dict(t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color=c["text_muted"], size=12),
                          xaxis_title="SHAP value for this prediction")
        fig.add_vline(x=0, line=dict(color=c["text_muted"], width=1))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(t("xai_row_compute_error", err=f"{type(e).__name__}: {e}"))


# ============================================================ NN PANEL
def _advanced_nn_panel():
    cfg = st.session_state.setdefault("nn_config", dict(DEFAULT_NN_CONFIG))
    status = torch_status()
    if status["cuda"]:
        st.success(t("nn_gpu_ready", name=status["device_name"]))
    elif status["torch"]:
        st.warning(t("nn_cpu_warning"))
    else:
        st.info(t("nn_no_torch"))

    if status["torch"]:
        opts = [t("nn_sklearn"), t("nn_pytorch")]
        fw = st.radio(t("nn_framework_label"), opts,
                      index=0 if cfg.get("framework", "sklearn") == "sklearn" else 1,
                      horizontal=True, key="nn_framework")
        cfg["framework"] = "sklearn" if fw == t("nn_sklearn") else "torch"
    else:
        cfg["framework"] = "sklearn"
    use_torch = cfg["framework"] == "torch" and status["torch"]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{t('nn_architecture')}**")
        layers_str = st.text_input(
            t("nn_hidden_layers"),
            value=", ".join(str(x) for x in cfg.get("hidden_layers", [64, 32])),
            key="nn_layers")
        try:
            cfg["hidden_layers"] = [max(4, int(x.strip()))
                                     for x in layers_str.split(",") if x.strip()]
            if not cfg["hidden_layers"]:
                cfg["hidden_layers"] = [64, 32]
        except ValueError:
            st.warning(t("nn_layers_error"))
        cfg["activation"] = st.selectbox(
            t("nn_activation"), ["relu", "tanh", "logistic", "identity"],
            index=["relu", "tanh", "logistic", "identity"].index(
                cfg.get("activation", "relu")), key="nn_act")
        if use_torch:
            cfg["dropout"] = st.slider(t("nn_dropout"), 0.0, 0.5,
                                        float(cfg.get("dropout", 0.0)), 0.05,
                                        key="nn_drop")
    with c2:
        st.markdown(f"**{t('nn_training')}**")
        if use_torch:
            cfg["optimizer"] = st.selectbox(
                t("nn_optimizer"), ["adam", "sgd", "adamw", "rmsprop"],
                index=["adam", "sgd", "adamw", "rmsprop"].index(
                    cfg.get("optimizer", "adam")), key="nn_opt")
        else:
            cfg["optimizer"] = "adam"
        cfg["learning_rate"] = st.number_input(
            t("nn_lr"), min_value=1e-5, max_value=1.0,
            value=float(cfg.get("learning_rate", 0.001)),
            step=0.0001, format="%.4f", key="nn_lr")
        cfg["weight_decay"] = st.number_input(
            t("nn_wd"), min_value=0.0, max_value=1.0,
            value=float(cfg.get("weight_decay", 0.0001)),
            step=0.0001, format="%.4f", key="nn_wd")
        if use_torch:
            cfg["batch_size"] = st.number_input(
                t("nn_batch"), min_value=4, max_value=512,
                value=int(cfg.get("batch_size", 32)), step=4, key="nn_bs")
        else:
            cfg["batch_size"] = 32
        cfg["max_epochs"] = st.number_input(
            t("nn_epochs"), min_value=10, max_value=2000,
            value=int(cfg.get("max_epochs", 100)), step=10, key="nn_ep")
        cfg["early_stopping"] = st.checkbox(
            t("nn_early"), value=bool(cfg.get("early_stopping", True)),
            key="nn_es")
    st.session_state.nn_config = cfg


# ============================================================ MANUAL HYPERPARAMS
def _render_manual_hyperparams(model_keys, task):
    if not model_keys:
        return {}
    st.markdown(
        '<div class="card"><h4>⚙️ Hyperparameters</h4>'
        '<p>You picked these by hand, so you can tune each one.</p></div>',
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
                    wk = f"hp_{key}_{p['name']}"
                    if p["kind"] == "int":
                        overrides[key][p["name"]] = st.slider(
                            p["name"], int(p["min"]), int(p["max"]),
                            int(p["default"]), int(p["step"]),
                            help=p.get("help", ""), key=wk)
                    elif p["kind"] == "float":
                        overrides[key][p["name"]] = st.number_input(
                            p["name"], float(p["min"]), float(p["max"]),
                            float(p["default"]), float(p["step"]),
                            help=p.get("help", ""), key=wk)
                    elif p["kind"] == "choice":
                        overrides[key][p["name"]] = st.selectbox(
                            p["name"], p["choices"],
                            index=p["choices"].index(p["default"])
                            if p["default"] in p["choices"] else 0,
                            help=p.get("help", ""), key=wk)
    return overrides


# ============================================================ ENHANCEMENTS
def _render_enhancement_panel(df, features, target):
    numeric_feats = [f for f in features
                     if pd.api.types.is_numeric_dtype(df[f])]
    if not numeric_feats:
        return []

    heading_col, help_col = st.columns([9, 1])
    with heading_col:
        st.markdown(f"### {t('setup_step45')}")
    with help_col:
        with st.popover("❓", help=t("setup_enhance_tooltip")):
            st.markdown(f"**{t('setup_enhance_when')}**")
            st.markdown(f"- {t('setup_enhance_square')}")
            st.markdown(f"- {t('setup_enhance_sqrt')}")
            st.markdown(f"- {t('setup_enhance_log')}")
            st.markdown(f"- {t('setup_enhance_cube')}")

    c = _theme_colors()
    st.markdown(
        f'<div class="card" style="border-left:3px solid {c["accent"]};">'
        f'<p>{t("setup_enhance_intro")}</p></div>',
        unsafe_allow_html=True)

    specs = st.session_state.get("enhancement_specs") or []

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        col = st.selectbox(t("setup_enhance_feature"),
                            numeric_feats, key="enh_col")
    with c2:
        op = st.selectbox(t("setup_enhance_transform"),
                           ["square", "sqrt", "log1p", "cube"], key="enh_op")
    with c3:
        st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        if st.button(t("setup_enhance_add"), key="enh_add",
                     use_container_width=True):
            specs.append({"col": col, "op": op, "name": f"{col}_{op}"})
            st.session_state["enhancement_specs"] = specs
            st.rerun()

    if specs:
        st.markdown(t("setup_enhance_active"))
        for i, s in enumerate(specs):
            cc1, cc2 = st.columns([5, 1])
            with cc1:
                st.markdown(f"• `{s['name']}` = {s['op']}(`{s['col']}`)")
            with cc2:
                if st.button(t("setup_enhance_remove"), key=f"enh_rm_{i}"):
                    specs.pop(i)
                    st.session_state["enhancement_specs"] = specs
                    st.rerun()
    return specs


def _apply_enhancements(df, specs):
    df = df.copy()
    for s in specs:
        col, op = s["col"], s["op"]
        if col not in df.columns:
            continue
        try:
            v = pd.to_numeric(df[col], errors="coerce")
            if op == "square":
                df[s["name"]] = v ** 2
            elif op == "sqrt":
                df[s["name"]] = np.sqrt(v.clip(lower=0))
            elif op == "log1p":
                df[s["name"]] = np.log1p(v.clip(lower=0))
            elif op == "cube":
                df[s["name"]] = v ** 3
        except Exception:
            continue
    return df


# ============================================================ RESET
def _reset_stages():
    for k in ("plan_approved", "proposed_plan", "clean_plan", "warnings",
              "results", "extras", "notes", "advice", "importance",
              "output_folder", "retried", "_pdf_path", "_tune_result",
              "_agent_suggested_hp"):
        st.session_state.pop(k, None)


def _reset_modality():
    _reset_stages()
    for k in ("target", "task", "features", "mode", "model_keys",
              "learning_mode", "metric_preference", "interaction_proposals",
              "interaction_specs", "manual_params", "enhancement_specs",
              "_show_enhance_ui"):
        st.session_state.pop(k, None)


# ============================================================ SETUP
def _stage_setup():
    df = st.session_state.df
    modality = st.session_state.get("modality", "tabular")
    can_ml = auth.can_use("ml")
    can_dl = auth.can_use("dl")

    if modality == "nlp":
        st.markdown(t("setup_nlp_step1"))
        st.divider()
    else:
        st.markdown(t("setup_ml_step1"))

        use_dl = st.session_state.get("learning_mode") == "dl" and can_dl

        with st.expander(t("setup_advanced_dl_toggle"), expanded=False):
            st.caption(t("setup_advanced_dl_help"))
            use_dl = st.checkbox("Enable Deep Learning", value=use_dl,
                                  key="setup_dl_checkbox",
                                  disabled=not can_dl)

        st.session_state.learning_mode = "dl" if use_dl else "ml"

        mode_title = (t("setup_dl_mode_title") if use_dl
                      else t("setup_ml_mode_title"))
        mode_body = (t("setup_dl_mode_body") if use_dl
                     else t("setup_ml_mode_body"))
        c = _theme_colors()
        st.markdown(
            f'<div class="card" style="border-left:3px solid {c["accent"]};">'
            f'<h4>{mode_title}</h4><p>{mode_body}</p></div>',
            unsafe_allow_html=True)
        if use_dl:
            with st.expander(t("setup_nn_expander"), expanded=True):
                _advanced_nn_panel()
        st.divider()
        st.markdown(t("setup_step2"))

    st.markdown(
        f'<div class="card"><h4>{t("setup_agent_card_title")}</h4>'
        f'<p>{t("setup_agent_card_body")}</p></div>',
        unsafe_allow_html=True)
    if st.button(t("setup_agent_btn"), key="setup_agent_btn"):
        with st.spinner(t("setup_agent_reading")):
            rec = recommend_target(df)
        if rec:
            st.session_state["agent_target_suggestion"] = rec
        st.rerun()

    suggestion = st.session_state.get("agent_target_suggestion")
    if suggestion:
        st.success(t("setup_agent_suggests",
                     t=suggestion["target"], task=suggestion["task"],
                     reason=suggestion["reason"]))

    options = list(df.columns)
    default_idx = 0
    if suggestion and suggestion["target"] in options:
        default_idx = options.index(suggestion["target"])
    target = st.selectbox(t("setup_target_label"), options=options,
                          index=default_idx, key="target_pick")
    guess, conf, reason = task_evidence(df[target])
    st.info(t("setup_reading", task=guess, conf=conf, reason=reason))

    usable, dropped = select_features(df, target)
    features = st.multiselect(t("setup_features_label"),
                               options=[c for c in df.columns if c != target],
                               default=usable, key="features_pick")
    for c, why in dropped:
        if c in features:
            st.warning(t("setup_feature_warn", c=c, why=why))

    text_chosen = [c for c in features if column_kind(df[c], c) == "text"]
    if modality == "nlp" and not text_chosen:
        st.error(t("setup_nlp_no_text"))

    metric_preference = "balanced"
    if guess == "classification":
        st.markdown(t("setup_step3"))
        choice = st.radio(
            t("setup_metric_label"),
            [t("setup_metric_balanced"),
             t("setup_metric_precision"),
             t("setup_metric_recall")],
            index=0, key="metric_preference_radio")
        if choice == t("setup_metric_precision"):
            metric_preference = "precision"
        elif choice == t("setup_metric_recall"):
            metric_preference = "recall"
        st.session_state.metric_preference = metric_preference

    st.markdown(t("setup_step4"))
    if st.button(t("setup_inter_btn"), key="setup_inter_btn"):
        numeric_chosen = [c for c in features
                          if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_chosen) < 2:
            st.info(t("setup_inter_need_two"))
        else:
            with st.spinner(t("setup_inter_thinking")):
                proposals = agent_propose_interactions(df, features, target, max_n=3)
            st.session_state["interaction_proposals"] = proposals
        st.rerun()

    proposals = st.session_state.get("interaction_proposals") or []
    selected_interactions = []
    if proposals:
        for i, p in enumerate(proposals):
            keep = st.checkbox(
                f"`{p['name']}` = {p['a']} {p['op']} {p['b']} — {p.get('reason', '')}",
                value=True, key=f"ix_prop_{i}")
            if keep:
                selected_interactions.append(p)
    st.session_state.interaction_specs = selected_interactions

    enhancements = _render_enhancement_panel(df, features, target)
    st.session_state["enhancement_specs"] = enhancements

    st.markdown(t("setup_step5"))
    mode_label = st.radio(t("setup_mode_label"),
                          [t("setup_mode_auto"), t("setup_mode_manual")],
                          key="model_mode_radio")
    mode = "auto" if mode_label == t("setup_mode_auto") else "manual"
    model_keys = []
    manual_overrides = {}
    if mode == "manual":
        learning_mode = (st.session_state.get("learning_mode", "ml")
                         if modality == "tabular" else "all")
        nn_cfg = (st.session_state.get("nn_config", DEFAULT_NN_CONFIG)
                  if modality == "tabular" else None)
        cat = (nlp_catalog(int(len(df) * 0.8)) if modality == "nlp"
               else catalog(guess, int(len(df) * 0.8),
                            nn_config=nn_cfg, mode=learning_mode))
        labels = {k: v["name"] for k, v in cat.items()}
        model_keys = st.multiselect(t("setup_manual_label"),
                                     options=list(labels.keys()),
                                     format_func=lambda k: labels[k],
                                     default=list(labels.keys())[:3],
                                     key="manual_models_pick")
        if modality == "tabular":
            manual_overrides = _render_manual_hyperparams(model_keys, guess)
    st.session_state.manual_params = manual_overrides

    if not (target and features):
        st.info(t("setup_pick_hint"))
        return
    if st.button(t("setup_continue_btn"), type="primary",
                 key="setup_continue_btn"):
        _reset_stages()
        st.session_state.target = target
        st.session_state.task = guess
        st.session_state.features = features
        st.session_state.mode = mode
        st.session_state.model_keys = model_keys
        st.rerun()


# ============================================================ PLAN
def _stage_plan():
    df = st.session_state.df
    st.markdown(t("plan_title"))
    st.caption(t("plan_caption"))
    if "proposed_plan" not in st.session_state:
        with st.spinner(t("plan_spinner")):
            proposed = get_plan(df)
        clean, warnings = validate_plan(proposed, df)
        restricted = restrict_plan(clean,
                                    st.session_state.features or list(df.columns))
        restricted["drop_duplicates"] = clean.get("drop_duplicates", False)
        st.session_state.proposed_plan = proposed
        st.session_state.clean_plan = restricted
        st.session_state.warnings = warnings

    restricted = st.session_state.clean_plan
    warnings = st.session_state.warnings
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(t("plan_will_happen"))
        yes, no = t("plan_yes"), t("plan_no")
        st.write(t("plan_drop_dupes",
                    v=(yes if restricted["drop_duplicates"] else no)))
        imp = restricted.get("impute") or {}
        if imp:
            st.write(t("plan_fill_empty"))
            for col, m in imp.items():
                st.write(t("plan_fill_one", c=col, m=m))
        else:
            st.write(t("plan_fill_none"))
        out = restricted.get("outlier_columns") or []
        st.write(t("plan_trim",
                    cols=(", ".join(out) if out else t("plan_nothing"))))
    with c2:
        st.markdown(t("plan_validator"))
        if warnings:
            for w in warnings:
                st.warning(w)
        else:
            st.success(t("plan_no_warnings"))
    with st.expander(t("plan_show_raw")):
        st.json(st.session_state.proposed_plan)
    c1, c2 = st.columns(2)
    if c1.button(t("plan_approve"), type="primary", use_container_width=True,
                 key="plan_approve_btn"):
        st.session_state.plan_approved = True
        st.rerun()
    if c2.button(t("plan_reject"), use_container_width=True,
                 key="plan_reject_btn"):
        _reset_stages()
        st.rerun()


# ============================================================ TRAIN
def _stage_train():
    df = st.session_state.df
    features = st.session_state.features
    target = st.session_state.target
    task = st.session_state.task
    modality = st.session_state.get("modality", "tabular")
    learning_mode = (st.session_state.get("learning_mode", "ml")
                     if modality == "tabular" else "all")
    nn_cfg = (st.session_state.get("nn_config", DEFAULT_NN_CONFIG)
              if modality == "tabular" else None)
    metric_preference = st.session_state.get("metric_preference", "balanced")
    interaction_specs = st.session_state.get("interaction_specs") or []
    enhancement_specs = st.session_state.get("enhancement_specs") or []
    source_name = st.session_state.get("source_name", "")
    manual_overrides = st.session_state.get("manual_params") or {}

    df_enhanced = _apply_enhancements(df, enhancement_specs)
    extra_cols = [s["name"] for s in enhancement_specs
                  if s["name"] in df_enhanced.columns]
    features_with_enh = features + [c for c in extra_cols if c not in features]
    df_run = df_enhanced[features_with_enh + [target]]

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
            st.error(t("train_stopped", err=str(e)))
            if st.button(t("train_back"), key="train_back_btn"):
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
                    "feature": X.columns,
                    "importance": r.importances_mean,
                }).sort_values("importance", ascending=False)
        except Exception:
            pass
        prog.progress(100, text=t("train_done"))
    st.session_state.results = results
    st.session_state.extras = extras
    st.session_state.notes = notes
    st.session_state.importance = importance
    st.session_state.log.append(f"Training done: best = {extras['best_name']}")
    st.rerun()


# ============================================================ TUNING INLINE
def _render_inline_tuning(extras, results, task, best, test_col, learning_mode):
    st.markdown(t("tune_inline_title"))
    st.caption(t("tune_inline_caption"))

    cat = catalog(task, len(st.session_state.df),
                  nn_config=st.session_state.get("nn_config", DEFAULT_NN_CONFIG),
                  mode=learning_mode)
    name_to_key = {v["name"]: k for k, v in cat.items()}
    model_key = name_to_key.get(best["Model"])
    if not model_key:
        st.info(t("tune_no_params"))
        return

    st.markdown(f"**Tuning: {best['Model']}**")
    st.caption(f"Current test score: **{float(best[test_col]):.3f}**")

    tuning_mode = st.radio(t("tune_approach"),
                            [t("tune_approach_manual"),
                             t("tune_approach_agent")],
                            horizontal=True, key="tune_approach_radio")

    p_defs = hyperparams(model_key, task)
    params_to_apply = {}

    if tuning_mode == t("tune_approach_manual"):
        if not p_defs:
            st.info(t("tune_no_params"))
        for p in p_defs:
            if p["kind"] == "int":
                params_to_apply[p["name"]] = st.slider(
                    p["name"], int(p["min"]), int(p["max"]),
                    int(p["default"]), int(p["step"]),
                    help=p.get("help", ""), key=f"tune_{p['name']}")
            elif p["kind"] == "float":
                params_to_apply[p["name"]] = st.number_input(
                    p["name"], float(p["min"]), float(p["max"]),
                    float(p["default"]), float(p["step"]),
                    help=p.get("help", ""), key=f"tune_{p['name']}")
            elif p["kind"] == "choice":
                params_to_apply[p["name"]] = st.selectbox(
                    p["name"], p["choices"],
                    index=p["choices"].index(p["default"])
                    if p["default"] in p["choices"] else 0,
                    help=p.get("help", ""), key=f"tune_{p['name']}")
    else:
        if st.button(t("tune_agent_ask"), key="tune_agent_ask_btn"):
            with st.spinner(t("tune_agent_thinking")):
                suggested = agent_tune_hyperparams(
                    model_key, task,
                    {p["name"]: p["default"] for p in p_defs})
            st.session_state["_agent_suggested_hp"] = suggested
            st.rerun()

        suggested = st.session_state.get("_agent_suggested_hp")
        if suggested and p_defs:
            valid = {p["name"]: p for p in p_defs}
            cleaned = {}
            for k, v in suggested.items():
                if k not in valid:
                    continue
                p = valid[k]
                try:
                    if p["kind"] == "int":
                        iv = int(v)
                        iv = max(int(p["min"]), min(int(p["max"]), iv))
                        cleaned[k] = iv
                    elif p["kind"] == "float":
                        fv = float(v)
                        fv = max(float(p["min"]), min(float(p["max"]), fv))
                        cleaned[k] = fv
                    elif p["kind"] == "choice":
                        if v in p["choices"]:
                            cleaned[k] = v
                except (ValueError, TypeError):
                    continue

            if cleaned and cleaned != suggested:
                st.caption(t("tune_agent_adjusted"))
            if cleaned:
                st.success(t("tune_agent_suggests", params=str(cleaned)))
                params_to_apply = cleaned
            else:
                st.warning(t("tune_no_params"))

    if st.button(t("tune_retrain_btn"), type="primary",
                 key="retrain_tuned_btn"):
        if not params_to_apply:
            st.warning("No parameters to apply.")
        else:
            st.session_state.pop("_tune_result", None)
            with st.spinner(t("tune_retraining")):
                df_run_tune = st.session_state.df[
                    st.session_state.features + [st.session_state.target]]
                try:
                    r2, n2, e2 = train_run(
                        df_run_tune, st.session_state.target,
                        st.session_state.clean_plan, task, [model_key],
                        self_check=False,
                        nn_config=st.session_state.get("nn_config", DEFAULT_NN_CONFIG),
                        mode=learning_mode,
                        metric_preference=extras.get("metric_preference", "balanced"),
                        source_name=st.session_state.get("source_name", ""),
                        hyperparams_overrides={model_key: params_to_apply})
                    old_score = float(best[test_col])
                    new_row = r2.iloc[0]
                    new_score = float(new_row[test_col])
                    delta = new_score - old_score

                    explanation_prompt = (
                        f"You are an ML engineer explaining a hyperparameter "
                        f"change to a colleague. Old score: {old_score:.3f}, "
                        f"new score: {new_score:.3f}, delta: {delta:+.3f}. "
                        f"Old params: {params_to_apply}. "
                        f"Model: {best['Model']}. Dataset rows: "
                        f"{extras.get('n_train', '?')}. "
                        "In 2-3 sentences, explain WHY the score changed the "
                        "way it did. Be specific about which parameter likely "
                        "caused what. Keep it plain and non-technical.")
                    try:
                        llm_explanation = (ask(explanation_prompt,
                                                temperature=0.3) or "").strip()
                    except Exception:
                        llm_explanation = ""

                    icon = "✅" if delta > 0.001 else "❌" if delta < -0.001 else "➖"
                    if llm_explanation:
                        st.session_state["_tune_result"] = (
                            f"**{icon} Score change: {delta:+.3f}** "
                            f"({old_score:.3f} → {new_score:.3f})\n\n"
                            f"{llm_explanation}")
                    elif delta > 0.001:
                        st.session_state["_tune_result"] = t(
                            "tune_improved", delta=delta,
                            old=old_score, new=new_score)
                    elif delta < -0.001:
                        st.session_state["_tune_result"] = t(
                            "tune_dropped", delta=delta,
                            old=old_score, new=new_score)
                    else:
                        st.session_state["_tune_result"] = t(
                            "tune_same", delta=delta)
                    st.rerun()
                except Exception as e:
                    st.error(t("tune_retrain_fail",
                                err=f"{type(e).__name__}: {e}"))

    if st.session_state.get("_tune_result"):
        st.info(st.session_state["_tune_result"])


# ============================================================ PREDICT
def _predict_manual(pipeline, df_train, features, target, interactions=None):
    interactions = interactions or []
    ix_names = {s["name"] for s in interactions}
    base_features = [f for f in features if f not in ix_names]
    values = {}
    cols = st.columns(2)
    for i, f in enumerate(base_features):
        series = df_train[f] if f in df_train.columns else pd.Series([0.0])
        with cols[i % 2]:
            kind = column_kind(series, f) if len(series) else "num"
            if kind in ("num", "numunit"):
                v = pd.to_numeric(series, errors="coerce").dropna()
                med = float(v.median()) if len(v) else 0.0
                values[f] = st.number_input(f, value=med, key=f"pred_in_{f}")
            elif kind in ("cat", "topcat", "bool"):
                uniq = series.dropna().astype(str).unique().tolist()[:200] or [""]
                values[f] = st.selectbox(f, uniq, key=f"pred_in_{f}")
            elif kind == "date":
                values[f] = st.date_input(f, key=f"pred_in_{f}")
            else:
                values[f] = st.text_area(f, "", height=80, key=f"pred_in_{f}")

    c1, c2 = st.columns(2)
    with c1:
        predict_clicked = st.button(t("predict_btn"), type="primary",
                                     key="predict_manual_btn",
                                     use_container_width=True)
    with c2:
        explain_clicked = st.button(t("prof_model_explain_btn"), type="primary",
                                     key="predict_manual_explain_btn",
                                     use_container_width=True)

    if predict_clicked or explain_clicked:
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
            st.error(t("predict_missing_feature", e=e))
            return
        try:
            pred = pipeline.predict(row)[0]
        except Exception as e:
            st.error(t("predict_failed", err=f"{type(e).__name__}: {e}"))
            return
        c = _theme_colors()
        st.markdown(
            f'<div class="card" style="border-left:3px solid {c["accent"]};">'
            f'<h4>{t("predict_result")}</h4>'
            f'<p style="font-size:1.6rem;font-weight:800;color:{c["accent"]};">'
            f'{target} = {pred}</p></div>', unsafe_allow_html=True)
        if explain_clicked:
            _render_row_xai(pipeline, row, features)


def _predict_batch(pipeline, features, target, interactions=None):
    uploaded = st.file_uploader(t("predict_upload_hint"),
                                 type=["csv", "xlsx", "xls"],
                                 key="predict_upload")
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
        st.error(f"Could not read: {type(e).__name__}: {e}")
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
        st.error(t("predict_missing", cols=", ".join(missing)))
        return
    try:
        preds = pipeline.predict(df_new[features])
    except Exception as e:
        st.error(t("predict_failed", err=f"{type(e).__name__}: {e}"))
        return
    out = df_new.copy()
    out[f"predicted_{target}"] = preds
    st.dataframe(out.head(20), use_container_width=True)
    st.download_button(t("predict_download"),
                       data=out.to_csv(index=False).encode("utf-8"),
                       file_name=f"predictions_{os.path.basename(tmp)}.csv",
                       mime="text/csv")


def _stage_predict():
    extras = st.session_state.extras
    df_train = st.session_state.df
    features = extras["features"]
    target = extras["target"]
    pipeline = extras["best_pipeline"]
    interactions = extras.get("interactions") or []
    st.divider()
    st.markdown(t("results_pred_title"))
    st.caption(t("results_pred_caption",
                  m=extras["best_name"], target=target))
    mode = st.radio(t("results_pred_method"),
                    [t("results_pred_manual"), t("results_pred_upload")],
                    horizontal=True, key="predict_mode")
    if mode == t("results_pred_manual"):
        _predict_manual(pipeline, df_train, features, target, interactions)
    else:
        _predict_batch(pipeline, features, target, interactions)


# ============================================================ RESULTS HELPERS
def _metric_explanation(task, best, base):
    if task == "regression":
        r2 = float(best["Test R2"])
        return t("results_metric_help", r2=r2, pct=max(r2, 0) * 100,
                 base=float(base["Test R2"]))
    f1 = float(best["F1"])
    return t("results_metric_help_clf", f1=f1, base=float(base["F1"]))


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
        reason = specific.get(name) or REASON.get(key, "chosen by the agent.")
        st.markdown(
            f'<div class="card"><h4>{name} '
            f'<span class="pill {pill_class}">{tag}</span></h4>'
            f'<p>{reason}</p></div>', unsafe_allow_html=True)


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
            fig = px.line(x=fpr, y=tpr)
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dash", color="#94a3b8"))
            fig.update_traces(line=dict(color="#22d3ee", width=3))
            fig.update_layout(height=340, title=f"ROC · AUC = {roc_auc:.3f}",
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#A1A1AA"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.line(x=rec, y=prec)
            fig.update_traces(line=dict(color="#e879f9", width=3))
            fig.update_layout(height=340, title=f"PR · AUC = {pr_auc:.3f}",
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#A1A1AA"))
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
        fig = px.scatter(x=y_pred, y=y_true)
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                      line=dict(dash="dash", color="#94a3b8"))
        fig.update_traces(marker=dict(color="#22d3ee", size=6, opacity=0.6))
        fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#A1A1AA"))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(x=y_pred, y=residuals)
        fig.add_shape(type="line", x0=lo, y0=0, x1=hi, y1=0,
                      line=dict(dash="dash", color="#94a3b8"))
        fig.update_traces(marker=dict(color="#e879f9", size=6, opacity=0.6))
        fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#A1A1AA"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f'<div class="card">'
        f'<p><b>{t("results_resid_left_title")}</b> '
        f'{t("results_resid_left_body")}</p>'
        f'<p style="margin-top:0.6rem;">'
        f'<b>{t("results_resid_right_title")}</b> '
        f'{t("results_resid_right_body")}</p>'
        f'</div>', unsafe_allow_html=True)


# ============================================================ RESULTS
def _stage_results():
    results = st.session_state.results
    extras = st.session_state.extras
    notes = st.session_state.notes
    task = st.session_state.task
    target = st.session_state.target
    mode = extras.get("mode", "tabular")
    learning_mode = extras.get("learning_mode", "ml")
    c = _theme_colors()

    best = results.iloc[0]
    base = results.iloc[-1]
    cv_col = "CV R2" if task == "regression" else "CV F1"
    test_col = "Test R2" if task == "regression" else "F1"

    st.markdown(t("results_title"))

    c1, c2, c3 = st.columns(3)
    with c1: ui.stat(best["Model"], t("results_best_model"))
    with c2: ui.stat(f"{float(best[cv_col]):.3f}", t("results_cv"))
    with c3: ui.stat(f"{float(best[test_col]):.3f}", t("results_test"))

    st.markdown(
        f'<div class="card"><h4>{t("results_what_numbers_title")}</h4>'
        f'<p>{_metric_explanation(task, best, base)}</p></div>',
        unsafe_allow_html=True)

    if mode != "nlp":
        _render_global_xai(extras, extras.get("X_test"))

    st.divider()
    st.markdown(t("results_leaderboard"))
    st.dataframe(results, use_container_width=True)

    with st.expander(t("results_feat_eng")):
        if mode == "nlp":
            st.markdown("**Text** → TF-IDF. **Numeric** → impute + scale.")
        else:
            plan = engineering_summary(st.session_state.df[
                st.session_state.features + [st.session_state.target]])
            for line in plan:
                st.markdown(f"• {line}")

    fig = px.bar(results, x="Model", y=cv_col, color=cv_col,
                 color_continuous_scale=[[0, c["border"]], [1, c["accent"]]],
                 text=cv_col)
    fig.update_layout(height=340, showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#A1A1AA"))
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
                                x=[cc.replace("Predicted ", "") for cc in cm.columns],
                                y=[rr.replace("Actual ", "") for rr in cm.index],
                                color_continuous_scale=[[0, c["border"]], [1, c["accent"]]],
                                aspect="auto", text_auto=True)
                fig.update_layout(height=420,
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="#A1A1AA"))
                st.plotly_chart(fig, use_container_width=True)
        _render_roc_pr(task, extras)

    if task == "regression":
        with st.expander(t("results_adv_diag"), expanded=False):
            _render_residual_plot(extras)

    st.divider()
    _render_inline_tuning(extras, results, task, best, test_col, learning_mode)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(t("results_self_check"))
        for line in extras.get("selfcheck", []):
            st.info(line)
    with c2:
        st.markdown(t("results_plain_words"))
        for line in extras.get("explanations", []):
            st.markdown(f"• {line}")

    _stage_predict()

    with st.expander(t("results_pipeline_notes")):
        for n in notes:
            st.write("• " + n)

    st.divider()
    st.markdown(t("results_save_export"))
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button(t("results_save_btn"), type="primary",
                     use_container_width=True, key="save_export_btn"):
            try:
                folder = export_all(
                    st.session_state.source_path, target, task,
                    st.session_state.clean_plan, st.session_state.warnings,
                    st.session_state.load_notes, results, notes, extras)
                st.session_state.output_folder = folder
                st.success(t("results_saved_to", folder=folder))
                st.rerun()
            except Exception as e:
                st.error(f"Save failed: {type(e).__name__}: {e}")
    with c2:
        st.download_button(t("results_csv_btn"),
                            data=results.to_csv(index=False),
                            file_name=f"leaderboard_{target}.csv",
                            mime="text/csv", use_container_width=True,
                            key="dl_leaderboard_btn")
    with c3:
        try:
            zip_bytes = _build_model_zip(extras, target, task, results,
                                          st.session_state.clean_plan)
            st.download_button(t("results_model_btn"), data=zip_bytes,
                                file_name=f"model_{target}.zip",
                                mime="application/zip",
                                use_container_width=True, key="dl_model_zip_btn")
        except Exception as e:
            st.button(t("results_model_btn"), disabled=True,
                      use_container_width=True,
                      help=t("results_model_fail", err=str(e)))

    if st.button(t("results_try_another"), use_container_width=True,
                 key="try_another_btn"):
        _reset_stages()
        st.rerun()


# ============================================================ ROUTER
def render_pipeline():
    if not auth.is_logged_in():
        ui.hero("Sign in required", "Needs an authenticated session.", pills=[])
        return

    if st.session_state.get("df") is not None:
        detected_modality = auto_detect_modality(st.session_state.df)
        if st.session_state.get("modality") != detected_modality:
            st.session_state.modality = detected_modality
            _reset_modality()
            st.rerun()
    else:
        ui.hero("Run Pipeline", "Load a dataset first.", pills=[])
        if st.button("🔎  Go to Data Profiler", type="primary", key="goto_prof"):
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
        if st.button("← Change target or columns", key="change_target_btn"):
            _reset_stages()
            for k in ("target", "task", "features", "mode", "model_keys",
                      "metric_preference", "interaction_proposals",
                      "interaction_specs", "manual_params",
                      "enhancement_specs", "_show_enhance_ui"):
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
        return
    history = load_history()
    runs = [h for h in history if h.get("source") == source]
    if not runs:
        st.info(t("tune_no_runs", source=source))
        return
    st.markdown(t("tune_history_title"))
    hist_df = pd.DataFrame([{
        "when": r["timestamp"], "target": r["target"], "task": r["task"],
        "features": r["n_features"], "best model": r["best_model"],
        "score": round(r["best_score"], 3),
    } for r in runs])
    st.dataframe(hist_df, use_container_width=True)