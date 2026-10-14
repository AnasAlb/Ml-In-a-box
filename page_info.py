"""
Info pages: Tutorial, About, Glossary, Ask (chat).
"""
import html
import os
import uuid

import pandas as pd
import plotly.express as px
import streamlit as st

import ui
from i18n import t, get_glossary
from ml import load_table_with_notes, profile_df
from llm import ask


# ============================================================ TUTORIAL
def _tutorial_card_css():
    st.markdown(
        """
        <style>
            .card.tut-static:hover {
                transform: none !important;
                border-color: #2a3350 !important;
                box-shadow: none !important;
            }
            .card.tut-static, .card.tut-static * {
                user-select: text !important;
                -webkit-user-select: text !important;
                cursor: text;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_tutorial():
    ui.hero(t("tut_hero_title"), t("tut_sub"),
            pills=[t("tut_pill_1"), t("tut_pill_2"), t("tut_pill_3")])

    _tutorial_card_css()

    st.markdown(
        f'<div class="card" style="border-left:3px solid #e879f9;padding:2rem;'
        f'text-align:center;">'
        f'<h4 style="margin:0 0 0.6rem 0;">{t("tut_video_title")}</h4>'
        f'<p style="color:#94a3b8;margin:0;">{t("tut_video_body")}</p>'
        f'</div>', unsafe_allow_html=True)

    steps = t("tutorial_steps")
    if not isinstance(steps, list):
        steps = []
    step_label = t("tut_step_label")
    for num, title, body in steps:
        st.markdown(
            f'<div class="card tut-static">'
            f'<h4><span class="pill">{step_label} {num}</span> {title}</h4>'
            f'<p>{body}</p></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button(t("tut_go_profile"), type="primary",
                     use_container_width=True):
            st.session_state.page = "profile"
            st.rerun()
    with c2:
        if st.button(t("tut_go_glossary"), use_container_width=True):
            st.session_state.page = "glossary"
            st.rerun()


# ============================================================ ABOUT
def render_about():
    ui.hero(
        t("about_hero_title"),
        t("about_hero_sub"),
        pills=[t("about_pill_1"), t("about_pill_2"), t("about_pill_3")],
    )

    ui.prose(t("about_prose"))

    st.divider()
    st.markdown(f"### {t('about_us_title')}")
    st.caption(t("about_us_caption"))
    st.markdown(
        f'<div class="card" style="border-left:3px solid #e879f9;padding:2rem;'
        f'text-align:center;">'
        f'<h4 style="margin:0 0 0.6rem 0;">{t("about_video_title")}</h4>'
        f'<p style="color:#94a3b8;margin:0;">{t("about_video_body")}</p>'
        f'</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown(f"### {t('about_team_title')}")
    c1, c2 = st.columns(2)

    def _team_card(col, name, role, phone, photo_candidates, fallback="👤"):
        with col:
            with st.container(border=True):
                photo = next((p for p in photo_candidates if os.path.isfile(p)), None)
                img_col, _ = st.columns([1, 1])
                with img_col:
                    if photo:
                        st.image(photo, width=220)
                    else:
                        st.markdown(
                            f'<div style="font-size:3rem;text-align:center;'
                            f'padding:1rem 0;">{fallback}</div>',
                            unsafe_allow_html=True)
                st.markdown(
                    f'<h4 style="margin-top:0.6rem;text-align:center;">{name}</h4>'
                    f'<p style="color:#b6c2d4;text-align:center;">{role}</p>'
                    f'<p style="color:#94a3b8;font-size:0.9rem;'
                    f'text-align:center;margin-top:0.6rem;">'
                    f'📱 <b>{phone}</b> <span style="opacity:0.65;">(WhatsApp)</span>'
                    f'</p>',
                    unsafe_allow_html=True)

    _team_card(
        c1,
        "Engr. Anas Yousef Albakoush",
        t("about_role_anas"),
        "+218 92-4834666",
        ("anas.png", "assets/anas.png", "images/anas.png", "photos/anas.png"),
    )
    _team_card(
        c2,
        "Engr. Muad Salah Himeedah",
        t("about_role_muad"),
        "+218 94-6312894",
        ("muad.png", "assets/muad.png", "images/muad.png", "photos/muad.png"),
    )

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button(t("about_btn_tutorial"), use_container_width=True):
            st.session_state.page = "tutorial"
            st.rerun()
    with c2:
        if st.button(t("about_btn_glossary"), use_container_width=True):
            st.session_state.page = "glossary"
            st.rerun()
    with c3:
        if st.button(t("about_btn_start"), type="primary",
                     use_container_width=True):
            st.session_state.page = "profile"
            st.rerun()

    ui.sisyphus()


# ============================================================ GLOSSARY
_GLOSS_PALETTE = ["#22d3ee", "#e879f9", "#fbbf24", "#4ade80", "#f43f5e",
                  "#67e8f9", "#f0a6fc", "#fcd34d", "#a78bfa", "#5fb9a0"]


def _find_term(label, glossary):
    for cat, entries in glossary.items():
        for tup in entries:
            if tup[0] == label:
                return cat, tup[0]
    return None, None


def _find_category(label, glossary):
    if label in glossary:
        return label
    return None


def _read_treemap_click(event):
    if event is None:
        return None
    try:
        sel = event.get("selection") if hasattr(event, "get") else getattr(event, "selection", None)
        if not sel:
            return None
        pts = sel.get("points") if hasattr(sel, "get") else getattr(sel, "points", None)
        if not pts:
            return None
        first = pts[0]
        label = first.get("label") if isinstance(first, dict) else getattr(first, "label", None)
        return label or None
    except Exception:
        return None


def render_glossary():
    glossary = get_glossary()

    ui.hero(
        t("gloss_hero_title"),
        t("gloss_hero_sub"),
        pills=[
            t("gloss_pill_categories").format(n=len(glossary)),
            t("gloss_pill_click"),
            t("gloss_pill_plain"),
        ],
    )

    rows = []
    for cat, entries in glossary.items():
        for term, *_ in entries:
            rows.append({"category": cat, "term": term, "size": 1})
    df = pd.DataFrame(rows)

    st.markdown(f"### {t('gloss_explore')}")
    fig = px.treemap(
        df, path=["category", "term"], values="size",
        color="category", color_discrete_sequence=_GLOSS_PALETTE,
    )
    fig.update_traces(
        textinfo="label",
        marker=dict(
            line=dict(color="#0a0d1a", width=2),
            pad=dict(t=22, l=3, r=3, b=3),
        ),
        hovertemplate="<b>%{label}</b><extra></extra>",
    )
    fig.update_layout(
        height=520,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8eef7", size=14),
    )

    event = None
    try:
        event = st.plotly_chart(
            fig, use_container_width=True,
            on_select="rerun",
            key="gloss_treemap",
        )
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)
        st.caption(t("gloss_old_streamlit"))
        event = None

    clicked = _read_treemap_click(event)
    if clicked:
        cat_from_term, term = _find_term(clicked, glossary)
        target_cat = target_term = None
        if term:
            target_cat, target_term = cat_from_term, term
        else:
            cat_only = _find_category(clicked, glossary)
            if cat_only:
                target_cat = cat_only
                target_term = glossary[cat_only][0][0]

        if target_cat:
            current_cat = st.session_state.get("gloss_cat_widget")
            current_term = st.session_state.get("gloss_term_widget")
            if target_cat != current_cat or target_term != current_term:
                st.session_state["gloss_cat_widget"] = target_cat
                st.session_state["gloss_term_widget"] = target_term
                st.rerun()

    st.caption(t("gloss_tip"))

    st.divider()
    st.markdown(f"### {t('gloss_full_explanation')}")

    categories = list(glossary.keys())
    if "gloss_cat_widget" not in st.session_state or \
            st.session_state["gloss_cat_widget"] not in categories:
        st.session_state["gloss_cat_widget"] = categories[0]

    def _on_cat_change():
        new_cat = st.session_state["gloss_cat_widget"]
        st.session_state["gloss_term_widget"] = glossary[new_cat][0][0]

    c1, c2 = st.columns(2)
    with c1:
        cat = st.selectbox(t("gloss_category"), categories,
                           key="gloss_cat_widget",
                           on_change=_on_cat_change)
    entries = glossary[cat]
    terms = [tup[0] for tup in entries]
    if "gloss_term_widget" not in st.session_state or \
            st.session_state["gloss_term_widget"] not in terms:
        st.session_state["gloss_term_widget"] = terms[0]
    with c2:
        term = st.selectbox(t("gloss_term"), terms, key="gloss_term_widget")

    lookup = {tup[0]: (tup[1], tup[2], tup[3]) for tup in entries}
    short, long_, how = lookup[term]

    st.markdown(
        f'<div class="card" style="border-left:3px solid #22d3ee;">'
        f'<h4>{term}</h4>'
        f'<p style="color:#67e8f9;font-weight:500;">{short}</p>'
        f'<p style="margin-top:0.6rem;">{long_}</p>'
        f'<p style="color:#94a3b8;font-size:0.85rem;margin-top:0.7rem;">'
        f'<b>{t("gloss_how_to_read")}</b> {how}</p>'
        f'<p style="color:#5c6a85;font-size:0.78rem;margin-top:0.7rem;">'
        f'📖 {t("gloss_further_reading")} <em>{t("gloss_wip")}</em></p>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ============================================================ CHAT
def _build_context():
    parts = []
    if st.session_state.get("source_name"):
        df = st.session_state.df
        parts.append(f"Dataset: {st.session_state.source_name}, {len(df)} rows, "
                     f"{len(df.columns)} columns.")
        parts.append(f"Columns: {list(df.columns)}")
    if st.session_state.get("target"):
        parts.append(f"Target: {st.session_state.target} ({st.session_state.task}).")
    if st.session_state.get("features"):
        parts.append(f"Input columns used: {st.session_state.features}")
    if st.session_state.get("results") is not None:
        parts.append("Leaderboard:\n" + st.session_state.results.to_string(index=False))
    if st.session_state.get("extras"):
        extras = st.session_state.extras
        parts.append("Self-check: " + " ".join(extras.get("selfcheck", [])))
        parts.append("Explanations: " + " ".join(extras.get("explanations", [])))
    return "\n".join(parts) if parts else "No dataset loaded yet."


def _answer(question):
    context = _build_context()
    prompt = (
        "You are the assistant inside an ML demo. Answer using ONLY the facts below. "
        "If not covered, say so plainly. 2-4 short sentences. Plain language.\n\n"
        f"--- session facts ---\n{context}\n--- end ---\n\nQuestion: {question}"
    )
    try:
        return (ask(prompt, temperature=0.2) or "").strip()
    except Exception as e:
        return f"(Assistant unavailable: {type(e).__name__})"


def _render_chat_upload():
    st.markdown(
        f'<div class="card" style="border-left:3px solid #fbbf24;">'
        f'<h4>{t("chat_no_data_title")}</h4>'
        f'<p>{t("chat_no_data_body")}</p>'
        f'</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(t("chat_upload_label"),
                                 type=["csv", "xlsx", "xls"], key="chat_upload",
                                 label_visibility="collapsed")
    if uploaded is None:
        return
    os.makedirs("outputs", exist_ok=True)
    suffix = os.path.splitext(uploaded.name)[1]
    tmp = os.path.join("outputs", f"_chat_{uuid.uuid4().hex[:6]}{suffix}")
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


def render_chat():
    ui.hero(
        t("chat_hero_title"),
        t("chat_hero_sub"),
        pills=[t("chat_pill_1"), t("chat_pill_2"), t("chat_pill_3")],
    )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.session_state.get("source_name"):
        ctx = (f"📄 **{st.session_state.source_name}** · "
               + t("chat_ctx_rows").format(
                    r=len(st.session_state.df),
                    c=len(st.session_state.df.columns)))
        if st.session_state.get("target"):
            ctx += " · " + t("chat_ctx_target").format(t=st.session_state.target)
        if st.session_state.get("results") is not None:
            ctx += " · " + t("chat_ctx_results")
        st.success(ctx)
    else:
        _render_chat_upload()
        st.divider()

    st.markdown(t("chat_try_one"))
    suggestions = t("chat_suggested")
    if not isinstance(suggestions, list) or not suggestions:
        suggestions = []
    cols = st.columns(4)
    for i, q in enumerate(suggestions):
        with cols[i % 4]:
            if st.button(q, key=f"chip_{i}", use_container_width=True):
                st.session_state.chat_history.append(("user", q))
                with st.spinner(t("chat_thinking")):
                    a = _answer(q)
                st.session_state.chat_history.append(("assistant", a))
                st.rerun()

    st.divider()
    for role, text in st.session_state.chat_history:
        safe = html.escape(text).replace("\n", "<br>")
        if role == "user":
            st.markdown(
                f'<div class="card" style="border-left:3px solid #22d3ee;">'
                f'<p style="color:#67e8f9;font-weight:600;margin-bottom:0.3rem;">'
                f'{t("chat_you")}</p>'
                f'<p style="color:#e8eef7;">{safe}</p></div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="card" style="border-left:3px solid #e879f9;">'
                f'<p style="color:#f0a6fc;font-weight:600;margin-bottom:0.3rem;">'
                f'{t("chat_assistant")}</p>'
                f'<p style="color:#e8eef7;">{safe}</p></div>',
                unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            q = st.text_input(t("chat_input_label"),
                              placeholder=t("chat_input_placeholder"),
                              label_visibility="collapsed")
        with c2:
            submitted = st.form_submit_button(t("chat_send"),
                                               use_container_width=True)
    if submitted and q.strip():
        st.session_state.chat_history.append(("user", q.strip()))
        with st.spinner(t("chat_thinking")):
            a = _answer(q.strip())
        st.session_state.chat_history.append(("assistant", a))
        st.rerun()
    if st.session_state.chat_history:
        if st.button(t("chat_clear_history")):
            st.session_state.chat_history = []
            st.rerun()