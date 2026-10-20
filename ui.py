"""
Shared UI helpers: theme, cards, hero, stat boxes, 3D robot, Byte (Wall-E
modal assistant), watermark, Sisyphus, glossary.
"""
import os
import textwrap
import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# THEME PALETTES
# ============================================================
def _palette(theme: str) -> dict:
    if theme == "light":
        return {
            "bg": "#F9FAFB", "bg_soft": "#FFFFFF", "card_bg": "#FFFFFF",
            "border": "#E5E7EB", "border_accent": "#D4AF37",
            "accent": "#34D399", "accent_soft": "rgba(52, 211, 153, 0.1)",
            "accent_text": "#059669", "accent_2": "#D4AF37",
            "text": "#111827", "text_muted": "#6B7280", "text_dim": "#9CA3AF",
            "shadow_sm": "0 4px 15px rgba(212, 175, 55, 0.08)",
            "shadow_md": "0 10px 30px -10px rgba(212, 175, 55, 0.2)",
            "glow": "rgba(52, 211, 153, 0.25)",
            "glow_2": "rgba(212, 175, 55, 0.25)",
            "button_primary_bg": "#D4AF37", "button_primary_text": "#FFFFFF",
            "button_primary_hover": "#B8860B",
            "button_secondary_bg": "#FFFFFF", "button_secondary_text": "#4B5563",
            "button_secondary_border": "#E5E7EB",
            "btn_secondary_hover_bg": "#ECFDF5",
            "btn_secondary_hover_text": "#059669",
            "btn_secondary_hover_border": "#34D399",
            "pill_bg": "rgba(52, 211, 153, 0.1)", "pill_text": "#059669",
            "pill_border": "rgba(52, 211, 153, 0.35)",
            "pill_purple_bg": "rgba(212, 175, 55, 0.1)",
            "pill_purple_text": "#B8860B",
            "pill_purple_border": "rgba(212, 175, 55, 0.35)",
            "pill_amber_bg": "rgba(251, 191, 36, 0.15)",
            "pill_amber_text": "#D97706",
            "pill_amber_border": "rgba(251, 191, 36, 0.4)",
            "pill_red_bg": "rgba(239, 68, 68, 0.1)", "pill_red_text": "#DC2626",
            "pill_red_border": "rgba(239, 68, 68, 0.3)",
            "input_bg": "#FFFFFF", "input_text": "#111827",
            "input_border": "#E5E7EB", "input_placeholder": "#9CA3AF",
            "robot_bg": "#FFFFFF", "robot_border": "#D4AF37",
            "robot_eye": "#34D399", "robot_eye_glow": "rgba(52, 211, 153, 0.85)",
            "robot_body": "#F3F4F6",
            "progress_grad": "linear-gradient(90deg, #34D399, #D4AF37)",
        }
    return {
        "bg": "#050505", "bg_soft": "#0A0A0A", "card_bg": "#111111",
        "border": "#27272A", "border_accent": "#00F0FF",
        "accent": "#00F0FF", "accent_soft": "rgba(0, 240, 255, 0.1)",
        "accent_text": "#00F0FF", "accent_2": "#A855F7",
        "text": "#FFFFFF", "text_muted": "#A1A1AA", "text_dim": "#52525B",
        "shadow_sm": "0 4px 20px rgba(0, 0, 0, 0.5)",
        "shadow_md": "0 10px 30px -10px rgba(0, 240, 255, 0.3)",
        "glow": "rgba(0, 240, 255, 0.3)",
        "glow_2": "rgba(168, 85, 247, 0.3)",
        "button_primary_bg": "#00F0FF", "button_primary_text": "#050505",
        "button_primary_hover": "#00D0E0",
        "button_secondary_bg": "#18181B", "button_secondary_text": "#FFFFFF",
        "button_secondary_border": "#27272A",
        "btn_secondary_hover_bg": "rgba(0, 240, 255, 0.05)",
        "btn_secondary_hover_text": "#00F0FF",
        "btn_secondary_hover_border": "#00F0FF",
        "pill_bg": "rgba(0, 240, 255, 0.1)", "pill_text": "#00F0FF",
        "pill_border": "rgba(0, 240, 255, 0.3)",
        "pill_purple_bg": "rgba(168, 85, 247, 0.1)",
        "pill_purple_text": "#C084FC",
        "pill_purple_border": "rgba(168, 85, 247, 0.3)",
        "pill_amber_bg": "rgba(251, 191, 36, 0.1)",
        "pill_amber_text": "#FCD34D",
        "pill_amber_border": "rgba(251, 191, 36, 0.3)",
        "pill_red_bg": "rgba(244, 63, 94, 0.1)", "pill_red_text": "#FB7185",
        "pill_red_border": "rgba(244, 63, 94, 0.3)",
        "input_bg": "#0F0F0F", "input_text": "#FFFFFF",
        "input_border": "#27272A", "input_placeholder": "#52525B",
        "robot_bg": "#18181B", "robot_border": "#00F0FF",
        "robot_eye": "#00F0FF", "robot_eye_glow": "rgba(0, 240, 255, 0.9)",
        "robot_body": "#09090B",
        "progress_grad": "linear-gradient(90deg, #00F0FF, #A855F7)",
    }


# ============================================================
# THEME BUILDER
# ============================================================
def _build_theme(p: dict) -> str:
    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    header[data-testid="stHeader"] {{ display: none !important; }}
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}

    * {{ transition: background-color 0.5s ease, color 0.5s ease,
                       border-color 0.5s ease, box-shadow 0.5s ease; }}

    .stApp {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: {p['bg']};
        color: {p['text']};
    }}

    .block-container {{
        padding-top: 7rem !important;
        padding-bottom: 5rem;
        max-width: 1400px;
    }}

    h1, h2, h3, h4 {{
        letter-spacing: -0.03em;
        color: {p['text']};
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800;
    }}

    /* ---------- PRIMARY TEXT CONTRAST FIX ---------- */
    /* Ensure native markdown, metrics, dataframes, and expanders use the high-contrast text color */
    .stMarkdown p, .stMarkdown li, .stMarkdown span, .stMarkdown div,
    .stText, .stDataFrame td, .stDataFrame th, .stDataFrame div,
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
        color: {p['text']} !important;
    }}
    
    /* Keep captions and form labels muted for hierarchy */
    .stCaption, .stCaption p, label, .st-emotion-cache-16idsys p {{
        color: {p['text_muted']} !important;
    }}

    /* Style Streamlit Alerts (info, warning, error, success) */
    div[data-testid="stNotification"] {{
        background-color: {p['bg_soft']} !important;
        border: 1px solid {p['border']} !important;
        border-radius: 12px;
    }}
    div[data-testid="stNotification"] p, div[data-testid="stNotification"] span {{
        color: {p['text']} !important;
    }}

    /* Dropdown menus text */
    div[data-baseweb="menu"] {{
        background-color: {p['card_bg']} !important;
    }}
    div[data-baseweb="menu"] div {{
        color: {p['text']} !important;
    }}
    div[data-baseweb="menu"] li:hover {{
        background-color: {p['accent_soft']} !important;
    }}

    /* ---------- INPUT STYLING ---------- */
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {{
        background-color: {p['input_bg']} !important;
        color: {p['text']} !important;
        border: 1px solid {p['input_border']} !important;
        border-radius: 10px !important;
    }}
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {{
        color: {p['text_dim']} !important;
        opacity: 1 !important;
    }}
    .stTextInput input:focus,
    .stTextArea textarea:focus,
    .stNumberInput input:focus {{
        border-color: {p['accent']} !important;
        box-shadow: 0 0 0 3px {p['accent_soft']} !important;
    }}

    /* Selectbox */
    .stSelectbox div[data-baseweb="select"] > div {{
        background-color: {p['input_bg']} !important;
        color: {p['text']} !important;
        border-color: {p['input_border']} !important;
    }}
    .stSelectbox div[data-baseweb="select"] svg {{
        fill: {p['text_muted']} !important;
    }}

    /* Multiselect */
    .stMultiSelect div[data-baseweb="select"] > div {{
        background-color: {p['input_bg']} !important;
        border-color: {p['input_border']} !important;
    }}
    .stMultiSelect span[data-baseweb="tag"] {{
        background-color: {p['accent_soft']} !important;
        color: {p['accent_text']} !important;
    }}

    /* Number input +/- buttons */
    .stNumberInput button {{
        background-color: {p['card_bg']} !important;
        color: {p['text_muted']} !important;
        border-color: {p['input_border']} !important;
    }}

    /* Expanders */
    details summary {{ color: {p['text']} !important; }}

    /* ---------- HERO ---------- */
    .hero {{
        padding: 4rem 3rem;
        border-radius: 32px;
        background: {p['bg_soft']};
        border: 1px solid {p['border_accent']};
        box-shadow: {p['shadow_md']};
        margin-bottom: 3rem;
        position: relative;
        overflow: hidden;
    }}
    .hero::before {{
        content: ""; position: absolute; top: -50%; right: -10%;
        width: 600px; height: 600px;
        background: radial-gradient(circle, {p['glow']}, transparent 70%);
        border-radius: 50%; pointer-events: none;
    }}
    .hero::after {{
        content: ""; position: absolute; bottom: -40%; left: -10%;
        width: 500px; height: 500px;
        background: radial-gradient(circle, {p['glow_2']}, transparent 70%);
        border-radius: 50%; pointer-events: none;
    }}
    .hero h1 {{
        font-size: 4rem; margin: 0; padding-bottom: 8px;
        background: linear-gradient(135deg, {p['text']} 0%, {p['accent']} 100%);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1; position: relative; z-index: 1;
    }}
    .hero p {{
        font-size: 1.2rem; color: {p['text_muted']};
        margin: 1rem 0 1.5rem 0; max-width: 700px;
        position: relative; z-index: 1; line-height: 1.6;
    }}

    /* ---------- CARDS ---------- */
    .card {{
        background: {p['card_bg']};
        border: 1px solid {p['border']};
        border-radius: 20px;
        padding: 1.8rem; margin-bottom: 1.5rem;
        box-shadow: {p['shadow_sm']};
        position: relative;
        overflow: hidden;
    }}
    .card:hover {{
        border-color: {p['accent']};
        box-shadow: {p['shadow_md']};
    }}
    .card h4 {{ margin: 0 0 0.6rem 0; color: {p['text']};
                font-size: 1.05rem; font-weight: 700; }}
    .card p {{ margin: 0; color: {p['text_muted']};
               font-size: 0.95rem; line-height: 1.6; }}

    /* ---------- PILLS ---------- */
    .pill {{
        display: inline-block; padding: 6px 16px; border-radius: 30px;
        background: {p['pill_bg']}; color: {p['pill_text']};
        font-size: 0.75rem; font-weight: 700;
        margin-right: 8px; margin-bottom: 8px;
        border: 1px solid {p['pill_border']};
    }}
    .pill.purple {{ background: {p['pill_purple_bg']};
                    color: {p['pill_purple_text']};
                    border-color: {p['pill_purple_border']}; }}
    .pill.amber  {{ background: {p['pill_amber_bg']};
                    color: {p['pill_amber_text']};
                    border-color: {p['pill_amber_border']}; }}
    .pill.red    {{ background: {p['pill_red_bg']};
                    color: {p['pill_red_text']};
                    border-color: {p['pill_red_border']}; }}

    /* ---------- STATS ---------- */
    .stat {{
        padding: 1.5rem; border-radius: 20px;
        background: {p['card_bg']}; border: 1px solid {p['border']};
        box-shadow: {p['shadow_sm']}; text-align: center;
    }}
    .stat .num {{ font-size: 2.2rem; font-weight: 800;
                  color: {p['accent']}; margin: 0; line-height: 1.1; }}
    .stat .lbl {{ font-size: 0.75rem; color: {p['text_muted']};
                  text-transform: uppercase; letter-spacing: 0.1em;
                  margin-top: 0.5rem; font-weight: 700; }}

    /* ---------- TRACE ---------- */
    .trace {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem; color: {p['text_muted']};
        padding: 6px 0 6px 12px;
        border-left: 2px solid {p['accent']};
        margin-bottom: 4px; background: {p['bg_soft']};
        border-radius: 0 8px 8px 0;
    }}
    .trace::before {{ content: "› "; color: {p['accent']}; font-weight: bold; }}

    /* ---------- PROSE ---------- */
    .prose {{ max-width: 820px; color: {p['text_muted']};
              font-size: 1.02rem; line-height: 1.75; }}
    .prose h3 {{ margin-top: 1.8rem; margin-bottom: 0.4rem;
                 color: {p['accent']}; font-size: 1.15rem; font-weight: 700; }}
    .prose p {{ margin: 0 0 1rem 0; }}
    .prose strong {{ color: {p['text']}; }}

    /* ---------- DEPLOYMENT LOCK ---------- */
    .deploy-locked {{ position: relative; overflow: hidden; border-radius: 20px; }}
    .deploy-locked .deploy-inner {{
        background: {p['card_bg']}; border: 1px solid {p['border']};
        border-radius: 20px; padding: 1.5rem 1.7rem;
    }}
    .deploy-locked .deploy-inner h4 {{ color: {p['pill_amber_text']}; margin: 0 0 0.5rem 0; }}
    .deploy-locked .deploy-inner p {{ color: {p['text_muted']}; margin: 0; font-size: 0.9rem; }}
    .deploy-locked .lock-overlay {{
        position: absolute; inset: 0;
        display: flex; align-items: center; justify-content: center;
        flex-direction: column; gap: 0.4rem;
        background: {p['bg']}ee; opacity: 0;
        transition: opacity 0.3s ease;
        border-radius: 20px; backdrop-filter: blur(4px);
        pointer-events: none; z-index: 10;
    }}
    .deploy-locked:hover .lock-overlay {{ opacity: 1; }}
    .deploy-locked .lock-overlay .lock-icon {{ font-size: 2.4rem; }}
    .deploy-locked .lock-overlay .lock-text {{ color: {p['pill_amber_text']};
                                                 font-weight: 700; font-size: 1rem; }}
    .deploy-locked .lock-overlay .lock-sub {{ color: {p['text_muted']};
                                                font-size: 0.82rem;
                                                max-width: 380px; text-align: center; }}

    /* ---------- BUTTONS ---------- */
    .stButton button[kind="primary"] {{
        background: {p['button_primary_bg']};
        color: {p['button_primary_text']};
        font-weight: 700; border: none;
        box-shadow: 0 4px 14px {p['glow']};
        border-radius: 12px; padding: 0.6rem 1.5rem;
    }}
    .stButton button[kind="primary"]:hover {{
        background: {p['button_primary_hover']};
        box-shadow: 0 6px 20px {p['glow']};
        transform: translateY(-1px);
    }}
    .stButton button:not([kind="primary"]) {{
        background: {p['button_secondary_bg']};
        color: {p['button_secondary_text']};
        border: 1px solid {p['button_secondary_border']};
        border-radius: 12px; padding: 0.6rem 1.5rem; font-weight: 600;
    }}
    .stButton button:not([kind="primary"]):hover {{
        border-color: {p['btn_secondary_hover_border']};
        color: {p['btn_secondary_hover_text']};
        background: {p['btn_secondary_hover_bg']};
    }}

    .stProgress > div > div > div div {{ background: {p['progress_grad']}; }}
    hr {{ border-color: {p['border']}; }}

    /* ---------- WATERMARK ---------- */
    .wm {{
        position: fixed; bottom: 0; left: 0; right: 0;
        text-align: center; padding: 1rem;
        background: linear-gradient(180deg, transparent, {p['bg']}dd 60%);
        color: {p['text_dim']}; font-size: 0.8rem;
        z-index: 900; pointer-events: none;
    }}
    .wm b {{ color: {p['text_muted']}; font-weight: 600; }}
    .wm .brand {{ color: {p['accent']}; font-weight: 700; }}
    .wm .sep {{ opacity: 0.5; margin: 0 0.5rem; }}

    /* ---------- SISYPHUS ---------- */
    .sisyphus {{
        margin-top: 3rem; margin-bottom: 2rem; text-align: right;
        color: {p['text_dim']}; font-style: italic; font-size: 0.95rem;
    }}
    .sisyphus em {{ color: {p['accent']}; font-style: italic; }}

    /* ---------- BYTE VISUAL ---------- */
    .byte-visual {{
        position: fixed; bottom: 25px; right: 25px;
        z-index: 9998; pointer-events: none;
        animation: bobByte 3s ease-in-out infinite;
    }}
    @keyframes bobByte {{
        0%, 100% {{ transform: translateY(0); }}
        50%      {{ transform: translateY(-8px); }}
    }}
    .byte {{
        position: relative; width: 80px; height: 80px;
    }}
    .byte-body {{
        position: absolute; bottom: 12px; left: 50%;
        transform: translateX(-50%);
        width: 52px; height: 40px;
        background: linear-gradient(180deg, #FCD34D 0%, #F59E0B 100%);
        border: 2px solid #78350F; border-radius: 6px;
        box-shadow: inset -4px -4px 8px rgba(0,0,0,0.25),
                    0 0 14px {p['glow']};
    }}
    .byte-body::after {{
        content: ""; position: absolute; bottom: 6px; left: 8px;
        width: 14px; height: 6px; background: #78350F; border-radius: 2px;
    }}
    .byte-neck {{
        position: absolute; bottom: 52px; left: 50%;
        transform: translateX(-50%);
        width: 6px; height: 10px; background: #78350F; border-radius: 2px;
    }}
    .byte-head {{
        position: absolute; bottom: 62px; left: 50%;
        transform: translateX(-50%);
        width: 46px; height: 22px;
        background: linear-gradient(180deg, #FCD34D 0%, #F59E0B 100%);
        border: 2px solid #78350F; border-radius: 11px;
        display: flex; align-items: center; justify-content: center;
        gap: 6px;
    }}
    .byte-eye {{
        width: 12px; height: 12px; background: #0EA5E9;
        border: 2px solid #78350F; border-radius: 50%;
        box-shadow: 0 0 6px rgba(14, 165, 233, 0.8);
        animation: blinkByte 4s infinite;
    }}
    @keyframes blinkByte {{
        0%, 90%, 100% {{ transform: scaleY(1); }}
        94% {{ transform: scaleY(0.1); }}
    }}
    .byte-treads {{
        position: absolute; bottom: 4px; left: 50%;
        transform: translateX(-50%);
        width: 66px; height: 14px;
        background: #1F2937; border: 2px solid #111827; border-radius: 8px;
    }}
    .byte-treads::before, .byte-treads::after {{
        content: ""; position: absolute; top: 50%;
        width: 6px; height: 6px; background: #4B5563;
        border-radius: 50%; transform: translateY(-50%);
    }}
    .byte-treads::before {{ left: 8px; }}
    .byte-treads::after  {{ right: 8px; }}

    /* Invisible click overlay on Byte */
    .st-key-byte_button {{
        position: fixed !important;
        bottom: 25px !important;
        right: 25px !important;
        width: 80px !important;
        height: 80px !important;
        z-index: 10000 !important;
        background: transparent !important;
    }}
    .st-key-byte_button .stButton > button {{
        width: 80px !important;
        height: 80px !important;
        opacity: 0 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        cursor: pointer !important;
    }}

    /* Modal look — dialog overlay gets themed */
    div[role="dialog"] {{
        background-color: {p['card_bg']} !important;
        border: 2px solid {p['accent']} !important;
        border-radius: 20px !important;
        box-shadow: 0 20px 50px rgba(0,0,0,0.4),
                    0 0 30px {p['glow']} !important;
    }}
    div[role="dialog"] h1,
    div[role="dialog"] h2,
    div[role="dialog"] h3,
    div[role="dialog"] h4 {{
        color: {p['text']} !important;
    }}
    div[role="dialog"] p,
    div[role="dialog"] span,
    div[role="dialog"] div {{
        color: {p['text_muted']};
    }}

    section[data-testid="stSidebar"] {{ display: none; }}
</style>
"""


# ============================================================
# PUBLIC INJECT
# ============================================================
def inject():
    theme = st.session_state.get("theme", "light")
    p = _palette(theme)
    st.markdown(_build_theme(p), unsafe_allow_html=True)


# ============================================================
# HTML HELPERS
# ============================================================
def _render_html(html: str):
    try:
        st.html(html)
    except AttributeError:
        cleaned = textwrap.dedent(html)
        cleaned = "\n".join(line.lstrip() for line in cleaned.split("\n"))
        st.markdown(cleaned, unsafe_allow_html=True)


# ============================================================
# SMOOTH INTERACTIVE SVG ROBOT
# ============================================================
def render_robot(caption="Agent is ready..."):
    theme = st.session_state.get("theme", "light")
    p = _palette(theme)

    svg_html = f'''
    <style>
        body {{ background-color: transparent; margin: 0; padding: 0; overflow: hidden; display: flex; justify-content: center; align-items: center; height: 100vh; }}
        .robot-arm {{ transition: transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1); transform-origin: 150px 120px; }}
        .robot-wrapper:hover .robot-arm {{ transform: rotate(-45deg) translate(10px, -20px); }}
        .blink {{ animation: blinker 4s infinite; }}
        @keyframes blinker {{ 0%, 90%, 100% {{ opacity: 1; }} 95% {{ opacity: 0; }} }}
        @keyframes float {{ 0%, 100% {{ transform: translateY(0px); }} 50% {{ transform: translateY(-10px); }} }}
    </style>
    <svg width="220" height="220" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" style="filter: drop-shadow(0 15px 25px {p['glow']});">
        <g class="robot-wrapper" style="animation: float 4s ease-in-out infinite; cursor: pointer;">
            <line x1="100" y1="40" x2="100" y2="20" stroke="{p['text_muted']}" stroke-width="4" stroke-linecap="round"/>
            <circle cx="100" cy="15" r="6" fill="{p['accent']}"><animate attributeName="opacity" values="0.3;1;0.3" dur="2s" repeatCount="indefinite"/></circle>
            <rect x="50" y="90" width="100" height="80" rx="20" fill="{p['robot_body']}" stroke="{p['robot_border']}" stroke-width="3"/>
            <rect x="70" y="110" width="60" height="40" rx="10" fill="{p['bg']}" stroke="{p['border']}" stroke-width="2"/>
            <circle cx="85" cy="130" r="5" fill="{p['accent']}"/>
            <circle cx="115" cy="130" r="5" fill="{p['accent']}"/>
            <rect x="30" y="100" width="20" height="50" rx="10" fill="{p['robot_body']}" stroke="{p['robot_border']}" stroke-width="3"/>
            <g class="robot-arm">
                <rect x="150" y="100" width="20" height="50" rx="10" fill="{p['robot_body']}" stroke="{p['robot_border']}" stroke-width="3"/>
                <circle cx="160" cy="155" r="12" fill="{p['robot_body']}" stroke="{p['robot_border']}" stroke-width="3"/>
            </g>
            <rect x="55" y="40" width="90" height="50" rx="25" fill="{p['robot_bg']}" stroke="{p['robot_border']}" stroke-width="3"/>
            <rect x="70" y="55" width="60" height="20" rx="10" fill="{p['bg']}" stroke="{p['border']}" stroke-width="2"/>
            <circle class="blink" cx="85" cy="65" r="6" fill="{p['robot_eye']}"/>
            <circle class="blink" cx="115" cy="65" r="6" fill="{p['robot_eye']}"/>
        </g>
    </svg>
    '''

    components.html(svg_html, height=230, scrolling=False)

    if caption:
        st.markdown(
            f'<p style="text-align:center;margin-top:-1rem;font-size:0.9rem;color:{p["text_muted"]};">{caption}</p>',
            unsafe_allow_html=True
        )


# ============================================================
# TRAINING GAME (TIC-TAC-TOE) — fixed minimax side-effects
# ============================================================
def render_training_game():
    """Embeds a playable Tic-Tac-Toe game to entertain users during training."""
    p = _palette(st.session_state.get("theme", "light"))

    style = f"""
        body {{
            font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            background: transparent; margin: 0; padding: 10px;
            color: {p['text_muted']};
        }}
        .board {{
            display: grid;
            grid-template-columns: repeat(3, 70px);
            grid-template-rows: repeat(3, 70px);
            gap: 8px; margin-bottom: 15px;
        }}
        .cell {{
            background: {p['card_bg']};
            border: 2px solid {p['border']};
            border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            font-size: 2.2rem; font-weight: 800; cursor: pointer;
            transition: all 0.2s ease; color: {p['accent']};
        }}
        .cell:hover:not(.x):not(.o) {{
            background: {p['accent_soft']};
            border-color: {p['accent']};
        }}
        .cell.x {{ color: {p['accent']}; text-shadow: 0 0 10px {p['glow']}; }}
        .cell.o {{ color: {p['accent_2']}; text-shadow: 0 0 10px {p['glow_2']}; }}
        .status {{
            font-size: 1.1rem; margin-bottom: 15px;
            font-weight: 600; color: {p['text']};
            min-height: 1.5rem;
        }}
        button {{
            background: {p['button_secondary_bg']};
            color: {p['button_secondary_text']};
            border: 1px solid {p['button_secondary_border']};
            padding: 8px 16px; border-radius: 8px;
            cursor: pointer; font-weight: 600;
        }}
        button:hover {{
            border-color: {p['accent']};
            color: {p['accent']};
        }}
    """

    body = """
        <div class="status" id="status">Your turn (X)</div>
        <div class="board">
            <div class="cell" data-i="0"></div>
            <div class="cell" data-i="1"></div>
            <div class="cell" data-i="2"></div>
            <div class="cell" data-i="3"></div>
            <div class="cell" data-i="4"></div>
            <div class="cell" data-i="5"></div>
            <div class="cell" data-i="6"></div>
            <div class="cell" data-i="7"></div>
            <div class="cell" data-i="8"></div>
        </div>
        <button onclick="resetGame()">Reset Game</button>

        <script>
            let board = ["", "", "", "", "", "", "", "", ""];
            let currentPlayer = "X";
            let gameActive = true;

            const winConditions = [
                [0, 1, 2], [3, 4, 5], [6, 7, 8],
                [0, 3, 6], [1, 4, 7], [2, 5, 8],
                [0, 4, 8], [2, 4, 6]
            ];

            const cells = document.querySelectorAll('.cell');
            const statusDisplay = document.getElementById('status');

            // Pure outcome check — must NOT mutate any game state.
            function getOutcome(b) {
                for (const [i, j, k] of winConditions) {
                    if (b[i] && b[i] === b[j] && b[i] === b[k]) return b[i];
                }
                if (b.every(v => v !== "")) return "draw";
                return null;
            }

            cells.forEach(cell => cell.addEventListener('click', handleCellClick));

            function handleCellClick(e) {
                const index = parseInt(e.target.getAttribute('data-i'));
                if (board[index] !== "" || !gameActive || currentPlayer !== "X") return;

                makeMove(index, "X");
                const outcome = getOutcome(board);
                if (outcome === "X") {
                    statusDisplay.innerText = "You win! 🎉";
                    gameActive = false;
                    return;
                }
                if (outcome === "draw") {
                    statusDisplay.innerText = "It's a draw!";
                    gameActive = false;
                    return;
                }
                currentPlayer = "O";
                statusDisplay.innerText = "AI is thinking...";
                setTimeout(aiMove, 400);
            }

            function makeMove(index, player) {
                board[index] = player;
                const cell = cells[index];
                cell.innerText = player;
                cell.classList.add(player.toLowerCase());
            }

            function aiMove() {
                if (!gameActive) return;
                let bestScore = -Infinity;
                let move = -1;
                for (let i = 0; i < 9; i++) {
                    if (board[i] === "") {
                        board[i] = "O";
                        const score = minimax(board, 0, false);
                        board[i] = "";
                        if (score > bestScore) {
                            bestScore = score;
                            move = i;
                        }
                    }
                }
                if (move === -1) return;
                makeMove(move, "O");
                const outcome = getOutcome(board);
                if (outcome === "O") {
                    statusDisplay.innerText = "AI wins! 🤖";
                    gameActive = false;
                    return;
                }
                if (outcome === "draw") {
                    statusDisplay.innerText = "It's a draw!";
                    gameActive = false;
                    return;
                }
                currentPlayer = "X";
                statusDisplay.innerText = "Your turn (X)";
            }

            function minimax(b, depth, isMaximizing) {
                const outcome = getOutcome(b);
                if (outcome === "O") return 10 - depth;
                if (outcome === "X") return depth - 10;
                if (outcome === "draw") return 0;

                if (isMaximizing) {
                    let best = -Infinity;
                    for (let i = 0; i < 9; i++) {
                        if (b[i] === "") {
                            b[i] = "O";
                            best = Math.max(best, minimax(b, depth + 1, false));
                            b[i] = "";
                        }
                    }
                    return best;
                } else {
                    let best = Infinity;
                    for (let i = 0; i < 9; i++) {
                        if (b[i] === "") {
                            b[i] = "X";
                            best = Math.min(best, minimax(b, depth + 1, true));
                            b[i] = "";
                        }
                    }
                    return best;
                }
            }

            function resetGame() {
                board = ["", "", "", "", "", "", "", "", ""];
                currentPlayer = "X";
                gameActive = true;
                statusDisplay.innerText = "Your turn (X)";
                cells.forEach(cell => {
                    cell.innerText = "";
                    cell.classList.remove('x', 'o');
                });
            }
        </script>
    """

    html = ("<!DOCTYPE html><html><head><meta charset='utf-8'><style>"
            + style + "</style></head><body>" + body + "</body></html>")
    components.html(html, height=420)


# ============================================================
# BYTE — the corner robot + modal chat
# ============================================================
def render_byte(tooltip="Hi, I'm Byte! Click me to chat."):
    """Render Byte's visual + invisible clickable overlay."""
    _render_html(
        '<div class="byte-visual" title="' + tooltip + '">'
        '<div class="byte">'
        '<div class="byte-head">'
        '<div class="byte-eye"></div><div class="byte-eye"></div>'
        '</div>'
        '<div class="byte-neck"></div>'
        '<div class="byte-body"></div>'
        '<div class="byte-treads"></div>'
        '</div></div>'
    )
    with st.container(key="byte_button"):
        if st.button(" ", key="byte_click", use_container_width=True):
            st.session_state["_byte_modal"] = "chat"
            st.rerun()


@st.dialog("👋 Hi, I'm Byte!", width="large")
def _byte_tour_dialog():
    st.markdown(
        "I'm your little guide for ML in a Box. Here's what each page does:"
    )
    items = [
        ("🏠 Home", "the landing page — quick-start buttons and a look at the pipeline."),
        ("🔎 Data Profiler", "upload a dataset (CSV/Excel) or a saved model (.zip)."),
        ("⚡ Run Pipeline", "train models, explain decisions, download the result."),
        ("📄 Doc Tools", "upload documents, ask questions with citations, summarize."),
        ("🤖 Agent Builder", "design your own AI agent and download a blueprint."),
        ("📖 Knowledge", "interactive concept map — AI, ML, DL, LLM, CV, NLP, Agentic."),
        ("ℹ️ About", "who built this and what it does."),
    ]
    for title, desc in items:
        st.markdown(f"**{title}** — {desc}")

    st.markdown("---")
    st.caption("Click me any time in the corner and I'll answer questions.")

    if st.button("Got it — let's go!", type="primary",
                 use_container_width=True, key="tour_close"):
        st.rerun()


@st.dialog("🤖 Byte — your assistant", width="large")
def _byte_chat_dialog(context_hint=""):
    st.caption("Ask me anything about this page, the app, or ML concepts.")

    if "byte_chat" not in st.session_state:
        st.session_state.byte_chat = []

    # Chat history
    for role, text in st.session_state.byte_chat[-8:]:
        if role == "user":
            st.markdown(
                f'<div style="background:{_palette(st.session_state.get("theme","light"))["accent_soft"]};'
                f'padding:0.6rem 0.9rem;border-radius:12px;margin-bottom:0.5rem;'
                f'font-size:0.9rem;line-height:1.4;">'
                f'<b>You:</b> {text}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="background:{_palette(st.session_state.get("theme","light"))["pill_purple_bg"]};'
                f'padding:0.6rem 0.9rem;border-radius:12px;margin-bottom:0.5rem;'
                f'font-size:0.9rem;line-height:1.4;">'
                f'<b>Byte:</b> {text}</div>',
                unsafe_allow_html=True,
            )

    # Input
    with st.form("byte_chat_form", clear_on_submit=True):
        q = st.text_input("Ask Byte", placeholder="e.g. what does this page do?",
                          label_visibility="collapsed", key="byte_input")
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            sent = st.form_submit_button("Send", use_container_width=True,
                                          type="primary")
        with c2:
            clear = st.form_submit_button("Clear", use_container_width=True)
        with c3:
            close = st.form_submit_button("Close", use_container_width=True)

    if close:
        st.session_state["_byte_modal"] = None
        st.rerun()

    if clear:
        st.session_state.byte_chat = []
        st.rerun()

    if sent and q.strip():
        st.session_state.byte_chat.append(("user", q.strip()))
        with st.spinner("Byte is thinking…"):
            a = _byte_answer(q.strip(), context_hint)
        st.session_state.byte_chat.append(("assistant", a))
        st.rerun()


def render_byte_chat(context_hint="", show_tour_if_new=True):
    """
    Called from app.py after render_byte(). Handles:
      - First-visit onboarding tour (once per session)
      - Modal chat while `_byte_modal == "chat"`
    """
    if (show_tour_if_new
            and not st.session_state.get("_byte_tour_done")):
        st.session_state["_byte_tour_done"] = True
        _byte_tour_dialog()

    if st.session_state.get("_byte_modal") == "chat":
        _byte_chat_dialog(context_hint)


def _byte_answer(question: str, context_hint: str) -> str:
    """Ask the LLM with Byte's personality, honoring language setting."""
    from llm import ask as _llm_ask
    lang = st.session_state.get("lang", "en")
    lang_line = ("Answer in **Arabic**." if lang == "ar"
                 else "Answer in English.")
    prompt = (
        "You are Byte — a friendly Wall-E-style robot assistant inside the "
        "app 'ML in a Box'. Answer in 2-4 short, warm sentences. Plain language, "
        "no jargon unless needed. If you don't know, say so honestly. "
        f"{lang_line}\n\n"
        f"--- Page context ---\n{context_hint or 'No page context.'}\n---\n\n"
        f"User asks: {question}"
    )
    try:
        return (_llm_ask(prompt, temperature=0.5) or "").strip()
    except Exception as e:
        return f"(Byte is offline right now: {type(e).__name__})"


# ============================================================
# PUBLIC HELPERS
# ============================================================
def watermark():
    st.markdown(
        '<div class="wm"><b>Engr. Anas Yousef Albakoush</b>'
        '<span class="sep">·</span>'
        '<b>Engr. Muad Salah Himeedah</b>'
        '<span class="sep">·</span>'
        '<span class="brand">ML in a Box</span></div>',
        unsafe_allow_html=True)


def sisyphus():
    st.markdown(
        '<div class="sisyphus">— <em>One must imagine Sisyphus happy</em></div>',
        unsafe_allow_html=True)


def hero(title, subtitle, pills=None):
    pills_html = ""
    if pills:
        pills_html = " ".join(f'<span class="pill">{p}</span>' for p in pills)
    _render_html(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>{pills_html}</div>')


def card(icon, title, body, pill=None):
    pill_html = f'<span class="pill">{pill}</span>' if pill else ""
    _render_html(
        f'<div class="card"><span class="icon">{icon}</span>'
        f'<h4>{title} {pill_html}</h4><p>{body}</p></div>')


def stat(num, label):
    _render_html(
        f'<div class="stat"><p class="num">{num}</p>'
        f'<div class="lbl">{label}</div></div>')


def trace_line(text):
    st.markdown(f'<div class="trace">{text}</div>', unsafe_allow_html=True)


def prose(html: str):
    cleaned = textwrap.dedent(html).strip()
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
    _render_html(f'<div class="prose">{cleaned}</div>')


def deployment_locked(title="Deploy to production",
                      body="Package the model as a REST API, add auth, and "
                           "serve it behind a load balancer.",
                      sub="This build is a research prototype."):
    _render_html(
        f'<div class="deploy-locked"><div class="deploy-inner">'
        f'<h4>🚀 {title}</h4><p>{body}</p></div>'
        f'<div class="lock-overlay"><div class="lock-icon">🔒</div>'
        f'<div class="lock-text">Deployment is locked</div>'
        f'<div class="lock-sub">{sub}</div></div></div>')


def profile_photo(path, fallback_icon="👤"):
    if path and os.path.isfile(path):
        st.image(path, use_container_width=True)
    else:
        st.markdown(
            f'<div style="text-align:center;font-size:4rem;padding:1rem;">'
            f'{fallback_icon}</div>', unsafe_allow_html=True)


# ============================================================
# GLOSSARY (kept for i18n.py compatibility)
# ============================================================
GLOSSARY_CATEGORIES = {
    "Regression metrics": [
        ("R² (R-squared)", "How much of the target's variation the model explains.",
         "1.0 is perfect.", "Above 0.7 is strong."),
        ("MAE", "Average prediction error.", "Mean abs error.", "Lower is better."),
        ("RMSE", "Like MAE, punishes big errors.", "Sqrt of MSE.", "Lower is better."),
    ],
    "Classification metrics": [
        ("Accuracy", "Fraction correct.", "Matches / total.", "Baseline check."),
        ("Precision", "True positives / (TP+FP).", "TP/(TP+FP).", "Higher better."),
        ("Recall", "TP / (TP+FN).", "TP/(TP+FN).", "Higher better."),
        ("F1", "Harmonic mean.", "Balance of both.", "0.9+ excellent."),
        ("Confusion matrix", "Predictions vs reality.", "Diagonal correct.",
         "Off-diag is mistakes."),
    ],
    "ML models": [
        ("Linear Regression", "Straight-line fit.", "Weighted sum.", "Fast, explainable."),
        ("Logistic Regression", "Linear classifier.", "Sigmoid.", "Interpretable."),
        ("Ridge / Lasso", "Regularized linear.", "Penalty on size.", "Ridge for correlation."),
        ("Decision Tree", "Yes/no questions.", "Split on feature.", "Interpretable, overfits."),
        ("Random Forest", "Many trees.", "Random subsets.", "Robust."),
        ("Gradient Boosting", "Trees fixing mistakes.", "Sequential.", "Often strongest."),
        ("Extra Trees", "More random forest.", "Random splits.", "Fast."),
        ("AdaBoost", "Boosting with stumps.", "Focus mistakes.", "Simple signal."),
        ("Bagging", "Random subsets.", "Vote/average.", "Variance reduction."),
        ("KNN", "Nearest neighbors.", "K closest.", "Sensitive to scale."),
        ("SVM", "Widest boundary.", "Margin max.", "Small clean data."),
        ("Naive Bayes", "Independent features.", "Probabilistic.", "Fast on text."),
        ("Neural Network (MLP)", "Layers of units.", "Backprop.", "Needs more data."),
        ("Deep Learning", "Many-layer nets.", "Scale up MLP.", "Best for images/text."),
    ],
    "Training concepts": [
        ("Baseline", "Dumbest model.", "Predicts average.", "Beat it."),
        ("Overfitting", "Memorised training.", "Train high, test low.", "Agent fixes."),
        ("Underfitting", "Too simple.", "Both low.", "Flexible model."),
        ("Class imbalance", "One class dominates.", "Accuracy lies.", "Warn at 5x."),
        ("Cross-validation", "Score over folds.", "Average.", "Compare to test."),
        ("Train/test split", "Learn/check split.", "Model never sees test.", "80/20 typical."),
        ("Feature importance", "Which inputs matter.", "Shuffle watch.", "Bigger = more."),
        ("Regularization", "Complexity penalty.", "Loss term.", "Smoother."),
        ("Hyperparameter", "Pre-training setting.", "Trees/LR.", "Setup panel."),
        ("Feature engineering", "New inputs from old.", "Ratios, dates.", "High leverage."),
        ("Interaction", "Combined columns.", "a*b.", "Joint effects."),
        ("Imputation", "Fill missing.", "Median/mode.", "Auto-proposed."),
        ("Standardization", "Rescale.", "Z-score.", "Distance models."),
        ("One-hot", "Categorical to binary.", "0/1 columns.", "Standard."),
    ],
    "Agentic AI": [
        ("Agentic AI", "LLM in a loop.", "Plan/execute/review.", "Four decisions."),
        ("Human-in-the-loop", "Pause for approval.", "Graph interrupts.", "Makes it agentic."),
        ("Guardrail", "Code check.", "Validate JSON.", "Logged."),
        ("Tool calling", "LLM runs a function.", "Structured request.", "Prevents arbitrary."),
        ("State", "What the agent remembers.", "LangGraph dict.", "Checkpointer saves."),
        ("Checkpoint", "Pause/resume point.", "interrupt().", "Real pause."),
        ("Self-check", "Inspects own output.", "Compares scores.", "Accept or fix."),
        ("Run history", "Log of runs.", "Reads before deciding.", "How agent learns."),
    ],
    "NLP": [
        ("Tokenization", "Split into units.", "First step.", "Feeds models."),
        ("Stop-words", "Common low-meaning.", "Often removed.", "Sometimes bad."),
        ("TF-IDF", "Frequency / rarity.", "Distinctive weigh more.", "Numeric matrix."),
        ("Bag of words", "Counts, no order.", "Vector of counts.", "Fast."),
        ("N-gram", "N consecutive words.", "Bigrams.", "Phrases."),
        ("Cleaning", "Strip HTML, etc.", "Toggles.", "Check stats."),
        ("Extractive summary", "Pick best sentences.", "Offline.", "Safe."),
        ("Abstractive summary", "Rewrite shorter.", "Uses LLM.", "Best for long text."),
        ("Topic clustering", "Group by theme.", "KMeans.", "Silhouette > 0.2."),
        ("Silhouette", "Cluster separation.", "-1 to 1.", "0.5+ strong."),
        ("Cosine similarity", "Doc similarity.", "1 = same.", "Search."),
        ("Sentiment analysis", "Pos/neg/neutral.", "LLM scores sample.", "Train classifier for prod."),
        ("Word cloud", "Frequent words larger.", "Whole corpus.", "Quick glance."),
    ],
}

GLOSSARY = {
    term: (short, long_, how)
    for entries in GLOSSARY_CATEGORIES.values()
    for term, short, long_, how in entries
}