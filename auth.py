"""
Login and role-based access control.

Roles:
  admin     — full access to everything
  visitor   — only Home, About, Knowledge, Tutorial
  others    — access is defined per-user by their modalities list
"""
import hashlib
import streamlit as st


_SALT = "ml-in-a-box::v2::"


def _hash(password: str) -> str:
    return hashlib.sha256((_SALT + password).encode("utf-8")).hexdigest()


_RAW_USERS = {
    "admin": {
        "password": "admin123",
        "display_name": "Administrator",
        "modalities": ["ml", "dl", "nlp", "cv"],
        "role": "admin",
    },
    "penguin": {
        "password": "penguin123",
        "display_name": "Penguin",
        "modalities": ["ml", "dl", "nlp", "cv"],
        "role": "admin",
    },
    "doctor": {
        "password": "doctor123",
        "display_name": "Dr. Evaluator",
        "modalities": ["ml", "dl", "nlp"],
        "role": "evaluator",
    },
    "anas": {
        "password": "anas123",
        "display_name": "Anas",
        "modalities": ["ml", "dl", "nlp", "cv"],
        "role": "admin",
    },
    "muad": {
        "password": "muad123",
        "display_name": "Muad",
        "modalities": ["ml", "dl", "nlp", "cv"],
        "role": "admin",
    },
    "nus": {
        "password": "nus123",
        "display_name": "Nus",
        "modalities": ["ml", "dl", "nlp", "cv"],
        "role": "researcher",
    },
    "visitor": {
        "password": "visitor",
        "display_name": "Visitor",
        "modalities": [],
        "role": "visitor",
    },
}

USERS = {
    name: {
        "password_hash": _hash(info["password"]),
        "display_name": info["display_name"],
        "modalities": info["modalities"],
        "role": info["role"],
    }
    for name, info in _RAW_USERS.items()
}


def verify(username: str, password: str):
    user = USERS.get(username)
    if not user:
        return None
    if user["password_hash"] != _hash(password):
        return None
    return {"username": username, **user}


def current_user():
    return st.session_state.get("user")


def is_logged_in() -> bool:
    return current_user() is not None


def login(username: str, password: str) -> bool:
    user = verify(username, password)
    if user is None:
        return False
    st.session_state.user = user
    return True


def logout():
    st.session_state.pop("user", None)


def can_use(modality: str) -> bool:
    u = current_user()
    if u is None:
        return False
    if u.get("role") == "visitor":
        return False
    return modality in u["modalities"]


def modality_labels() -> list:
    u = current_user()
    if u is None:
        return []
    return [m.upper() for m in u["modalities"]]


def _login_css() -> str:
    theme = st.session_state.get("theme", "light")
    if theme == "light":
        bg = "#F9FAFB"
        card_bg = "#FFFFFF"
        border = "#D4AF37"
        accent = "#D4AF37"
        accent_text = "#FFFFFF"
        text = "#111827"
        text_muted = "#6B7280"
        shadow = "0 20px 50px -15px rgba(212, 175, 55, 0.25)"
        hint_bg = "#F9FAFB"
        hint_border = "#D4AF37"
        hint_text = "#4B5563"
        code_bg = "#F3F4F6"
        code_text = "#059669"
        input_border = "#E5E7EB"
    else:
        bg = "#050505"
        card_bg = "#0A0A0A"
        border = "#00F0FF"
        accent = "#00F0FF"
        accent_text = "#050505"
        text = "#FFFFFF"
        text_muted = "#A1A1AA"
        shadow = "0 20px 50px -15px rgba(0, 240, 255, 0.3)"
        hint_bg = "#111111"
        hint_border = "#00F0FF"
        hint_text = "#A1A1AA"
        code_bg = "#18181B"
        code_text = "#00F0FF"
        input_border = "#27272A"

    return f"""
    <style>
        header[data-testid="stHeader"] {{ display: none !important; }}
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        section[data-testid="stSidebar"] {{ display: none !important; }}

        * {{ transition: background-color 0.5s ease, color 0.5s ease,
                           border-color 0.5s ease, box-shadow 0.5s ease; }}

        .stApp {{
            background: {bg} !important;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }}

        .block-container {{
            padding-top: 8vh !important;
            max-width: 500px !important;
        }}

        .login-card {{
            background: {card_bg};
            border: 1px solid {border};
            border-radius: 24px;
            padding: 2.6rem 2.4rem;
            box-shadow: {shadow};
            text-align: center;
            margin-bottom: 1.5rem;
        }}
        .login-card h1 {{
            font-size: 2.1rem;
            margin: 0 0 0.5rem 0;
            color: {text};
            font-weight: 800;
            letter-spacing: -0.03em;
        }}
        .login-card p.sub {{
            color: {text_muted};
            margin: 0;
            font-size: 0.95rem;
        }}

        .demo-hint {{
            margin-top: 1.5rem;
            padding: 1.1rem 1.3rem;
            background: {hint_bg};
            border: 1px dashed {hint_border};
            border-radius: 16px;
            color: {hint_text};
            font-size: 0.82rem;
            line-height: 1.9;
            text-align: left;
        }}
        .demo-hint b {{
            color: {text};
            display: block;
            margin-bottom: 0.5rem;
            font-size: 0.85rem;
            letter-spacing: 0.02em;
        }}
        .demo-hint code {{
            background: {code_bg};
            color: {code_text};
            padding: 2px 7px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.78rem;
        }}

        /* Form submit button */
        .stFormSubmitButton button {{
            background: {accent} !important;
            color: {accent_text} !important;
            font-weight: 700 !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.7rem 1.5rem !important;
            width: 100% !important;
            font-size: 0.98rem !important;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15) !important;
            margin-top: 0.5rem;
        }}
        .stFormSubmitButton button:hover {{
            filter: brightness(0.92);
            transform: translateY(-1px);
        }}

        /* Inputs */
        .stTextInput input {{
            background: {card_bg} !important;
            border: 1px solid {input_border} !important;
            color: {text} !important;
            border-radius: 10px !important;
            padding: 0.65rem 0.9rem !important;
        }}
        .stTextInput input:focus {{
            border-color: {accent} !important;
            box-shadow: 0 0 0 3px {border}33 !important;
        }}
        .stTextInput label {{
            color: {text_muted} !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
        }}

        /* Success/error alerts */
        .stAlert {{
            border-radius: 12px !important;
            margin-top: 0.8rem !important;
        }}
    </style>
    """


def render_login():
    """Render the full-screen login page."""
    st.markdown(_login_css(), unsafe_allow_html=True)

    st.markdown(
        '<div class="login-card">'
        '<h1>🤖 ML in a Box</h1>'
        '<p class="sub">Sign in to continue.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input(
            "Username", key="login_user",
            placeholder="e.g. admin",
        )
        password = st.text_input(
            "Password", type="password",
            key="login_pass",
            placeholder="••••••••",
        )
        submitted = st.form_submit_button("Sign in", use_container_width=True)

    if submitted:
        if login(username.strip(), password):
            st.rerun()
        else:
            st.error("Wrong username or password.")

    st.markdown(
        '<div class="demo-hint">'
        '<b>Demo accounts</b>'
        'admin / <code>admin123</code><br>'
        'penguin / <code>penguin123</code><br>'
        'doctor / <code>doctor123</code><br>'
        'anas / <code>anas123</code><br>'
        'muad / <code>muad123</code><br>'
        'nus / <code>nus123</code><br>'
        'visitor / <code>visitor</code> &nbsp;·&nbsp; read-only tour'
        '</div>',
        unsafe_allow_html=True,
    )