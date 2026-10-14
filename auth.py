"""
Login and role-based access control.

Users are defined in the USERS dict below. Each user has:
  - password_hash   sha256 of the plaintext password (precomputed here for demo)
  - display_name    shown in the sidebar
  - modalities      list of what they can use: "ml", "dl", "nlp", "cv"
  - role            free-text label for the badge

This is demo-grade auth. For a real product, replace the dict with a database,
add per-user salts, and rate-limit login attempts.
"""
import hashlib
import streamlit as st


_SALT = "ml-in-a-box::v1::"


def _hash(password: str) -> str:
    return hashlib.sha256((_SALT + password).encode("utf-8")).hexdigest()


# Passwords are hashed at import time. Plaintext lives here only for demo
# convenience — swap for a real credential store before any real use.
_RAW_USERS = {
    "admin": {
        "password": "admin123",
        "display_name": "Administrator",
        "modalities": ["ml", "dl", "nlp", "cv"],
        "role": "admin",
    },
    "doctor": {
        "password": "doctor123",
        "display_name": "Dr. Evaluator",
        "modalities": ["ml", "dl", "nlp"],
        "role": "evaluator",
    },
    "jabo": {
        "password": "jabo123",
        "display_name": "jababwe",
        "modalities": ["ml", "dl"],
        "role": "student",
    },
    "nlp_user": {
        "password": "nlp123",
        "display_name": "NLP Researcher",
        "modalities": ["nlp"],
        "role": "researcher",
    },
    "sambatica": {
        "password": "sambatica123",
        "display_name": "Sambatica Researcher",
        "modalities": ["cv"],
        "role": "researcher",
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
    """Return the user dict on success, or None."""
    user = USERS.get(username)
    if not user:
        return None
    if user["password_hash"] != _hash(password):
        return None
    return {"username": username, **user}


def current_user():
    """Return the logged-in user dict, or None."""
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
    return modality in u["modalities"]


def modality_labels() -> list:
    u = current_user()
    if u is None:
        return []
    return [m.upper() for m in u["modalities"]]


def render_login():
    """Render the full-screen login page. Returns nothing; caller should st.stop()."""
    # Hide the sidebar on the login page by injecting CSS
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] { display: none; }
            .login-wrap {
                max-width: 460px; margin: 6vh auto 0 auto;
            }
            .login-card {
                background: linear-gradient(180deg, #131a2e 0%, #10172a 100%);
                border: 1px solid #2a3350; border-radius: 20px;
                padding: 2.4rem 2.2rem;
                box-shadow: 0 24px 60px rgba(34,211,238,0.08);
            }
            .login-card h1 {
                font-size: 2rem; margin: 0 0 0.4rem 0;
                background: linear-gradient(135deg, #67e8f9, #e879f9);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                background-clip: text; font-weight: 800;
            }
            .login-card p.sub {
                color: #94a3b8; margin: 0 0 1.6rem 0; font-size: 0.95rem;
            }
            .demo-hint {
                margin-top: 1.4rem; padding: 0.9rem 1.1rem;
                background: rgba(34,211,238,0.06);
                border: 1px dashed rgba(34,211,238,0.35);
                border-radius: 12px;
                color: #94a3b8; font-size: 0.8rem; line-height: 1.6;
            }
            .demo-hint code {
                background: #1c2540; color: #67e8f9;
                padding: 1px 6px; border-radius: 5px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(
            '<div class="login-card">'
            '<h1>🤖 ML in a Box</h1>'
            '<p class="sub">Sign in to continue.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        with st.form("login_form"):
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            submitted = st.form_submit_button("Sign in", use_container_width=True,
                                               type="primary")

        if submitted:
            if login(username.strip(), password):
                st.success("Signed in.")
                st.rerun()
            else:
                st.error("Wrong username or password.")

        st.markdown(
            '<div class="demo-hint">'
            '<b>Demo accounts</b><br>'
            'admin / <code>admin123</code> &nbsp;·&nbsp; full access (ML, DL, NLP, CV)<br>'
            'doctor / <code>doctor123</code> &nbsp;·&nbsp; ML, DL, NLP<br>'
            'jabo / <code>jabo123</code> &nbsp;·&nbsp; ML, DL<br>'
            'nlp_user / <code>nlp123</code> &nbsp;·&nbsp; NLP only<br>'
            'sambatica / <code>sambatica123</code> &nbsp;·&nbsp; ML, CV<br>'
            '</div>',
            unsafe_allow_html=True,
        )