import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Safely attempt to read from Streamlit secrets, otherwise fall back to .env
try:
    api_key = st.secrets.get("LLM_API_KEY", os.getenv("LLM_API_KEY"))
    base_url = st.secrets.get("LLM_BASE_URL", os.getenv("LLM_BASE_URL"))
    model_name = st.secrets.get("LLM_MODEL", os.getenv("LLM_MODEL"))
except Exception:
    # This catches the StreamlitSecretNotFoundError
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model_name = os.getenv("LLM_MODEL")

# Don't instantiate the client here — that would raise if api_key is None,
# crashing the app at import time. We build it lazily on the first `ask()`.
client = None
MODEL = model_name or "gpt-3.5-turbo"


def ask(prompt, system="You are a helpful assistant.", temperature=0.2):
    global client
    if not api_key:
        raise ValueError(
            "LLM_API_KEY is not set. Add it to your `.env` file or Streamlit "
            "secrets. Until then, LLM features are unavailable, but the rest "
            "of the app still works."
        )
    if client is None:
        client = OpenAI(api_key=api_key, base_url=base_url)

    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return r.choices[0].message.content