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

client = OpenAI(api_key=api_key, base_url=base_url)
MODEL = model_name

def ask(prompt, system="You are a helpful assistant.", temperature=0.2):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return r.choices[0].message.content