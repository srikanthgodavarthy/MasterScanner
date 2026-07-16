"""
Groq client helper for free-tier LLM calls (news sentiment/impact tagging).

Mirrors utils/openai_client.py's pattern exactly, on purpose: pull credentials
from st.secrets (works identically locally via .streamlit/secrets.toml and on
Streamlit Community Cloud's secrets UI), cache the client, and fail soft
(return None) if not configured so the rest of the app keeps working without
it.

Why Groq and not another "free" provider
-----------------------------------------
Groq's API is OpenAI-SDK-compatible (same `openai` package already in
requirements.txt, just a different `base_url`), so it drops into the exact
same call pattern used by utils/openai_client.py / pages/agent.py without a
new dependency. Groq's free tier (as of mid-2026) covers Llama 3.x models
with generous per-minute/per-day request limits — comfortably enough for a
periodic batch-classification job over a few dozen headlines, which is all
utils/news_sentiment.py needs. If your Groq usage grows past the free tier,
or you'd rather standardize on Gemini, swap this module out; nothing else
in the news pipeline depends on Groq specifically — utils/news_sentiment.py
only calls get_client() / get_model() / _is_available() from here.

Usage
-----
from utils.groq_client import get_client, get_model, _is_available
"""

from __future__ import annotations

import logging

import streamlit as st

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Llama 3.3 70B — best quality/speed tradeoff on Groq's free tier for a
# classification task like this. Falls back to a smaller/faster model via
# GROQ_MODEL in secrets if you hit rate limits or want lower latency.
DEFAULT_MODEL = "llama-3.3-70b-versatile"


@st.cache_resource(show_spinner=False)
def get_client():
    """
    Returns an initialised OpenAI-SDK client pointed at Groq's endpoint,
    or None if the API key is absent.
    """
    try:
        from openai import OpenAI

        api_key: str = st.secrets["GROQ_API_KEY"]
        if not api_key:
            return None

        return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    except KeyError:
        logger.info("Groq secrets not found; news sentiment tagging disabled.")
        return None
    except Exception as exc:
        logger.warning("Groq client init failed: %s", exc)
        return None


def get_model() -> str:
    """Model override via secrets, falling back to DEFAULT_MODEL."""
    try:
        return st.secrets.get("GROQ_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
    except Exception:
        return DEFAULT_MODEL


def _is_available() -> bool:
    return get_client() is not None
