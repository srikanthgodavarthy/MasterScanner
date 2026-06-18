"""
OpenAI client helper for the MasterScanner Agent tab.

Mirrors utils/supabase_client.py's pattern: pull credentials from st.secrets
(works identically locally via .streamlit/secrets.toml and on Streamlit
Community Cloud's secrets UI), cache the client, and fail soft (return None)
if not configured so the rest of the app keeps working without it.

Usage
-----
from utils.openai_client import get_client, _is_available, DEFAULT_MODEL
"""

from __future__ import annotations

import logging

import streamlit as st

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"


@st.cache_resource(show_spinner=False)
def get_client():
    """
    Returns an initialised OpenAI client, or None if the API key is absent.
    """
    try:
        from openai import OpenAI

        api_key: str = st.secrets["OPENAI_API_KEY"]
        if not api_key:
            return None

        return OpenAI(api_key=api_key)

    except KeyError:
        logger.info("OpenAI secrets not found; agent tab disabled.")
        return None
    except Exception as exc:
        logger.warning("OpenAI client init failed: %s", exc)
        return None


def get_model() -> str:
    """Model override via secrets, falling back to DEFAULT_MODEL."""
    try:
        return st.secrets.get("OPENAI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
    except Exception:
        return DEFAULT_MODEL


def _is_available() -> bool:
    return get_client() is not None
