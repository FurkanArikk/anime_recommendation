"""Streamlit chat front end. Phase 1: placeholder that verifies it can reach the API."""

import httpx
import streamlit as st

from anime_rec.config import get_settings

st.set_page_config(page_title="Anime Recommender", page_icon="🎌", layout="wide")
st.title("Anime Recommender")

api_url = get_settings().api_url
try:
    health = httpx.get(f"{api_url}/health", timeout=5).json()
    st.success(f"API reachable at {api_url} — version {health['version']}")
except httpx.HTTPError as exc:
    st.error(f"API not reachable at {api_url}: {exc}")

st.info("The chat interface arrives in phase 7.")
