"""Streamlit client for the semantic product search API."""

from __future__ import annotations

import os
from time import perf_counter

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Amazon Semantic Product Search",
    page_icon="🔎",
    layout="wide",
)

st.title("Amazon Semantic Product Search")
st.caption(
    "Semantic retrieval with Sentence Transformers, FastAPI and FAISS. "
    "The Streamlit app is only the client; retrieval is served by the API."
)

with st.sidebar:
    st.subheader("Search settings")
    top_k = st.slider("Top K", min_value=1, max_value=20, value=10)
    st.caption(f"API: {API_URL}")

    if st.button("Check API health", use_container_width=True):
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") == "ok":
                st.success("API is healthy")
            else:
                st.warning("API responded, but status was unexpected")
            st.json(payload)
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")

query = st.text_input(
    "Search products",
    placeholder="e.g. wireless gaming mouse",
)
search_clicked = st.button("Search", type="primary")

if search_clicked:
    cleaned_query = query.strip()
    if not cleaned_query:
        st.warning("Enter a search query first.")
    else:
        try:
            started = perf_counter()
            response = requests.post(
                f"{API_URL}/search",
                json={"query": cleaned_query, "top_k": top_k},
                timeout=30,
            )
            round_trip_ms = (perf_counter() - started) * 1000
            response.raise_for_status()
            payload = response.json()

            api_latency_ms = payload.get("latency_ms")
            results = payload.get("results", [])

            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Results", len(results))
            metric_b.metric(
                "Backend search",
                f"{api_latency_ms:.1f} ms" if isinstance(api_latency_ms, (int, float)) else "n/a",
            )
            metric_c.metric("HTTP round trip", f"{round_trip_ms:.1f} ms")

            st.subheader("Results")
            if not results:
                st.info("No products were returned.")
            else:
                for item in results:
                    rank = item.get("rank", "-")
                    title = item.get("title", "Untitled product")
                    product_id = item.get("product_id", "-")
                    score = item.get("score")

                    with st.container(border=True):
                        st.markdown(f"**{rank}. {title}**")
                        left, right = st.columns([3, 1])
                        left.caption(f"Product ID: {product_id}")
                        if isinstance(score, (int, float)):
                            right.caption(f"Similarity: {score:.4f}")
        except requests.Timeout:
            st.error("The search request timed out.")
        except requests.RequestException as exc:
            st.error(f"Search API request failed: {exc}")
        except ValueError:
            st.error("The API returned an invalid response.")

st.divider()
st.caption(
    "Architecture: Streamlit UI → HTTP POST /search → FastAPI → embedding model → FAISS index"
)
