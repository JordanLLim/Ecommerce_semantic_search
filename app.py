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
    "Streamlit is only the client; retrieval, filtering and reranking stay in the API."
)

with st.sidebar:
    st.subheader("Search settings")
    top_k = st.slider("Top K", min_value=1, max_value=20, value=10)
    candidate_k = st.slider("Candidate K", min_value=20, max_value=200, value=50, step=10)
    hybrid = st.toggle("Dense + lexical fusion", value=False)
    rerank = st.toggle("Cross-encoder reranking", value=False)
    brand = st.text_input("Brand filter", placeholder="optional")
    locale = st.text_input("Locale filter", placeholder="e.g. us")
    user_keywords_text = st.text_input(
        "Preference keywords",
        placeholder="e.g. wireless lightweight",
        help="Small post-retrieval preference boost for the demo.",
    )
    experiment_key = st.text_input(
        "A/B experiment key",
        placeholder="optional stable user/session id",
        help="When provided, the backend deterministically assigns dense or enhanced search.",
    )
    st.caption(f"API: {API_URL}")

    health_col, metrics_col = st.columns(2)
    if health_col.button("Health", use_container_width=True):
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            response.raise_for_status()
            st.json(response.json())
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")

    if metrics_col.button("Metrics", use_container_width=True):
        try:
            response = requests.get(f"{API_URL}/metrics", timeout=5)
            response.raise_for_status()
            st.json(response.json())
        except requests.RequestException as exc:
            st.error(f"Could not load metrics: {exc}")

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
        user_keywords = [part for part in user_keywords_text.strip().split() if part]
        request_payload = {
            "query": cleaned_query,
            "top_k": top_k,
            "candidate_k": max(candidate_k, top_k),
            "hybrid": hybrid,
            "rerank": rerank,
            "brand": brand.strip() or None,
            "locale": locale.strip() or None,
            "user_keywords": user_keywords,
            "experiment_key": experiment_key.strip() or None,
        }

        try:
            started = perf_counter()
            response = requests.post(
                f"{API_URL}/search",
                json=request_payload,
                timeout=60,
            )
            round_trip_ms = (perf_counter() - started) * 1000
            response.raise_for_status()
            payload = response.json()

            results = payload.get("results", [])
            metric_a, metric_b, metric_c, metric_d = st.columns(4)
            metric_a.metric("Results", len(results))
            metric_b.metric("Backend", f"{payload.get('latency_ms', 0):.1f} ms")
            metric_c.metric("Retrieval", f"{payload.get('retrieval_latency_ms', 0):.1f} ms")
            metric_d.metric("HTTP round trip", f"{round_trip_ms:.1f} ms")

            st.caption(
                f"Variant: {payload.get('variant', 'dense')} · "
                f"hybrid={payload.get('hybrid', False)} · "
                f"reranked={payload.get('reranked', False)} · "
                f"cache_hit={payload.get('cache_hit', False)} · "
                f"rerank={payload.get('rerank_latency_ms', 0):.1f} ms"
            )

            st.subheader("Results")
            if not results:
                st.info("No products were returned. Filters may be too restrictive.")
            else:
                for item in results:
                    rank = item.get("rank", "-")
                    title = item.get("title", "Untitled product")
                    product_id = item.get("product_id", "-")
                    score = item.get("score")
                    reranker_score = item.get("reranker_score")

                    with st.container(border=True):
                        st.markdown(f"**{rank}. {title}**")
                        left, middle, right = st.columns([3, 1, 1])
                        left.caption(f"Product ID: {product_id}")
                        if isinstance(score, (int, float)):
                            middle.caption(f"Dense: {score:.4f}")
                        if isinstance(reranker_score, (int, float)):
                            right.caption(f"Reranker: {reranker_score:.4f}")
                        elif isinstance(item.get("combined_score"), (int, float)):
                            right.caption(f"Combined: {item['combined_score']:.4f}")
        except requests.Timeout:
            st.error("The search request timed out.")
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = response.json().get("detail", "")
            except ValueError:
                pass
            st.error(f"Search API request failed: {detail or exc}")
        except requests.RequestException as exc:
            st.error(f"Search API request failed: {exc}")
        except ValueError:
            st.error("The API returned an invalid response.")

st.divider()
st.caption(
    "Architecture: Streamlit → HTTP API → query embedding → FAISS IVF → optional filtering/fusion/reranking → JSON response"
)
