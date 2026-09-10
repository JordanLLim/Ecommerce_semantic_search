"""Consumer-facing storefront for the semantic product search API."""

from __future__ import annotations

import html
import os
from time import perf_counter

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Findly Shop",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: rgba(255,255,255,.96);}
.block-container {max-width: 1220px; padding-top: 1.5rem; padding-bottom: 3rem;}
[data-testid="stSidebar"] {background: #fffaf7; border-right: 1px solid #f0e7e2;}
.storebar {display:flex; align-items:center; justify-content:space-between; gap:18px; margin-bottom:10px;}
.brand-wrap {display:flex; align-items:center; gap:18px;}
.brandmark {font-size:31px; font-weight:800; letter-spacing:-1px; color:#ee4d2d;}
.tagline {font-size:14px; color:#777;}
.nav {font-size:13px; color:#666; word-spacing:12px;}
.hero {padding:30px 32px; border-radius:20px; background:linear-gradient(120deg,#fff2ea,#fffaf7); margin:12px 0 22px; border:1px solid #f2dfd5;}
.hero h1 {font-size:35px; margin:0 0 7px; letter-spacing:-1.2px; color:#252537;}
.hero p {margin:0; color:#666; font-size:15px;}
.results-summary {display:flex; justify-content:space-between; align-items:end; margin:24px 0 10px;}
.results-title {font-size:25px; font-weight:750; color:#2c2c3a;}
.results-meta {font-size:13px; color:#8b8b96; margin-top:3px;}
.product-card {height:100%; min-height:185px; border:1px solid #ece9e7; border-radius:15px; padding:18px; background:white; box-shadow:0 2px 8px rgba(0,0,0,.025); margin-bottom:14px;}
.product-card:hover {border-color:#f0a18e; box-shadow:0 5px 18px rgba(0,0,0,.06); transform:translateY(-1px); transition:.15s ease;}
.product-thumb {height:72px; border-radius:11px; background:linear-gradient(135deg,#faf6f3,#f3efec); display:flex; align-items:center; justify-content:center; font-size:28px; margin-bottom:14px;}
.product-title {font-size:15px; font-weight:650; line-height:1.42; color:#30303e; min-height:43px; margin-bottom:11px;}
.meta {font-size:12px; color:#91919a; margin-top:8px;}
.badge {display:inline-block; background:#fff0ea; color:#d84b2b; border-radius:999px; padding:3px 8px; font-size:11px; margin-right:6px;}
.tech {margin-top:28px; padding-top:14px; border-top:1px solid #eee; color:#909099; font-size:12px;}
div.stButton > button[kind="primary"] {background:#ee4d2d; border-color:#ee4d2d; border-radius:9px; height:42px;}
[data-testid="stTextInput"] input {border-radius:9px;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="storebar"><div class="brand-wrap"><div class="brandmark">Findly</div>'
    '<div class="tagline">Smarter product discovery across 1.3M items</div></div>'
    '<div class="nav">Electronics Home Gaming Accessories</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero"><h1>Find products by meaning, not just keywords.</h1>'
    '<p>Search naturally across a 1.3M-product marketplace catalog.</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Filters")
    locale = st.selectbox("Marketplace", ["All", "US", "JP", "ES"])
    brand = st.text_input("Brand", placeholder="Any brand")
    st.markdown("---")
    st.markdown("### Ranking")
    hybrid = st.toggle(
        "Semantic + keyword fusion",
        value=True,
        help="Fuses dense semantic similarity with lexical overlap inside the ANN candidate set.",
    )
    rerank = st.toggle(
        "Deep reranking",
        value=False,
        help="Optional CrossEncoder second-stage ranking. Slower on CPU.",
    )
    top_k = st.slider("Results", 5, 20, 10)
    candidate_k = st.slider("Candidate pool", 20, 100, 50, 10)
    st.caption("Tip: keep deep reranking off for the fastest storefront experience.")

query_col, button_col = st.columns([6, 1])
with query_col:
    query = st.text_input(
        "Search",
        placeholder="Search for products, brands or needs — e.g. quiet mechanical keyboard",
        label_visibility="collapsed",
    )
with button_col:
    search_clicked = st.button("Search", type="primary", use_container_width=True)

if search_clicked:
    cleaned_query = query.strip()
    if not cleaned_query:
        st.warning("Type something to search for.")
    else:
        request_payload = {
            "query": cleaned_query,
            "top_k": top_k,
            "candidate_k": max(candidate_k, top_k),
            "hybrid": hybrid,
            "rerank": rerank,
            "brand": brand.strip() or None,
            "locale": None if locale == "All" else locale.lower(),
            "user_keywords": [],
            "experiment_key": None,
        }

        try:
            with st.spinner("Searching marketplace..."):
                started = perf_counter()
                response = requests.post(f"{API_URL}/search", json=request_payload, timeout=60)
                round_trip_ms = (perf_counter() - started) * 1000
                response.raise_for_status()
                payload = response.json()

            results = payload.get("results", [])
            safe_query = html.escape(cleaned_query)
            st.markdown(
                f'<div class="results-summary"><div><div class="results-title">Results for “{safe_query}”</div>'
                f'<div class="results-meta">{len(results)} products returned · {round_trip_ms:.0f} ms end-to-end</div></div></div>',
                unsafe_allow_html=True,
            )

            if not results:
                st.info("No matching products found. Try removing a filter or using broader words.")
            else:
                for row_start in range(0, len(results), 2):
                    cols = st.columns(2)
                    for offset, col in enumerate(cols):
                        idx = row_start + offset
                        if idx >= len(results):
                            continue
                        item = results[idx]
                        title = html.escape(str(item.get("title", "Untitled product")))
                        product_id = html.escape(str(item.get("product_id", "-")))
                        item_brand = item.get("brand")
                        item_locale = item.get("locale")
                        badges = ""
                        if item_brand:
                            badges += f'<span class="badge">{html.escape(str(item_brand))}</span>'
                        if item_locale:
                            badges += f'<span class="badge">{html.escape(str(item_locale).upper())}</span>'
                        with col:
                            st.markdown(
                                '<div class="product-card">'
                                '<div class="product-thumb">🛍️</div>'
                                f'<div class="product-title">{title}</div>'
                                f'{badges}'
                                f'<div class="meta">Product ID · {product_id}</div>'
                                '</div>',
                                unsafe_allow_html=True,
                            )

            with st.expander("Search performance"):
                a, b, c, d = st.columns(4)
                a.metric("Retrieval", f"{payload.get('retrieval_latency_ms', 0):.1f} ms")
                b.metric("Reranking", f"{payload.get('rerank_latency_ms', 0):.1f} ms")
                c.metric("Backend", f"{payload.get('latency_ms', 0):.1f} ms")
                d.metric("HTTP", f"{round_trip_ms:.1f} ms")
                st.caption(
                    f"Mode: {'semantic + lexical fusion' if payload.get('hybrid') else 'semantic only'} · "
                    f"reranked={payload.get('reranked', False)} · cache={payload.get('cache_hit', False)}"
                )
        except requests.Timeout:
            st.error("Search took too long. Try again without deep reranking.")
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = response.json().get("detail", "")
            except ValueError:
                pass
            st.error(f"Search failed: {detail or exc}")
        except requests.RequestException as exc:
            st.error(f"Search service unavailable: {exc}")

st.markdown(
    '<div class="tech">Portfolio demo · Sentence Transformers · FAISS IVF · FastAPI · 1.3M-product catalog</div>',
    unsafe_allow_html=True,
)
