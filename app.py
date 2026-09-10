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
#MainMenu, footer {visibility:hidden;}
[data-testid="stHeader"] {background:rgba(255,255,255,.96);}
.block-container {max-width:1240px; padding-top:1rem; padding-bottom:3rem;}
[data-testid="stSidebar"] {background:#fffaf7; border-right:1px solid #f1e5de;}

.storebar {display:flex;align-items:center;justify-content:space-between;padding:8px 0 12px;}
.brand-row {display:flex;align-items:center;gap:14px;}
.brandmark {font-size:31px;font-weight:850;letter-spacing:-1.2px;color:#ee4d2d;}
.brandcopy {font-size:13px;color:#777;}
.navline {font-size:13px;color:#666;word-spacing:12px;}
.hero {padding:24px 28px;border:1px solid #f2e2da;border-radius:18px;background:linear-gradient(120deg,#fff4ee,#fffdfb);margin-bottom:18px;}
.hero-kicker {font-size:12px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#d94a2d;margin-bottom:5px;}
.hero h1 {font-size:31px;line-height:1.15;margin:0 0 7px;letter-spacing:-.8px;color:#292734;}
.hero p {margin:0;color:#706b72;font-size:14px;}
.search-meta {font-size:13px;color:#8a858d;margin:10px 0 13px;}
.section-title {font-size:24px;font-weight:780;color:#302d3a;margin:8px 0 2px;}
.product-card {height:100%;min-height:245px;border:1px solid #ece8e5;border-radius:15px;background:#fff;overflow:hidden;box-shadow:0 2px 11px rgba(33,28,26,.045);margin-bottom:16px;}
.product-card:hover {border-color:#ef9a86;box-shadow:0 6px 18px rgba(33,28,26,.075);transform:translateY(-1px);}
.product-visual {height:118px;background:linear-gradient(135deg,#f8f5f3,#fff8f4);display:flex;align-items:center;justify-content:center;color:#d9b3a8;font-size:42px;border-bottom:1px solid #f2eeeb;}
.product-body {padding:14px 15px 15px;}
.product-title {font-size:14px;font-weight:680;line-height:1.42;color:#2f2c36;min-height:40px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;}
.product-brand {font-size:12px;color:#8a858d;margin-top:8px;}
.product-foot {display:flex;justify-content:space-between;align-items:center;margin-top:13px;font-size:11px;color:#aaa4aa;}
.market-badge {padding:3px 8px;border-radius:999px;background:#fff0ea;color:#d8492d;font-weight:650;}
.status-pill {display:inline-block;padding:4px 9px;border-radius:999px;background:#f4f8f4;color:#4b7653;font-size:11px;font-weight:650;}
.small-note {font-size:12px;color:#918b92;}
div.stButton > button[kind="primary"] {background:#ee4d2d;border-color:#ee4d2d;border-radius:10px;font-weight:700;}
div.stButton > button[kind="primary"]:hover {background:#d94425;border-color:#d94425;}
[data-testid="stTextInput"] input {border-radius:10px;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="storebar">
  <div class="brand-row">
    <div class="brandmark">Findly</div>
    <div class="brandcopy">Semantic marketplace search across 1.3M products</div>
  </div>
  <div class="navline">Electronics &nbsp; Home &nbsp; Gaming &nbsp; Accessories</div>
</div>
<div class="hero">
  <div class="hero-kicker">Product discovery</div>
  <h1>Search by what you need, not just exact keywords.</h1>
  <p>Find semantically related products across the Amazon ESCI catalog with fast approximate retrieval.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Refine results")
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
        help="Runs an optional CrossEncoder on a smaller candidate subset. Slower on CPU.",
    )
    top_k = st.slider("Results shown", 5, 20, 10)
    candidate_k = st.slider("Retrieval pool", 20, 100, 50, 10)
    st.markdown("---")
    st.caption("Tip: keep deep reranking off for the fastest storefront experience.")
    st.caption(f"API · {API_URL}")

query_col, button_col = st.columns([7, 1.25])
with query_col:
    query = st.text_input(
        "Search",
        placeholder="Search products, brands or needs — e.g. quiet mechanical keyboard",
        label_visibility="collapsed",
    )
with button_col:
    search_clicked = st.button("Search", type="primary", use_container_width=True)

if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_payload" not in st.session_state:
    st.session_state.last_payload = None
if "last_round_trip" not in st.session_state:
    st.session_state.last_round_trip = None

if search_clicked:
    cleaned_query = query.strip()
    if not cleaned_query:
        st.warning("Enter a search query first.")
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
            with st.spinner("Searching the catalog..."):
                started = perf_counter()
                response = requests.post(f"{API_URL}/search", json=request_payload, timeout=60)
                round_trip_ms = (perf_counter() - started) * 1000
                response.raise_for_status()
                payload = response.json()
            st.session_state.last_query = cleaned_query
            st.session_state.last_payload = payload
            st.session_state.last_round_trip = round_trip_ms
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

payload = st.session_state.last_payload
if payload:
    results = payload.get("results", [])
    cleaned_query = st.session_state.last_query
    round_trip_ms = st.session_state.last_round_trip or 0

    st.markdown(f'<div class="section-title">Results for “{html.escape(cleaned_query)}”</div>', unsafe_allow_html=True)
    mode_label = "Semantic + lexical" if payload.get("hybrid") else "Semantic"
    rerank_label = " + reranked" if payload.get("reranked") else ""
    st.markdown(
        f'<div class="search-meta">{len(results)} products · {round_trip_ms:.0f} ms end-to-end · {mode_label}{rerank_label}</div>',
        unsafe_allow_html=True,
    )

    if not results:
        st.info("No matching products found. Try removing a filter or using broader terms.")
    else:
        columns_per_row = 3
        for start in range(0, len(results), columns_per_row):
            cols = st.columns(columns_per_row)
            for offset, item in enumerate(results[start : start + columns_per_row]):
                title = html.escape(str(item.get("title", "Untitled product")))
                product_id = html.escape(str(item.get("product_id", "-")))
                raw_brand = item.get("brand")
                item_brand = html.escape(str(raw_brand)) if raw_brand else "Independent seller"
                raw_locale = item.get("locale")
                item_locale = html.escape(str(raw_locale).upper()) if raw_locale else "GLOBAL"
                with cols[offset]:
                    st.markdown(
                        f"""
<div class="product-card">
  <div class="product-visual">▣</div>
  <div class="product-body">
    <div class="product-title">{title}</div>
    <div class="product-brand">{item_brand}</div>
    <div class="product-foot">
      <span>#{product_id}</span>
      <span class="market-badge">{item_locale}</span>
    </div>
  </div>
</div>
""",
                        unsafe_allow_html=True,
                    )

    with st.expander("Search performance & engineering details"):
        a, b, c, d = st.columns(4)
        a.metric("Retrieval", f"{payload.get('retrieval_latency_ms', 0):.1f} ms")
        b.metric("Reranking", f"{payload.get('rerank_latency_ms', 0):.1f} ms")
        c.metric("Backend", f"{payload.get('latency_ms', 0):.1f} ms")
        d.metric("HTTP", f"{round_trip_ms:.1f} ms")
        cache_state = "hit" if payload.get("cache_hit") else "miss"
        st.caption(
            f"Variant: {payload.get('variant', 'dense')} · hybrid={payload.get('hybrid', False)} · "
            f"reranked={payload.get('reranked', False)} · cache={cache_state}"
        )
        st.markdown(
            '<span class="status-pill">1.3M catalog</span> '
            '<span class="status-pill">FAISS IVF</span> '
            '<span class="status-pill">FastAPI</span>',
            unsafe_allow_html=True,
        )

st.markdown("---")
st.markdown(
    '<div class="small-note">Findly is a portfolio search system built on the Amazon ESCI dataset. '
    'Product images and prices are intentionally not fabricated because they are not present in the serving artifacts.</div>',
    unsafe_allow_html=True,
)
