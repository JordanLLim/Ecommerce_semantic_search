"""Consumer-facing storefront for the semantic product search API."""

from __future__ import annotations

import os
from time import perf_counter

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Findly Shop", page_icon="🛍️", layout="wide")

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {max-width: 1180px; padding-top: 1.2rem; padding-bottom: 3rem;}
[data-testid="stSidebar"] {background: #fff8f4; border-right: 1px solid #f1e4dc;}
.storebar {display:flex; align-items:center; gap:18px; margin-bottom:8px;}
.brandmark {font-size:30px; font-weight:800; letter-spacing:-1px; color:#ee4d2d;}
.tagline {font-size:14px; color:#777;}
.hero {padding:28px 30px; border-radius:18px; background:linear-gradient(120deg,#fff3ed,#fffaf7); margin:12px 0 22px; border:1px solid #f4e3da;}
.hero h1 {font-size:34px; margin:0 0 7px; letter-spacing:-1px;}
.hero p {margin:0; color:#666;}
.result-card {border:1px solid #ececec; border-radius:14px; padding:17px 18px; margin:10px 0; background:white; box-shadow:0 2px 10px rgba(0,0,0,.035);}
.result-card:hover {border-color:#ee4d2d; box-shadow:0 5px 16px rgba(0,0,0,.07);}
.product-title {font-size:16px; font-weight:650; line-height:1.45; margin-bottom:9px;}
.meta {font-size:12px; color:#888;}
.badge {display:inline-block; background:#fff1ec; color:#d94325; border-radius:999px; padding:3px 8px; font-size:11px; margin-right:6px;}
.tech {margin-top:26px; padding-top:14px; border-top:1px solid #eee; color:#888; font-size:12px;}
div.stButton > button[kind="primary"] {background:#ee4d2d; border-color:#ee4d2d;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="storebar"><div class="brandmark">Findly</div><div class="tagline">Smarter product discovery across 1.3M items</div></div>', unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>Find what you actually mean.</h1><p>Search naturally. Find relevant products without matching every exact keyword.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Refine results")
    locale = st.selectbox("Marketplace", ["All", "US", "JP", "ES"])
    brand = st.text_input("Brand", placeholder="Any brand")
    st.markdown("---")
    st.markdown("### Search mode")
    hybrid = st.toggle("Smart keyword boost", value=True, help="Combines semantic similarity with lexical overlap.")
    rerank = st.toggle("Deep reranking", value=False, help="More compute-intensive second-stage ranking.")
    top_k = st.slider("Results", 5, 20, 10)
    candidate_k = st.slider("Candidate pool", 20, 100, 50, 10)
    st.caption("Deep reranking is optional and may be slower on CPU.")

query_col, button_col = st.columns([6, 1])
with query_col:
    query = st.text_input("Search", placeholder="Try: wireless gaming keyboard", label_visibility="collapsed")
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
            with st.spinner("Finding the best matches..."):
                started = perf_counter()
                response = requests.post(f"{API_URL}/search", json=request_payload, timeout=60)
                round_trip_ms = (perf_counter() - started) * 1000
                response.raise_for_status()
                payload = response.json()

            results = payload.get("results", [])
            st.markdown(f"### Results for ‘{cleaned_query}’")
            st.caption(f"{len(results)} products · {round_trip_ms:.0f} ms end-to-end")

            if not results:
                st.info("No matching products found. Try removing a filter or using broader words.")
            else:
                for item in results:
                    title = str(item.get("title", "Untitled product"))
                    product_id = str(item.get("product_id", "-"))
                    item_brand = item.get("brand")
                    item_locale = item.get("locale")
                    score = item.get("combined_score") if hybrid else item.get("score")
                    badges = ""
                    if item_brand:
                        badges += f'<span class="badge">{item_brand}</span>'
                    if item_locale:
                        badges += f'<span class="badge">{str(item_locale).upper()}</span>'
                    score_text = f" · Match {float(score) * 100:.0f}%" if isinstance(score, (int, float)) else ""
                    st.markdown(
                        f'<div class="result-card"><div class="product-title">{title}</div>{badges}'
                        f'<div class="meta">Product {product_id}{score_text}</div></div>',
                        unsafe_allow_html=True,
                    )

            with st.expander("Search performance"):
                a, b, c = st.columns(3)
                a.metric("Retrieval", f"{payload.get('retrieval_latency_ms', 0):.1f} ms")
                b.metric("Reranking", f"{payload.get('rerank_latency_ms', 0):.1f} ms")
                c.metric("Backend", f"{payload.get('latency_ms', 0):.1f} ms")
                st.caption(f"Mode: {'semantic + lexical' if payload.get('hybrid') else 'semantic'} · reranked={payload.get('reranked', False)} · cache={payload.get('cache_hit', False)}")
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

st.markdown('<div class="tech">Portfolio demo · Sentence Transformers · FAISS IVF · FastAPI · 1.3M-product catalog</div>', unsafe_allow_html=True)
