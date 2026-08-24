# Ecommerce_semantic_search

E-commerce Semantic Search & Ranking Engine

A work-in-progress machine learning project built on the Amazon Shopping Queries (ESCI) dataset to study semantic product search, ranking, and domain-specific model adaptation.

The current pipeline evaluates a frozen MiniLM text encoder on query-product ranking using NDCG@10, followed by error analysis to identify failure modes such as negation, numeric constraints, exact-product intent, and lexical-overlap traps.

Current baseline: 0.8415 mean NDCG@10 on a fixed 1,000-query evaluation subset.

Planned next stages include domain-specific fine-tuning, hard-negative mining, reranking, knowledge distillation, and large-scale vector retrieval using FAISS.
