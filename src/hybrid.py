"""Sparse retrieval and rank fusion for hybrid product search."""

from __future__ import annotations

import heapq
import math
import re
from collections import Counter, defaultdict

_TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "") if len(token) > 1]


class BM25Retriever:
    """Small dependency-free BM25 index aligned with product metadata rows.

    The postings store precomputed per-document term contributions so the online
    path only accumulates weights for query terms. This preserves the same BM25
    scoring formula while avoiding repeated denominator math on every request.
    """

    def __init__(self, documents, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        lengths = []
        raw_postings = defaultdict(list)
        document_frequency = Counter()

        for doc_id, text in enumerate(documents):
            terms = tokenize(text)
            lengths.append(len(terms))
            counts = Counter(terms)
            for term, frequency in counts.items():
                raw_postings[term].append((doc_id, frequency))
                document_frequency[term] += 1

        self.document_count = len(lengths)
        self.avg_length = sum(lengths) / max(1, self.document_count)
        self.idf = {
            term: math.log(1.0 + (self.document_count - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

        self.postings = {}
        avg_length = max(self.avg_length, 1e-9)
        for term, entries in raw_postings.items():
            idf = self.idf[term]
            weighted = []
            for doc_id, frequency in entries:
                length = lengths[doc_id]
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / avg_length
                )
                weight = idf * (frequency * (self.k1 + 1.0)) / denominator
                weighted.append((doc_id, weight))
            self.postings[term] = weighted

    def search(self, query: str, top_k: int = 100):
        if top_k <= 0:
            return []

        scores = defaultdict(float)
        for term in tokenize(query):
            for doc_id, weight in self.postings.get(term, ()):
                scores[doc_id] += weight

        if len(scores) <= top_k:
            return sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return heapq.nlargest(top_k, scores.items(), key=lambda item: item[1])


def reciprocal_rank_fusion(rankings, k: int = 60):
    """Fuse independent ranked lists without requiring score calibration."""
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[int(doc_id)] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)
