"""Sparse retrieval and rank fusion for hybrid product search."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

_TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "") if len(token) > 1]


class BM25Retriever:
    """Small dependency-free BM25 index aligned with product metadata rows.

    This implementation is intentionally simple and suitable for portfolio/local
    serving. Large distributed deployments would normally use a dedicated sparse
    engine such as OpenSearch or Elasticsearch.
    """

    def __init__(self, documents, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.lengths = []
        self.postings = defaultdict(list)
        document_frequency = Counter()

        for doc_id, text in enumerate(documents):
            terms = tokenize(text)
            self.lengths.append(len(terms))
            counts = Counter(terms)
            for term, frequency in counts.items():
                self.postings[term].append((doc_id, frequency))
                document_frequency[term] += 1

        self.document_count = len(self.lengths)
        self.avg_length = sum(self.lengths) / max(1, self.document_count)
        self.idf = {
            term: math.log(1.0 + (self.document_count - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 100):
        scores = defaultdict(float)
        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, frequency in self.postings[term]:
                length = self.lengths[doc_id]
                denominator = frequency + self.k1 * (1.0 - self.b + self.b * length / max(self.avg_length, 1e-9))
                scores[doc_id] += idf * (frequency * (self.k1 + 1.0)) / denominator
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]


def reciprocal_rank_fusion(rankings, k: int = 60):
    """Fuse independent ranked lists without requiring score calibration."""
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[int(doc_id)] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)
