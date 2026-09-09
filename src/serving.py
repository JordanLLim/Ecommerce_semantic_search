"""Small serving utilities used by the FastAPI search service."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any


class LRUCache:
    """Thread-safe in-process LRU cache for repeated search requests."""

    def __init__(self, max_size: int = 256):
        self.max_size = max(1, int(max_size))
        self._items: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            if key not in self._items:
                return None
            value = self._items.pop(key)
            self._items[key] = value
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._items:
                self._items.pop(key)
            self._items[key] = value
            while len(self._items) > self.max_size:
                self._items.popitem(last=False)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._items)


@dataclass
class SearchMetrics:
    requests: int = 0
    errors: int = 0
    cache_hits: int = 0
    total_latency_ms: float = 0.0
    variant_counts: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, latency_ms: float, variant: str, cache_hit: bool = False) -> None:
        with self._lock:
            self.requests += 1
            self.total_latency_ms += float(latency_ms)
            self.variant_counts[variant] = self.variant_counts.get(variant, 0) + 1
            if cache_hit:
                self.cache_hits += 1

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def snapshot(self, cache_size: int) -> dict[str, Any]:
        with self._lock:
            mean = self.total_latency_ms / self.requests if self.requests else 0.0
            return {
                "requests": self.requests,
                "errors": self.errors,
                "cache_hits": self.cache_hits,
                "cache_size": cache_size,
                "mean_backend_latency_ms": round(mean, 3),
                "variant_counts": dict(self.variant_counts),
            }


def cache_key(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assign_variant(experiment_key: str | None) -> str:
    """Deterministically assign dense or enhanced search for simple A/B testing."""
    if not experiment_key:
        return "dense"
    bucket = int(hashlib.sha256(experiment_key.encode("utf-8")).hexdigest()[:8], 16) % 2
    return "dense" if bucket == 0 else "enhanced"


def log_query(path: str, record: dict[str, Any]) -> None:
    """Append non-blocking-scale JSONL telemetry suitable for a local demo."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    enriched = {"timestamp": time(), **record}
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(enriched, ensure_ascii=False) + "\n")
