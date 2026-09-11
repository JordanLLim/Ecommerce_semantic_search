"""Measure local HTTP transport overhead without running search inference.

Useful on Windows when urllib may inherit system proxy/PAC settings. The script compares
localhost vs 127.0.0.1 with proxy lookup disabled so transport overhead is not confused
with backend search latency.
"""
from __future__ import annotations
import json, statistics, time
from urllib.request import ProxyHandler, build_opener


def measure(url: str, repeats: int = 10) -> None:
    opener = build_opener(ProxyHandler({}))
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        with opener.open(url, timeout=10) as response:
            json.load(response)
        times.append((time.perf_counter() - started) * 1000)
    print(f"{url}: mean={statistics.mean(times):.1f} ms, min={min(times):.1f} ms, max={max(times):.1f} ms")


if __name__ == "__main__":
    measure("http://localhost:8000/health")
    measure("http://127.0.0.1:8000/health")
