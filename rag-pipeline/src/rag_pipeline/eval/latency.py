"""

Lightweight latency instrumentation.

A process-wide registry of timing measurements organized by tag. No external
deps (no OpenTelemetry yet). Designed for ad-hoc benchmark cells and Phase 5
load-testing — when we move to a long-running service in Phase 6 we'll swap
this for OTel without changing the user-facing API.

Usage:
    from rag_pipeline.eval import LatencyTracker

    # Context manager
    with LatencyTracker.timer("retrieve.hybrid"):
        results = retriever.retrieve(query)

    # Decorator
    @timed("answer.full_pipeline")
    def my_func(...): ...

    # Report
    print(LatencyTracker.report())
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from typing import Callable


class LatencyTracker:
    """Process-wide registry of timings, organized by tag.

    All times are stored in milliseconds. Thread-unsafe by design — Phase 5
    benchmarks run single-threaded; Phase 6 service will swap this for OTel.
    """

    _measurements: dict[str, list[float]] = defaultdict(list)

    @classmethod
    def record(cls, tag: str, duration_ms: float) -> None:
        cls._measurements[tag].append(duration_ms)

    @classmethod
    @contextmanager
    def timer(cls, tag: str):
        """Time a code block: `with LatencyTracker.timer('xyz'): ...`"""
        start = time.perf_counter()
        try:
            yield
        finally:
            cls.record(tag, (time.perf_counter() - start) * 1000.0)

    @classmethod
    def reset(cls, tag: str | None = None) -> None:
        """Clear all measurements, or just one tag."""
        if tag is None:
            cls._measurements.clear()
        else:
            cls._measurements.pop(tag, None)

    @classmethod
    def stats(cls, tag: str) -> dict | None:
        values = cls._measurements.get(tag, [])
        if not values:
            return None
        sorted_vals = sorted(values)
        return {
            "count":   len(values),
            "mean_ms": statistics.mean(values),
            "p50_ms":  _percentile(sorted_vals, 0.50),
            "p95_ms":  _percentile(sorted_vals, 0.95),
            "p99_ms":  _percentile(sorted_vals, 0.99),
            "min_ms":  min(values),
            "max_ms":  max(values),
        }

    @classmethod
    def report(cls) -> str:
        """Tabular report of all tracked tags."""
        if not cls._measurements:
            return "(no measurements recorded)"
        header = f"{'tag':<32s} {'n':>4s} {'mean':>9s} {'p50':>9s} {'p95':>9s} {'p99':>9s} {'min':>9s} {'max':>9s}"
        lines = [header, "-" * len(header)]
        for tag in sorted(cls._measurements.keys()):
            s = cls.stats(tag)
            lines.append(
                f"{tag:<32s} {s['count']:>4d} "
                f"{s['mean_ms']:>7.1f}ms {s['p50_ms']:>7.1f}ms "
                f"{s['p95_ms']:>7.1f}ms {s['p99_ms']:>7.1f}ms "
                f"{s['min_ms']:>7.1f}ms {s['max_ms']:>7.1f}ms"
            )
        return "\n".join(lines)


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation percentile."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] if lo == hi else sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def timed(tag: str) -> Callable:
    """Decorator: record the wrapped function's latency under `tag`."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with LatencyTracker.timer(tag):
                return func(*args, **kwargs)
        return wrapper
    return decorator