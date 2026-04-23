"""Simple timing logger for route_opt pipeline stages."""

import time
from contextlib import contextmanager
from typing import Any

_TIMINGS: list[tuple[str, float]] = []


@contextmanager
def stage(name: str):
    """Log time spent in a named stage."""
    t0 = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - t0
        _TIMINGS.append((name, elapsed))
        print(f"  [PROF] {name}: {elapsed:.2f}s")


def reset():
    _TIMINGS.clear()


def summary():
    total = sum(t for _, t in _TIMINGS)
    print(f"\n=== Pipeline Summary (total {total:.2f}s) ===")
    for name, elapsed in _TIMINGS:
        pct = elapsed / total * 100 if total else 0
        print(f"  {name:40s} {elapsed:8.2f}s  ({pct:5.1f}%)")
    print("=" * 55)
    reset()
