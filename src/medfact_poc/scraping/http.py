"""Shared HTTP client and config for evidence sources.

Centralizes rate limiting and TLS handling. Some environments such as corporate proxies
and sandboxes terminate TLS with a self-signed cert that breaks default verification.
Set ``MEDFACT_INSECURE_TLS=1`` or ``MEDFACT_CA_BUNDLE=/path/to/ca.pem`` to cope without
editing code. NCBI asks callers to identify themselves with ``NCBI_EMAIL``, and an
``NCBI_API_KEY`` raises the rate limit from 3 to 10 requests per second.
"""

from __future__ import annotations

import os
import threading
import time

import httpx

NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "")
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
# 3 req/s without a key, 10 with one (NCBI policy). Stay just under.
_MIN_INTERVAL = 0.34 if not NCBI_API_KEY else 0.11


def _verify() -> bool | str:
    if os.environ.get("MEDFACT_INSECURE_TLS") == "1":
        return False
    bundle = os.environ.get("MEDFACT_CA_BUNDLE")
    return bundle if bundle else True


class RateLimiter:
    """Process-wide minimum interval between requests, thread-safe."""

    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delta = now - self._last
            if delta < self._min_interval:
                time.sleep(self._min_interval - delta)
            self._last = time.monotonic()


_ncbi_limiter = RateLimiter(_MIN_INTERVAL)
_generic_limiter = RateLimiter(0.2)


def make_client(timeout: float = 30.0) -> httpx.Client:
    return httpx.Client(
        timeout=timeout,
        verify=_verify(),
        headers={"User-Agent": f"medfact-poc-harness ({NCBI_EMAIL or 'anon'})"},
        follow_redirects=True,
    )


def ncbi_params(extra: dict[str, str]) -> dict[str, str]:
    """Augment E-utilities params with identification and key when available."""
    params = dict(extra)
    if NCBI_EMAIL:
        params["email"] = NCBI_EMAIL
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    return params


def ncbi_throttle() -> None:
    _ncbi_limiter.wait()


def generic_throttle() -> None:
    _generic_limiter.wait()
