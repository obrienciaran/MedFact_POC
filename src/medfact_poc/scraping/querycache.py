"""Process-wide cache of retrieval results, backed by DuckDB.

Across a large corpus the same claim recurs constantly (many papers assert the same statin,
aspirin, or HRT effect), and those claims normalize to identical searches. Without a cache each
paper re-issues those searches and re-fetches the same studies, so retrieval time grows with the
corpus even though the distinct work is small. This cache stores two things in DuckDB and serves
them to every later paper, so each unique unit of work hits the network only once:

  * ``query_results`` maps a search query to its result, keyed by ``(source, query, page_size)``.
  * ``records`` maps a study id to its fetched record (title, abstract, publication types).

Both are checked first; the PubMed or Europe PMC API is called only on a miss, and the result is
then written back. It is opt-in: set ``MEDFACT_QUERY_CACHE`` to a file path (or to ``1``/``on``
for the default path) to enable it; leave it unset to fetch live every time, preserving the
existing behaviour of a no-cache run. Access is guarded by a lock because the filter and harness
fan papers out across threads, while the network call itself runs outside the lock.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import duckdb

DEFAULT_PATH = Path("data/cache/query_cache.duckdb")

_lock = threading.Lock()
_cache: QueryCache | None = None
_resolved = False


class QueryCache:
    """DuckDB-backed map from ``(source, query, page_size)`` to a JSON result payload."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(self.path))
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS query_results (
                source    VARCHAR,
                query     VARCHAR,
                page_size INTEGER,
                payload   VARCHAR,
                PRIMARY KEY (source, query, page_size)
            );
            CREATE TABLE IF NOT EXISTS records (
                ext_id  VARCHAR PRIMARY KEY,
                payload VARCHAR
            );
            """
        )
        self._db_lock = threading.Lock()

    def get(self, source: str, query: str, page_size: int) -> str | None:
        with self._db_lock:
            row = self._con.execute(
                "SELECT payload FROM query_results WHERE source = ? AND query = ? AND page_size = ?",
                [source, query, page_size],
            ).fetchone()
        return row[0] if row else None

    def put(self, source: str, query: str, page_size: int, payload: str) -> None:
        with self._db_lock:
            self._con.execute(
                "INSERT INTO query_results VALUES (?,?,?,?) "
                "ON CONFLICT (source, query, page_size) DO UPDATE SET payload = excluded.payload",
                [source, query, page_size, payload],
            )

    def get_records(self, ext_ids: list[str]) -> dict[str, str]:
        """Return the cached ``{ext_id: payload}`` for the ids that are present."""
        if not ext_ids:
            return {}
        placeholders = ",".join("?" * len(ext_ids))
        with self._db_lock:
            rows = self._con.execute(
                f"SELECT ext_id, payload FROM records WHERE ext_id IN ({placeholders})",
                list(ext_ids),
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def put_records(self, items: dict[str, str]) -> None:
        """Store ``{ext_id: payload}`` fetched records, overwriting any existing payload."""
        if not items:
            return
        with self._db_lock:
            for ext_id, payload in items.items():
                self._con.execute(
                    "INSERT INTO records VALUES (?,?) "
                    "ON CONFLICT (ext_id) DO UPDATE SET payload = excluded.payload",
                    [ext_id, payload],
                )


def get_query_cache() -> QueryCache | None:
    """Return the shared cache if ``MEDFACT_QUERY_CACHE`` enables it, else ``None``.

    Resolved once per process. If the cache file cannot be opened, caching is disabled rather
    than failing the run.
    """
    global _cache, _resolved
    if _resolved:
        return _cache
    with _lock:
        if _resolved:
            return _cache
        setting = os.environ.get("MEDFACT_QUERY_CACHE", "").strip()
        if setting and setting.lower() not in {"0", "off", "false", "no"}:
            path = DEFAULT_PATH if setting.lower() in {"1", "on", "true", "yes"} else setting
            try:
                _cache = QueryCache(path)
            except Exception as exc:  # noqa: BLE001 - a broken cache must not abort retrieval
                print(f"WARN: query cache disabled, could not open {path}. "
                    f"{type(exc).__name__}: {exc}")
                _cache = None
        _resolved = True
    return _cache
