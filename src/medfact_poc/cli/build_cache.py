"""Pre-fetch candidate evidence for the gold set into DuckDB.

Running this first lets ``medfact-run`` operate offline (``--use-cache``) and keeps
expensive API calls out of the measurement loop.
"""

from __future__ import annotations

import argparse

from ..orchestration.harness import GOLD_PATH, embed_pool, load_gold, retrieve_pool
from ..scraping.http import make_client
from ..scraping.sources import get_sources
from ..store import DEFAULT_DB, Store
from ..transformation import semantic


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch + embed candidate evidence for the gold set.")
    ap.add_argument("--gold", default=str(GOLD_PATH))
    ap.add_argument("--db", default=str(DEFAULT_DB))
    args = ap.parse_args()

    gold_set = load_gold(args.gold)
    embedder = semantic.get_embedder()
    sources = get_sources()
    print(f"Building cache for {len(gold_set.entries)} claims (embedder={embedder.name}) ...")
    with Store(args.db) as store, make_client() as client:
        for g in gold_set.entries:
            pool = retrieve_pool(g, store=store, client=client, use_cache=False, sources=sources)
            embed_pool(g, pool, store=store, embedder=embedder)
            ak = sum(1 for k in g.answer_key if store.has_candidate(k))
            print(f"  {g.id:30s} pool={len(pool):3d}  answer-key cached={ak}/{len(g.answer_key)}")
    print(f"Cache written to {args.db}")


if __name__ == "__main__":
    main()
