"""Render the evidence graph from stored run data to an interactive HTML file.

Run ``medfact-run`` first so the store holds stance results, then:

    medfact-graph # writes reports/graph.html
"""

from __future__ import annotations

import argparse

from ..orchestration.harness import GOLD_PATH, load_gold
from ..reporting.graph import build_graph_data, render_html
from ..store import DEFAULT_DB, Store


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the evidence graph to interactive HTML.")
    ap.add_argument("--gold", default=str(GOLD_PATH))
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--out", default="reports/graph.html")
    ap.add_argument("--no-physics", action="store_true", help="disable force-directed layout")
    args = ap.parse_args()

    gold = load_gold(args.gold)
    with Store(args.db) as store:
        data = build_graph_data(gold, store)
    out = render_html(data, args.out, physics=not args.no_physics)
    print(f"Graph: {len(data.nodes)} nodes, {len(data.edges)} edges")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
