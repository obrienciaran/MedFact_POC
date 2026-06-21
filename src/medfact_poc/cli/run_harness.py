"""Run the recall measurement and write a report.

    medfact-run # live APIs + configured backends
    medfact-run --use-cache # offline, from a prior medfact-build-cache
"""

from __future__ import annotations

import argparse

from ..orchestration.harness import GOLD_PATH, load_gold, run
from ..reporting.metrics import compute
from ..reporting.report import write_reports
from ..store import DEFAULT_DB
from ..transformation import semantic
from ..transformation.stance import get_stance_backend


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure negative-evidence recall on the gold set.")
    ap.add_argument("--gold", default=str(GOLD_PATH))
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--use-cache", action="store_true", help="read pools from cache (offline)")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    gold_set = load_gold(args.gold)
    reports, _labels = run(gold_set, db_path=args.db, use_cache=args.use_cache)
    metrics = compute(reports)

    md, csv_path = write_reports(
        gold_set, reports, metrics,
        embedder_name=semantic.get_embedder().name,
        stance_name=get_stance_backend().name,
        out_dir=args.out,
    )
    print(f"Retrieval recall: {metrics.retrieval_recall*100:.0f}%  "
        f"(stance overall {metrics.stance_recall_overall*100:.0f}%, "
        f"false-contradiction {metrics.false_contradiction_rate*100:.0f}%)")
    print(f"Report:  {md}")
    print(f"CSV:     {csv_path}")


if __name__ == "__main__":
    main()
