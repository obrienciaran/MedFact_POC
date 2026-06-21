"""Filter a lakehouse of PubMed XML for truthfulness.

Point it at a directory of PubMed/MEDLINE XML files and it writes a flat CSV (one row per
paper: identifier, verdict, score, action, metadata) plus an interactive HTML graph.

    medfact-filter --input data/pubmed_xml # offline stub backends
    MEDFACT_LLM_PROVIDER=gemini MEDFACT_EXTRACT_BACKEND=llm \
        MEDFACT_STANCE_BACKEND=llm MEDFACT_RETRIEVER=live \
        medfact-filter --input data/pubmed_xml # real backends

Backends default to deterministic offline stubs, so a run needs no API key. Select real
providers with MEDFACT_LLM_PROVIDER in {anthropic, openai, gemini} plus the matching key.
"""

from __future__ import annotations

import argparse
from collections import Counter

from ..orchestration.pipeline import run_filter
from ..reporting.flat_report import write_flat_csv
from ..reporting.graph import FILTER_HOW_TO_READ, build_paper_graph_data, render_html
from ..transformation.ingest import load_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter PubMed XML papers for truthfulness.")
    ap.add_argument("--input", required=True, help="directory of PubMed XML files (or one file)")
    ap.add_argument("--out-csv", default="reports/filter.csv")
    ap.add_argument("--out-html", default="reports/filter.html")
    ap.add_argument("--limit", type=int, default=20, help="max evidence candidates per claim")
    ap.add_argument("--max-workers", type=int, default=None, help="papers scored concurrently")
    args = ap.parse_args()

    papers = list(load_dir(args.input))
    if not papers:
        print(f"No PubMed XML papers found under {args.input}")
        return

    verdicts = run_filter(papers, limit=args.limit, max_workers=args.max_workers)
    csv_path = write_flat_csv(verdicts, args.out_csv)
    html_path = render_html(
        build_paper_graph_data(verdicts), args.out_html,
        title="MedFact Filter Results",
        subtitle=("Each node is a PubMed paper coloured by its truthfulness verdict. Red dots and "
                "edges mark works that refuted it; green dots and edges mark works that supported it."),
        how_to_read=FILTER_HOW_TO_READ,
    )

    actions = Counter(v.action.value for v in verdicts)
    print(f"Scored {len(verdicts)} papers: "
        f"{actions.get('keep', 0)} keep, {actions.get('downweight', 0)} downweight, "
        f"{actions.get('drop', 0)} drop")
    print(f"CSV:   {csv_path}")
    print(f"Graph: {html_path}")


if __name__ == "__main__":
    main()
