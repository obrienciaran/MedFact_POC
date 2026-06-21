"""Render harness results to a markdown report and a per-claim CSV."""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

from ..schema import ClaimReport, GoldSet
from .metrics import Metrics

REPORT_DIR = Path("reports")


def write_reports(
    gold: GoldSet,
    reports: list[ClaimReport],
    metrics: Metrics,
    *,
    embedder_name: str,
    stance_name: str,
    out_dir: str | Path = REPORT_DIR,
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    md_path = out_dir / f"recall-{stamp}.md"
    csv_path = out_dir / f"recall-{stamp}.csv"
    md_path.write_text(_render_md(gold, reports, metrics, embedder_name, stance_name))
    _write_csv(csv_path, gold, reports)
    return md_path, csv_path


def _render_md(
    gold: GoldSet, reports: list[ClaimReport], m: Metrics, embedder: str, stance: str
) -> str:
    pct = lambda x: f"{x * 100:.0f}%"
    by_id = {g.id: g for g in gold.entries}
    lines = [
        "# Negative-Evidence Recall Harness Report",
        "",
        f"- Embedder: `{embedder}`  |  Stance backend: `{stance}`",
        f"- Gold: {m.n_reversed} reversed claims, {m.n_controls} controls",
        "",
        "## Headline",
        "",
        "| Metric | Value | Notes |",
        "|---|---|---|",
        f"| **Retrieval recall** | **{pct(m.retrieval_recall)}** | answer-key doc in pool, stance-independent (most trustworthy) |",
        f"| Recall@1 / @5 / @10 / @20 | {pct(m.recall_at_k[1])} / {pct(m.recall_at_k[5])} / {pct(m.recall_at_k[10])} / {pct(m.recall_at_k[20])} | rank of answer-key doc by semantic similarity |",
        f"| Stance recall (conditional) | {pct(m.stance_recall_conditional)} | of retrieved answer-key docs, fraction recognized as refuting |",
        f"| Stance recall (overall) | {pct(m.stance_recall_overall)} | retrieval and stance combined |",
        f"| Soft refutation recall | {pct(m.soft_refutation_recall)} | any refuting doc found (stance-dependent, softer) |",
        f"| **False-contradiction rate** | **{pct(m.false_contradiction_rate)}** | controls wrongly flagged contradicted (lower is better) |",
        "",
        "## Failure taxonomy (reversed claims that missed)",
        "",
    ]
    if m.failure_taxonomy:
        lines.append("| Bucket | Count |")
        lines.append("|---|---|")
        for bucket, n in sorted(m.failure_taxonomy.items(), key=lambda x: -x[1]):
            lines.append(f"| `{bucket}` | {n} |")
    else:
        lines.append("_No failures on reversed claims._")
    lines += [
        "",
        "## Per-claim",
        "",
        "| Claim | Status | #cand | AK retrieved | AK rank | AK recognized | refuting | bucket |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        g = by_id.get(r.claim_id)
        label = (g.claim_text[:48] + "…") if g and len(g.claim_text) > 48 else (g.claim_text if g else r.claim_id)
        lines.append(
            f"| {label} | {r.status.value} | {r.n_candidates} | "
            f"{'✓' if r.answer_key_retrieved else '✗'} | {r.answer_key_rank or '-'} | "
            f"{'✓' if r.answer_key_recognized else '✗'} | "
            f"{'✓' if r.refuting_found else '✗'} | `{r.failure_bucket.value}` |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "- **Retrieval recall** is the floor on everything downstream. If a known landmark",
        "  refutation is not even retrieved, no scoring layer can use it.",
        "- A large gap between retrieval recall and stance recall means the bottleneck is the",
        "  stance judge rather than retrieval, which calls for a different fix.",
        "- `not_indexed` and `entity_miss` are retrieval failures. `retrieved_not_recognized`,",
        "  `condition_mismatch`, and `tier_inversion` are stance or aggregation failures.",
        "- With a stub stance or embedder backend these numbers are placeholders for plumbing",
        "  validation only. Re-run with `MEDFACT_STANCE_BACKEND=anthropic` and `MEDFACT_EMBED_BACKEND=sbert`",
        "  for a real measurement.",
        "",
    ]
    return "\n".join(lines)


def _write_csv(path: Path, gold: GoldSet, reports: list[ClaimReport]) -> None:
    by_id = {g.id: g for g in gold.entries}
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "claim_id", "claim_text", "status", "n_candidates", "answer_key_retrieved",
            "answer_key_rank", "answer_key_recognized", "refuting_found",
            "top_refuting_tier", "failure_bucket", "false_contradiction",
        ])
        for r in reports:
            g = by_id.get(r.claim_id)
            w.writerow([
                r.claim_id, g.claim_text if g else "", r.status.value, r.n_candidates,
                r.answer_key_retrieved, r.answer_key_rank or "", r.answer_key_recognized,
                r.refuting_found, f"{r.top_refuting_tier:.2f}", r.failure_bucket.value,
                r.false_contradiction,
            ])
