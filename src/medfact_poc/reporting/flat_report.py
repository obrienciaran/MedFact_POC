"""Write the filter's verdicts to the flat table the end user consumes.

One row per paper. The first columns are the paper identifier and the truthfulness
verdict. The rest are the metadata a curator needs to act on or audit a row: the score, the
recommended training action, how many claims were refuted, the strongest refuting evidence
tier, and the PMIDs of the refuting works.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ..schema import PaperVerdict

COLUMNS = [
    "pmid", "title", "verdict", "score", "action", "n_claims", "n_refuted_claims",
    "top_refuting_tier", "refuting_pmids", "notes",
]


def write_flat_csv(verdicts: list[PaperVerdict], path: str | Path) -> Path:
    """Write one row per paper to ``path``. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for v in verdicts:
            w.writerow([
                v.pmid, v.title, v.verdict.value, f"{v.score:.3f}", v.action.value,
                v.n_claims, v.n_refuted_claims, f"{v.top_refuting_tier:.2f}",
                ";".join(v.refuting_pmids), v.notes,
            ])
    return path
