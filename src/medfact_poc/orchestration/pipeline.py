"""Filter orchestration: PubMed paper, claims, evidence, stance, verdict.

For each paper, extract its claims, retrieve evidence for each claim, judge stance, and
score. Papers are independent, so they run concurrently. The default backends are all
offline stubs, so this runs end to end with no key. Swap in real LLM or retriever backends
via env (see ``llm``, ``transformation.extract``, ``scraping.evidence``,
``transformation.stance``).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

from ..base.extractor import ClaimExtractor
from ..base.retriever import Retriever
from ..base.stance import StanceBackend
from ..schema import Action, PaperRecord, PaperVerdict, Verdict
from ..scraping.evidence import get_retriever
from ..transformation.extract import get_extractor
from ..transformation.scoring import score_claim, score_paper
from ..transformation.stance import classify_batch, get_stance_backend


def run_paper(
    paper: PaperRecord,
    *,
    extractor: ClaimExtractor,
    retriever: Retriever,
    stance_backend: StanceBackend,
    limit: int = 20,
) -> PaperVerdict:
    """Score a single paper end to end."""
    claim_verdicts = []
    for claim in extractor.extract(paper):
        candidates = retriever.retrieve(paper, claim, limit=limit)
        labels = classify_batch(stance_backend, claim.as_gold_entry(), candidates)
        claim_verdicts.append(score_claim(claim, candidates, labels))
    return score_paper(paper, claim_verdicts)


def _run_paper_safe(
    paper: PaperRecord,
    *,
    extractor: ClaimExtractor,
    retriever: Retriever,
    stance_backend: StanceBackend,
    limit: int,
) -> PaperVerdict:
    """Run one paper, but turn an unexpected failure into an ``unverified`` row instead of
    aborting the whole batch. A transient network error on one paper should not discard the
    work already done on the others (which matters when those calls cost money)."""
    try:
        return run_paper(
            paper, extractor=extractor, retriever=retriever,
            stance_backend=stance_backend, limit=limit,
        )
    except Exception as exc:  # noqa: BLE001 - isolate one paper's failure from the batch
        print(f"ERROR scoring {paper.pmid}, keeping as unverified. {type(exc).__name__}: {exc}")
        return PaperVerdict(
            pmid=paper.pmid, title=paper.title, verdict=Verdict.UNVERIFIED, score=0.5,
            action=Action.KEEP, n_claims=0, n_refuted_claims=0, top_refuting_tier=0.0,
            notes=f"error: {type(exc).__name__}: {exc}",
        )


def run_filter(
    papers: Iterable[PaperRecord],
    *,
    extractor: ClaimExtractor | None = None,
    retriever: Retriever | None = None,
    stance_backend: StanceBackend | None = None,
    limit: int = 20,
    max_workers: int | None = None,
) -> list[PaperVerdict]:
    """Score a corpus of papers concurrently. Returns one PaperVerdict per paper."""
    extractor = extractor or get_extractor()
    retriever = retriever or get_retriever()
    stance_backend = stance_backend or get_stance_backend()
    papers = list(papers)
    if not papers:
        return []
    workers = max_workers or int(os.environ.get("MEDFACT_FILTER_CONCURRENCY", "4"))
    workers = max(1, min(workers, len(papers)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(
            executor.map(
                lambda p: _run_paper_safe(
                    p, extractor=extractor, retriever=retriever,
                    stance_backend=stance_backend, limit=limit,
                ),
                papers,
            )
        )
