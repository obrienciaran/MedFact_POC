"""Aggregate ClaimReports into the harness's headline metrics.

The decomposition is the whole point. Retrieval recall (stance-independent, the
trustworthy number) is reported separately from stance recall (conditional on retrieval).
Recall@k shows rank sensitivity. The false-contradiction rate guards against a harness that
flags everything. The failure taxonomy says where recall dies.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from pydantic import BaseModel

from ..schema import ClaimReport, ClaimStatus, FailureBucket

RECALL_KS = (1, 5, 10, 20)


class Metrics(BaseModel):
    n_reversed: int
    n_controls: int
    # ground-truth-anchored, stance-independent
    retrieval_recall: float
    recall_at_k: dict[int, float]
    # stance-dependent
    stance_recall_conditional: float  # among reversed where answer-key was retrieved
    stance_recall_overall: float  # over all reversed
    soft_refutation_recall: float  # any refuting doc found (softer, stance-dependent)
    # control quality
    false_contradiction_rate: float
    # diagnosis
    failure_taxonomy: dict[str, int]


def compute(reports: list[ClaimReport]) -> Metrics:
    reversed_ = [r for r in reports if r.status is ClaimStatus.REVERSED]
    controls = [r for r in reports if r.status is ClaimStatus.STILL_TRUE]
    nr = len(reversed_)

    retrieval_recall = _mean(r.answer_key_retrieved for r in reversed_)

    recall_at_k = {}
    for k in RECALL_KS:
        recall_at_k[k] = _mean(
            (r.answer_key_rank is not None and r.answer_key_rank <= k) for r in reversed_
        )

    retrieved = [r for r in reversed_ if r.answer_key_retrieved]
    stance_recall_conditional = _mean(r.answer_key_recognized for r in retrieved)
    stance_recall_overall = _mean(r.answer_key_recognized for r in reversed_)
    soft_refutation_recall = _mean(r.refuting_found for r in reversed_)

    false_contradiction_rate = _mean(r.false_contradiction for r in controls)

    taxonomy = Counter(r.failure_bucket.value for r in reversed_ if r.failure_bucket is not FailureBucket.NONE)

    return Metrics(
        n_reversed=nr,
        n_controls=len(controls),
        retrieval_recall=retrieval_recall,
        recall_at_k=recall_at_k,
        stance_recall_conditional=stance_recall_conditional,
        stance_recall_overall=stance_recall_overall,
        soft_refutation_recall=soft_refutation_recall,
        false_contradiction_rate=false_contradiction_rate,
        failure_taxonomy=dict(taxonomy),
    )


def _mean(bools: Iterable[bool]) -> float:
    vals = [1.0 if b else 0.0 for b in bools]
    return sum(vals) / len(vals) if vals else 0.0
