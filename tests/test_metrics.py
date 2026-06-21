import math

from medfact_poc.reporting.metrics import compute
from medfact_poc.schema import ClaimReport, ClaimStatus, FailureBucket


def _rev(**kw):
    base = dict(
        claim_id="r", status=ClaimStatus.REVERSED, n_candidates=10,
        answer_key_retrieved=False, answer_key_rank=None, refuting_found=False,
        answer_key_recognized=False, top_refuting_tier=0.0,
        failure_bucket=FailureBucket.NONE, false_contradiction=False,
    )
    base.update(kw)
    return ClaimReport(**base)


def _ctrl(false_contradiction):
    return ClaimReport(
        claim_id="c", status=ClaimStatus.STILL_TRUE, n_candidates=5,
        answer_key_retrieved=False, answer_key_rank=None, refuting_found=false_contradiction,
        answer_key_recognized=False, top_refuting_tier=0.0,
        failure_bucket=FailureBucket.NONE, false_contradiction=false_contradiction,
    )


def test_decomposed_recall_math():
    reports = [
        _rev(answer_key_retrieved=True, answer_key_rank=2, answer_key_recognized=True, refuting_found=True),
        _rev(answer_key_retrieved=True, answer_key_rank=8, failure_bucket=FailureBucket.RETRIEVED_NOT_RECOGNIZED),
        _rev(answer_key_retrieved=False, failure_bucket=FailureBucket.NOT_INDEXED),
        _rev(answer_key_retrieved=True, answer_key_rank=1, answer_key_recognized=True, refuting_found=True),
        _ctrl(True),
        _ctrl(False),
    ]
    m = compute(reports)
    assert m.n_reversed == 4 and m.n_controls == 2
    assert math.isclose(m.retrieval_recall, 0.75)
    assert math.isclose(m.recall_at_k[1], 0.25)
    assert math.isclose(m.recall_at_k[5], 0.5)
    assert math.isclose(m.recall_at_k[10], 0.75)
    # conditional stance recall: 2 recognized of 3 retrieved
    assert math.isclose(m.stance_recall_conditional, 2 / 3, rel_tol=1e-6)
    assert math.isclose(m.stance_recall_overall, 0.5)
    assert math.isclose(m.soft_refutation_recall, 0.5)  # r1 and r4
    assert math.isclose(m.false_contradiction_rate, 0.5)
    assert m.failure_taxonomy == {"retrieved_not_recognized": 1, "not_indexed": 1}


def test_empty_is_safe():
    m = compute([])
    assert m.retrieval_recall == 0.0
    assert m.failure_taxonomy == {}
