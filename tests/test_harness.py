import pytest

from medfact_poc.orchestration import harness
from medfact_poc.transformation.semantic import StubEmbedder
from medfact_poc.schema import (
    Candidate,
    ClaimStatus,
    FailureBucket,
    GoldEntry,
    NormalizedClaim,
    Stance,
    StanceLabel,
)
from medfact_poc.transformation.stance import StubStance
from medfact_poc.store import Store


def _gold(status=ClaimStatus.REVERSED, answer_key=("A",)):
    return GoldEntry(
        id="t1", claim_text="Drug X reduces mortality.",
        normalized=NormalizedClaim(intervention="drug x", outcome="mortality"),
        status=status, answer_key=list(answer_key),
    )


def _cand(ext_id, abstract="", pub_types=()):
    return Candidate(source="pubmed", ext_id=ext_id, title=ext_id, abstract=abstract,
                     pub_types=list(pub_types))


def _run(monkeypatch, gold, pool, tmp_path):
    monkeypatch.setattr(harness, "retrieve_pool", lambda g, **kw: pool)
    with Store(tmp_path / "t.duckdb") as store:
        store.upsert_candidates(pool)
        return harness.run_claim(
            gold, store=store, client=None, embedder=StubEmbedder(),
            stance_backend=StubStance(), use_cache=True,
        )


def test_success_answer_key_recognized(monkeypatch, tmp_path):
    pool = [_cand("A", "treatment showed no benefit and increased risk"), _cand("B", "unrelated")]
    report, _ = _run(monkeypatch, _gold(), pool, tmp_path)
    assert report.answer_key_retrieved
    assert report.refuting_found
    assert report.answer_key_recognized
    assert report.failure_bucket is FailureBucket.NONE


def test_not_indexed_when_answer_key_absent(monkeypatch, tmp_path):
    pool = [_cand(x, "neutral text") for x in ("B", "C", "D", "E")]
    report, _ = _run(monkeypatch, _gold(), pool, tmp_path)
    assert not report.answer_key_retrieved
    assert report.failure_bucket is FailureBucket.NOT_INDEXED


def test_entity_miss_on_tiny_pool(monkeypatch, tmp_path):
    pool = [_cand("B", "neutral")]
    report, _ = _run(monkeypatch, _gold(), pool, tmp_path)
    assert report.failure_bucket is FailureBucket.ENTITY_MISS


def test_control_false_contradiction(monkeypatch, tmp_path):
    pool = [_cand("Z", "this therapy showed no benefit"), _cand("Y", "supportive")]
    gold = _gold(status=ClaimStatus.STILL_TRUE, answer_key=())
    report, _ = _run(monkeypatch, gold, pool, tmp_path)
    assert report.status is ClaimStatus.STILL_TRUE
    assert report.refuting_found  # stub fired on "no benefit"
    assert report.false_contradiction
    assert report.failure_bucket is FailureBucket.NONE


def test_classify_failure_condition_mismatch():
    gold = _gold()
    labels = [StanceLabel(claim_id="t1", candidate_ext_id="A", stance=Stance.NEUTRAL,
                          condition_match=False)]
    bucket = harness._classify_failure(
        gold, [_cand("A")], labels, answer_key_retrieved=True,
        answer_key_recognized=False, refuting_found=False, top_refuting_tier=0.0,
        answer_set={"A"},
    )
    assert bucket is FailureBucket.CONDITION_MISMATCH


def test_classify_failure_tier_inversion():
    gold = _gold()
    labels = [StanceLabel(claim_id="t1", candidate_ext_id="A", stance=Stance.NEUTRAL,
                          condition_match=True)]
    bucket = harness._classify_failure(
        gold, [_cand("A")], labels, answer_key_retrieved=True,
        answer_key_recognized=False, refuting_found=True, top_refuting_tier=0.2,
        answer_set={"A"},
    )
    assert bucket is FailureBucket.TIER_INVERSION


def test_load_gold_real_file():
    gs = harness.load_gold()
    assert gs.reversed_entries and gs.control_entries
    # every reversed claim must carry at least one answer-key PMID
    assert all(e.answer_key for e in gs.reversed_entries)
    # controls carry none
    assert all(not e.answer_key for e in gs.control_entries)
