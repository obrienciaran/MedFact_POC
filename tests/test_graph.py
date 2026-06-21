from medfact_poc.reporting.graph import build_graph_data, render_html
from medfact_poc.schema import (
    Candidate,
    ClaimStatus,
    GoldEntry,
    GoldSet,
    NormalizedClaim,
    Stance,
    StanceLabel,
)
from medfact_poc.store import Store


def _gold():
    return GoldSet(
        entries=[
            GoldEntry(
                id="c1", claim_text="Drug X reduces mortality.",
                normalized=NormalizedClaim(intervention="drug x", outcome="mortality"),
                status=ClaimStatus.REVERSED, answer_key=["A", "MISSING"],
            ),
            GoldEntry(
                id="c2", claim_text="Statins help.",
                normalized=NormalizedClaim(intervention="statin", outcome="death"),
                status=ClaimStatus.STILL_TRUE,
            ),
        ]
    )


def _seed_store(store):
    a = Candidate(source="pubmed", ext_id="A", title="Trial A", year=2010,
                  pub_types=["Randomized Controlled Trial"], retracted_by=["B"])
    b = Candidate(source="pubmed", ext_id="B", title="Retraction of A", year=2012,
                  pub_types=["Retraction of Publication"])
    store.upsert_candidates([a, b])
    store.upsert_stance([
        StanceLabel(claim_id="c1", candidate_ext_id="A", stance=Stance.REFUTES, confidence=0.8),
        StanceLabel(claim_id="c1", candidate_ext_id="B", stance=Stance.NEUTRAL, confidence=0.3),
    ])


def test_build_graph_data(tmp_path):
    gold = _gold()
    with Store(tmp_path / "g.duckdb") as store:
        _seed_store(store)
        data = build_graph_data(gold, store)

    node_ids = {n.id for n in data.nodes}
    assert {"claim:c1", "claim:c2", "ev:A", "ev:B", "ev:MISSING"} <= node_ids

    ev_a = next(n for n in data.nodes if n.id == "ev:A")
    assert ev_a.is_answer_key
    assert not next(n for n in data.nodes if n.id == "ev:B").is_answer_key

    stance_edges = {(e.source, e.target): e for e in data.edges if e.kind == "stance"}
    assert stance_edges[("claim:c1", "ev:A")].stance == "refutes"
    assert stance_edges[("claim:c1", "ev:B")].stance == "neutral"

    assert any(e.kind == "missing" and e.target == "ev:MISSING" for e in data.edges)
    # A reciprocal retraction link collapses to a single arrow from the notice (B) to the
    # paper it withdrew (A), so the two purple labels can no longer overlap.
    retraction_edges = [e for e in data.edges if e.kind == "retraction"]
    assert len(retraction_edges) == 1
    assert retraction_edges[0].source == "ev:B" and retraction_edges[0].target == "ev:A"
    assert retraction_edges[0].label == "retracts"


def test_render_html_smoke(tmp_path):
    gold = _gold()
    out = tmp_path / "graph.html"
    with Store(tmp_path / "g.duckdb") as store:
        _seed_store(store)
        data = build_graph_data(gold, store)
    render_html(data, out, physics=False)
    assert out.exists()
    html = out.read_text()
    assert "vis-network" in html  # rendering engine loaded
    assert "ev:A" in html  # an answer-key node made it into the page
    assert "ev:MISSING" in html  # the recall-gap node is present
