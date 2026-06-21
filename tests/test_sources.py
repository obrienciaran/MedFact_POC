import json
from pathlib import Path

from medfact_poc.schema import Candidate, NormalizedClaim
from medfact_poc.scraping import europepmc, pubmed

FIX = Path(__file__).parent / "fixtures"

_CLAIM = NormalizedClaim(intervention="drug x", outcome="mortality")


def test_parse_efetch_xml_fields():
    cands = pubmed.parse_efetch_xml((FIX / "efetch_sample.xml").read_text())
    assert len(cands) == 2
    a = {c.ext_id: c for c in cands}["11111111"]
    assert a.source == "pubmed"
    assert "increases mortality" in a.title
    assert "increased mortality" in a.abstract
    assert "Randomized Controlled Trial" in a.pub_types
    assert a.year == 2009
    assert a.doi == "10.1056/NEJMoa0000001"
    # RetractionIn -> retracted_by; CommentIn ignored
    assert a.retracted_by == ["99999999"]
    assert a.is_retraction_of == []
    assert a.evidence_tier == 0.8  # RCT


def test_parse_efetch_retraction_of_and_medlinedate():
    cands = {c.ext_id: c for c in pubmed.parse_efetch_xml((FIX / "efetch_sample.xml").read_text())}
    b = cands["22222222"]
    assert b.is_retraction_of == ["33333333"]
    assert b.year == 2010  # parsed from MedlineDate


def test_parse_europepmc_json():
    payload = json.loads((FIX / "europepmc_sample.json").read_text())
    cands = europepmc.parse_search_json(payload)
    assert len(cands) == 2
    # PMID used as ext_id when present (aligns with gold answer keys)
    first = cands[0]
    assert first.ext_id == "33031652"
    assert first.source == "europepmc"
    assert "did not reduce mortality" in first.abstract
    assert first.year == 2020
    # string pubType normalized to list; missing PMID -> source-prefixed id
    second = cands[1]
    assert second.ext_id == "PPR:PPR123456"
    assert second.pub_types == ["Preprint"]


def test_pubmed_source_dedupes_pmids_across_queries(monkeypatch):
    calls = {"esearch": 0}

    def fake_esearch(q, *, retmax, client):
        calls["esearch"] += 1
        return ["1", "2", "1"]  # duplicates within and across query formulations

    captured = {}

    def fake_efetch(pmids, *, client):
        captured["pmids"] = pmids
        return [Candidate(source="pubmed", ext_id=p, title=p) for p in pmids]

    monkeypatch.setattr(pubmed, "esearch", fake_esearch)
    monkeypatch.setattr(pubmed, "efetch", fake_efetch)
    out = pubmed.PubMedSource().search_claim(_CLAIM, limit=5, client=None)
    assert calls["esearch"] >= 2  # several query formulations are issued
    assert captured["pmids"] == ["1", "2"]  # deduped, order preserved, one efetch
    assert [c.ext_id for c in out] == ["1", "2"]


def test_europepmc_source_dedupes_candidates(monkeypatch):
    def fake_search(q, *, page_size, client):
        return [
            Candidate(source="europepmc", ext_id="A", title="A"),
            Candidate(source="europepmc", ext_id="B", title="B"),
        ]

    monkeypatch.setattr(europepmc, "search", fake_search)
    out = europepmc.EuropePMCSource().search_claim(_CLAIM, limit=5, client=None)
    assert sorted(c.ext_id for c in out) == ["A", "B"]
