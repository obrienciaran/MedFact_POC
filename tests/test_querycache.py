from medfact_poc.schema import Candidate
from medfact_poc.scraping import pubmed, querycache
from medfact_poc.scraping.querycache import QueryCache


def test_query_cache_roundtrip(tmp_path):
    cache = QueryCache(tmp_path / "q.duckdb")
    assert cache.get("pubmed", "aspirin AND stroke", 20) is None
    cache.put("pubmed", "aspirin AND stroke", 20, '["1","2"]')
    assert cache.get("pubmed", "aspirin AND stroke", 20) == '["1","2"]'
    # page_size is part of the key
    assert cache.get("pubmed", "aspirin AND stroke", 50) is None
    # conflicting put updates the payload
    cache.put("pubmed", "aspirin AND stroke", 20, '["3"]')
    assert cache.get("pubmed", "aspirin AND stroke", 20) == '["3"]'


def test_get_query_cache_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MEDFACT_QUERY_CACHE", raising=False)
    monkeypatch.setattr(querycache, "_resolved", False)
    monkeypatch.setattr(querycache, "_cache", None)
    assert querycache.get_query_cache() is None


def test_esearch_uses_cache(tmp_path, monkeypatch):
    calls = {"n": 0}

    class _Resp:
        @staticmethod
        def json():
            return {"esearchresult": {"idlist": ["111", "222"]}}

    def fake_ncbi_get(client, url, params):
        calls["n"] += 1
        return _Resp()

    cache = QueryCache(tmp_path / "q.duckdb")
    monkeypatch.setattr(pubmed, "get_query_cache", lambda: cache)
    monkeypatch.setattr(pubmed, "_ncbi_get", fake_ncbi_get)

    first = pubmed.esearch("statins AND mortality", retmax=20, client=object())
    second = pubmed.esearch("statins AND mortality", retmax=20, client=object())
    assert first == second == ["111", "222"]
    assert calls["n"] == 1  # second call served from cache, no network


def test_efetch_caches_records(tmp_path, monkeypatch):
    fetched_batches = []

    def fake_network(pmids, *, client=None):
        fetched_batches.append(list(pmids))
        return [Candidate(source="pubmed", ext_id=p, title=f"t{p}") for p in pmids]

    cache = QueryCache(tmp_path / "q.duckdb")
    monkeypatch.setattr(pubmed, "get_query_cache", lambda: cache)
    monkeypatch.setattr(pubmed, "_efetch_network", fake_network)

    first = pubmed.efetch(["1", "2"])
    assert [c.ext_id for c in first] == ["1", "2"]
    assert fetched_batches == [["1", "2"]]  # both fetched from network

    # second call overlaps: only the new id hits the network, order preserved
    second = pubmed.efetch(["2", "3"])
    assert [c.ext_id for c in second] == ["2", "3"]
    assert fetched_batches == [["1", "2"], ["3"]]
