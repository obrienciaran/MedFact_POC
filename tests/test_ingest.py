from pathlib import Path

from medfact_poc.transformation.ingest import parse_pubmed_xml

FIXTURE = Path(__file__).parent / "fixtures" / "efetch_sample.xml"


def test_parse_pubmed_xml():
    papers = parse_pubmed_xml(FIXTURE.read_text())
    by_pmid = {p.pmid: p for p in papers}

    assert set(by_pmid) == {"11111111", "22222222"}

    rct = by_pmid["11111111"]
    assert rct.title.startswith("Intensive glucose control")
    assert "Randomized Controlled Trial" in rct.pub_types
    assert rct.doi == "10.1056/NEJMoa0000001"
    assert rct.year == 2009
    assert "increased mortality" in rct.abstract
    # the comment/correction graph is grouped by RefType
    assert rct.comments_corrections["RetractionIn"] == ["99999999"]
    assert rct.comments_corrections["CommentIn"] == ["88888888"]

    notice = by_pmid["22222222"]
    assert notice.year == 2010  # parsed from a MedlineDate
    assert notice.comments_corrections == {"RetractionOf": ["33333333"]}


def test_parse_skips_records_without_pmid():
    xml = "<PubmedArticleSet><PubmedArticle><MedlineCitation></MedlineCitation></PubmedArticle></PubmedArticleSet>"
    assert parse_pubmed_xml(xml) == []
