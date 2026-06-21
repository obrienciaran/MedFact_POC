"""Leaf extractors for MEDLINE/PubMed efetch XML.

Both the harness source (``scraping/pubmed.py``, which builds Candidates from a live
efetch call) and the filter ingester (``transformation/ingest.py``, which builds
PaperRecords from corpus files) parse the same ``PubmedArticle`` schema. These pure
element helpers are the shared leaves, kept in one place so the two parsers cannot drift.
Each parser still owns how it assembles its own model and which relationships it keeps.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET


def text(node: ET.Element, path: str) -> str | None:
    el = node.find(path)
    return el.text.strip() if el is not None and el.text else None


def abstract(art: ET.Element) -> str:
    return " ".join(
        (el.text or "").strip() for el in art.findall(".//Abstract/AbstractText")
    ).strip()


def pub_types(art: ET.Element) -> list[str]:
    return [
        (el.text or "").strip()
        for el in art.findall(".//PublicationTypeList/PublicationType")
        if el.text
    ]


def year(art: ET.Element) -> int | None:
    y = text(art, ".//JournalIssue/PubDate/Year")
    if y and y.isdigit():
        return int(y)
    medline_date = text(art, ".//JournalIssue/PubDate/MedlineDate")
    if medline_date and medline_date[:4].isdigit():
        return int(medline_date[:4])
    return None


def doi(art: ET.Element) -> str | None:
    for el in art.findall(".//ArticleIdList/ArticleId"):
        if el.get("IdType") == "doi" and el.text:
            return el.text.strip()
    for el in art.findall(".//ELocationID"):
        if el.get("EIdType") == "doi" and el.text:
            return el.text.strip()
    return None
