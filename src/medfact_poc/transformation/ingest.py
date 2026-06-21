"""Ingest PubMed/MEDLINE XML into the papers the filter scores.

The end user points the filter at a folder of PubMed XML files (an efetch dump or the
MEDLINE baseline). XML is the right input because it carries structure that plain text
drops: the PMID, separated title and abstract, ``PublicationTypeList`` (the evidence
tier), MeSH terms, and the ``CommentsCorrectionsList`` that names the works which
retracted, corrected, or commented on the paper. Those links are the strongest refutation
signal we get for free.

``parse_pubmed_xml`` is pure and unit-tested. ``load_dir`` walks a directory of ``.xml``
files lazily so a large corpus is streamed rather than read whole into memory. A file that
cannot be read or is not a PubMed article set prints an error and is skipped. A paper with
no ``CommentsCorrectionsList`` prints a highlight, because that is the offline truthfulness
signal, and is still kept.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

from ..schema import PaperRecord
from . import medline

# RefTypes that indicate another work disputes or amends this paper. Kept explicit so the
# retriever and scorer can treat "this was retracted" differently from "this was cited".
DISPUTE_REFTYPES = ("RetractionIn", "CommentIn", "ErratumIn", "CorrectionIn", "UpdateIn", "RepublishedIn")


def parse_pubmed_xml(xml_text: str) -> list[PaperRecord]:
    """Parse a PubmedArticleSet into PaperRecords. Pure and safe to unit test."""
    root = ET.fromstring(xml_text)
    out: list[PaperRecord] = []
    for art in root.findall(".//PubmedArticle"):
        pmid = medline.text(art, ".//MedlineCitation/PMID")
        if not pmid:
            continue
        out.append(
            PaperRecord(
                pmid=pmid,
                title=medline.text(art, ".//Article/ArticleTitle") or "",
                abstract=medline.abstract(art),
                pub_types=medline.pub_types(art),
                mesh=_mesh(art),
                year=medline.year(art),
                journal=medline.text(art, ".//Journal/Title"),
                doi=medline.doi(art),
                comments_corrections=_corrections(art),
            )
        )
    return out


def load_dir(path: str | Path, *, pattern: str = "*.xml") -> Iterator[PaperRecord]:
    """Yield PaperRecords from every readable PubMed XML file under ``path`` (recursive)."""
    root = Path(path)
    files = sorted(root.rglob(pattern)) if root.is_dir() else [root]
    for file in files:
        yield from _load_file(file)


def _load_file(file: Path) -> Iterator[PaperRecord]:
    """Read and parse one file, printing a clear error for anything the filter cannot use."""
    try:
        xml_text = file.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: cannot read {file}, skipping. {exc}")
        return
    try:
        records = parse_pubmed_xml(xml_text)
    except ET.ParseError as exc:
        print(f"ERROR: {file} is not valid XML, skipping. {exc}")
        return
    if not records:
        print(f"ERROR: {file} is not PubMed article XML (no PubmedArticle with a PMID), skipping.")
        return
    for record in records:
        if not record.comments_corrections:
            print(f"NOTE: {record.pmid} in {file.name} has no CommentsCorrectionsList, so the offline truthfulness signal is unavailable.")
        yield record


def _mesh(art: ET.Element) -> list[str]:
    return [
        (el.text or "").strip()
        for el in art.findall(".//MeshHeadingList/MeshHeading/DescriptorName")
        if el.text
    ]


def _corrections(art: ET.Element) -> dict[str, list[str]]:
    """Group CommentsCorrectionsList referenced PMIDs by RefType."""
    grouped: dict[str, list[str]] = {}
    for cc in art.findall(".//CommentsCorrectionsList/CommentsCorrections"):
        ref_type = cc.get("RefType", "")
        ref_pmid = medline.text(cc, "PMID")
        if ref_type and ref_pmid:
            grouped.setdefault(ref_type, []).append(ref_pmid)
    return grouped
