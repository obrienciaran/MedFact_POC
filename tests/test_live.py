"""Opt-in live network test. Deselected by default (see pyproject addopts).

Run explicitly with:  pytest -m live
Requires outbound HTTPS to NCBI E-utilities. Set NCBI_EMAIL to be polite.
"""

import pytest

from medfact_poc.scraping import pubmed

pytestmark = pytest.mark.live


def test_esearch_efetch_roundtrip_live():
    # WHI 2002 — a known landmark; verify we can search and fetch it.
    pmids = pubmed.esearch(
        '"estrogen plus progestin" AND "coronary heart disease" AND "postmenopausal"', retmax=20
    )
    assert pmids, "esearch returned no PMIDs"
    cands = pubmed.efetch(pmids[:5])
    assert cands
    assert any(c.year and c.year <= 2010 for c in cands)
