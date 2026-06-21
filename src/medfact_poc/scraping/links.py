"""Pool expansion via PubMed retraction and correction links.

A secondary contradiction signal. If any candidate in the pool carries a
``RetractionIn`` link (it was retracted) or is itself a retraction notice, the
referenced PMIDs are strong contradiction evidence and should be pulled into the pool
even if the keyword queries missed them.

For the consensus-reversal slice this fires rarely, because reversals are usually
superseding higher-tier studies rather than formal retractions. That is why it is
secondary. It is cheap and occasionally decisive, for example in fraud cases.
"""

from __future__ import annotations

import httpx

from ..schema import Candidate
from . import pubmed


def expand_via_links(
    pool: list[Candidate], *, client: httpx.Client | None = None
) -> list[Candidate]:
    """Fetch PMIDs referenced by retraction links on pool members.

    Returns only the newly fetched candidates. The caller merges them into the pool.
    """
    have = {c.ext_id for c in pool}
    wanted: set[str] = set()
    for c in pool:
        for pmid in (*c.retracted_by, *c.is_retraction_of):
            if pmid not in have:
                wanted.add(pmid)
    if not wanted:
        return []
    fetched = pubmed.efetch(sorted(wanted), client=client)
    for c in fetched:
        c.retrieved_by = ["links"]
    return fetched
