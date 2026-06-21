"""Query construction from a normalized claim.

Generates several formulations per claim because retrieval recall is set here. A single
rigid query is the easiest way to manufacture a ``not_indexed`` false negative. The
formulations run as a ladder from loose (bare concept terms) to contradiction-targeted
(high-tier publication types, harm/risk language), and the caller unions their results, so
no single over-narrow query can drop the disproving study.

Terms are passed UNQUOTED so the database's automatic term mapping can match them. Quoting a
long multi-word phrase forces an exact-string match that descriptive claim text almost never
satisfies (a quoted ``"estrogen plus progestin hormone replacement therapy"`` will not match
the landmark trial that phrases it differently). Terms are also sanitized first to strip the
parenthetical annotations and punctuation carried in from the claim's prose.
"""

from __future__ import annotations

import re

from ..schema import NormalizedClaim

# Publication types that carry contradiction weight. Used as PubMed filters and as
# Europe PMC query fragments.
HIGH_TIER_PUBMED = (
    '("Meta-Analysis"[ptyp] OR "Systematic Review"[ptyp] OR '
    '"Randomized Controlled Trial"[ptyp] OR "Guideline"[ptyp] OR "Practice Guideline"[ptyp])'
)

# Contradiction-seeking fragment. "no benefit" stays quoted as a genuine short phrase.
_CONTRADICTION_PUBMED = '(risk OR harm OR mortality OR increased OR "no benefit" OR retracted)'


def _sanitize(term: str | None) -> str:
    """Reduce a prose claim term to a plain keyword string.

    Drops parenthetical annotations (``"stress and gastric acid (etiologic claim)"`` carries
    the note ``(etiologic claim)``) and punctuation, while keeping hyphens so tokens like
    ``beta-carotene`` survive. Returns an empty string for empty input.
    """
    if not term:
        return ""
    without_parens = re.sub(r"\([^)]*\)", " ", term)
    without_punct = re.sub(r"[^\w\s-]", " ", without_parens)
    return re.sub(r"\s+", " ", without_punct).strip()


def _dedup(queries: list[str]) -> list[str]:
    """Drop empty and duplicate queries while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def pubmed_queries(claim: NormalizedClaim) -> list[str]:
    """Ordered PubMed queries; the caller unions the results.

    Three rungs, chosen because a per-rung contribution check on the gold slice found these
    cover every answer key a wider set did, so retrieval stays the same while the query count
    per claim stays low: ``core`` for broad relevance, ``high-tier`` to surface the trials and
    reviews that overturn consensus, and a contradiction-seeking rung. A core-plus-population
    rung was dropped because it added no answer key the others missed and tended to over-narrow.
    """
    intervention = _sanitize(claim.intervention)
    outcome = _sanitize(claim.outcome)
    core = " AND ".join(t for t in (intervention, outcome) if t)
    if not core:
        return []
    queries = [
        core,                                    # loose: let automatic term mapping work
        f"{core} AND {HIGH_TIER_PUBMED}",        # high-tier: surfaces landmark trials / reviews
        f"{core} AND {_CONTRADICTION_PUBMED}",   # contradiction-seeking
    ]
    return _dedup(queries)


def europepmc_queries(claim: NormalizedClaim) -> list[str]:
    """Ordered Europe PMC queries, mirroring the PubMed core and high-tier rungs."""
    intervention = _sanitize(claim.intervention)
    outcome = _sanitize(claim.outcome)
    core = " AND ".join(t for t in (intervention, outcome) if t)
    if not core:
        return []
    high_tier = (
        f"({core}) AND (PUB_TYPE:\"Meta-Analysis\" OR PUB_TYPE:\"Systematic Review\" "
        f'OR PUB_TYPE:"Randomized Controlled Trial")'
    )
    return _dedup([core, high_tier])
