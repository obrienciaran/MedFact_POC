"""Query construction from a normalized claim.

Generates several formulations per claim because retrieval recall is set here. A single
rigid query is the easiest way to manufacture a ``not_indexed`` false negative. The
formulations deliberately bias toward surfacing contradicting high-tier evidence
(systematic reviews, meta-analyses, RCTs, guidelines), because support-seeking-only
queries are the field-wide failure this harness exists to test.
"""

from __future__ import annotations

from ..schema import NormalizedClaim

# Publication types that carry contradiction weight. Used as PubMed filters and as
# Europe PMC query fragments.
HIGH_TIER_PUBMED = (
    '("Meta-Analysis"[ptyp] OR "Systematic Review"[ptyp] OR '
    '"Randomized Controlled Trial"[ptyp] OR "Guideline"[ptyp] OR "Practice Guideline"[ptyp])'
)


def pubmed_queries(claim: NormalizedClaim) -> list[str]:
    """Ordered list of PubMed query strings, from broad to contradiction-targeted."""
    terms = claim.as_query_terms()
    core = " AND ".join(f'"{t}"' for t in (claim.intervention, claim.outcome) if t)
    broad = " AND ".join(f'"{t}"' for t in terms if t)
    queries = [core, broad, f"{core} AND {HIGH_TIER_PUBMED}"]
    # Explicit contradiction-seeking formulation.
    queries.append(f"{core} AND (risk OR harm OR mortality OR increased OR no benefit OR retracted)")
    # Drop duplicates while preserving order.
    seen: set[str] = set()
    out = []
    for q in queries:
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def europepmc_queries(claim: NormalizedClaim) -> list[str]:
    """Ordered list of Europe PMC query strings."""
    core = " AND ".join(claim.as_query_terms()[:2])
    broad = " AND ".join(claim.as_query_terms())
    high_tier = (
        f"({core}) AND (PUB_TYPE:\"Meta-Analysis\" OR PUB_TYPE:\"Systematic Review\" "
        f'OR PUB_TYPE:"Randomized Controlled Trial")'
    )
    queries = [core, broad, high_tier]
    seen: set[str] = set()
    out = []
    for q in queries:
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out
