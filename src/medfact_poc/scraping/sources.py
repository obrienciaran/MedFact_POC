"""The active set of evidence providers.

Each provider implements the Source interface (see ``base.source``). Add a provider to
``get_sources`` to include it in retrieval. Order is not significant because the harness
deduplicates candidates by ext_id across sources.
"""

from __future__ import annotations

from ..base.source import Source


def get_sources() -> list[Source]:
    """Return the active evidence providers used by retrieval."""
    from .europepmc import EuropePMCSource
    from .pubmed import PubMedSource

    return [PubMedSource(), EuropePMCSource()]
