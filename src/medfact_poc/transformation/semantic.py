"""Sorts the studies a search already found, putting the most relevant ones first.

An embedder is a model that turns a piece of text into a list of numbers that stands in
for what the text means, so two texts with similar meaning end up with similar numbers
even if they do not share the same words. This module embeds the claim and each
retrieved study's title and abstract, then uses how close those numbers are to sort the
studies from most to least relevant.

This module does not search PubMed or Europe PMC itself. It only re-orders the pool of
studies those searches already returned. If a study was never returned by the search in
the first place, no amount of re-ordering can find it (see the ``not_indexed`` failure
bucket).

Why re-order at all? A claim's pool can hold hundreds of studies, and having a language
model read every one to judge its stance would be slow and expensive. The validation tool
(``orchestration.harness``) uses this ranking to pick the top ``STANCE_TOP_K`` (20 by
default) studies to send to that stance step.

In validation, the study already known to disprove each claim is always sent to the
stance step regardless of where this module ranks it, so a bad ranking can never hide it
from the headline recall numbers. The real filter has no such known answer for new
papers, so a future ranking step there would need to be trustworthy on its own; the
``recall@k`` metric (``reporting/metrics.py``) tracks how often the right study would
have landed near the top.

Two ranking backends are available (see ``base.embedder``):
  * ``stub`` (the default) is fast, offline, and does not understand meaning, so its
    order is close to random. It lets tests and offline runs work without a network
    connection or a downloaded model.
  * ``sbert`` is a real, pretrained biomedical language model. Turn it on with the
    ``embed`` optional dependency and ``MEDFACT_EMBED_BACKEND=sbert``.
"""

from __future__ import annotations

import hashlib
import math
import os

from ..base.embedder import Embedder

_STUB_DIM = 256


class StubEmbedder:
    """Fake, offline ranking based on word hashing. It does not understand meaning, so its
    order is close to random. It exists so tests and offline runs work without a network
    connection or a downloaded model."""

    name = "stub"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * _STUB_DIM
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % _STUB_DIM] += 1.0
        return vec


class SentenceTransformerEmbedder:
    """Real ranking using a pretrained biomedical language model, via the
    sentence-transformers package (an optional dependency)."""

    def __init__(self, model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO") -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.encode(texts, normalize_embeddings=False)]


def get_embedder() -> Embedder:
    backend = os.environ.get("MEDFACT_EMBED_BACKEND", "stub").lower()
    if backend == "sbert":
        return SentenceTransformerEmbedder(
            os.environ.get("MEDFACT_EMBED_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO")
        )
    return StubEmbedder()


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def rank_by_similarity(
    query_vec: list[float], pool: list[tuple[str, list[float]]]
) -> list[tuple[str, float]]:
    """Rank an (ext_id, vector) pool by cosine to query_vec, descending."""
    scored = [(ext_id, cosine(query_vec, vec)) for ext_id, vec in pool]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
