"""Semantic ranking within a retrieved candidate pool.

Important scoping note: there is no local full-text corpus, so this channel does not
perform dense retrieval over external literature. It re-ranks the pool that the keyword
and source channels surfaced, by cosine similarity to the claim. It therefore measures the
rankability of contradicting evidence given our queries. It cannot recover documents the
API queries never returned, and that limitation is itself a finding the harness should
surface (see the ``not_indexed`` bucket).

The embedding backend is pluggable (see ``base.embedder``). The default ``stub`` backend
is deterministic and dependency-free so the whole harness runs offline. Install the
``embed`` extra and set ``MEDFACT_EMBED_BACKEND=sbert`` for real biomedical embeddings.
"""

from __future__ import annotations

import hashlib
import math
import os

from ..base.embedder import Embedder

_STUB_DIM = 256


class StubEmbedder:
    """Deterministic hashing embedder. Not semantically meaningful. It exists so the
    plumbing runs and tests are reproducible without network or model weights."""

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
    """Real biomedical embeddings via sentence-transformers (optional dependency)."""

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
