"""Stance classification: does a candidate support, refute, or stay neutral on the claim
as asserted (conditions included)?

This is measurement instrumentation, not the production annotator. Using an LLM as the
stance judge here is acceptable precisely because the stance step is one of the variables
the harness measures. Its correctness shows up as the gap between retrieval recall and
stance recall. It must never be promoted into an "is this claim true" judge in a real
pipeline.

Backends are pluggable (see ``base.stance``):
  * ``stub`` is a deterministic lexical heuristic, dependency-free, for plumbing.
  * ``llm`` is any provider behind ``LLMClient`` (Claude, ChatGPT, Gemini) with a
    condition-aware prompt returning structured JSON. ``anthropic``, ``openai``, and
    ``gemini`` are accepted as aliases that also pin the provider.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

from ..base.llm import LLMClient
from ..base.stance import StanceBackend
from ..llm import PROVIDERS, get_llm
from ..schema import Candidate, GoldEntry, Stance, StanceLabel

# Lexical cues used only by the stub backend.
_REFUTE_CUES = (
    "no benefit", "no significant", "increased risk", "increased mortality", "harm",
    "did not reduce", "no effect", "ineffective", "not associated with",
    "associated with increased", "retracted", "withdrawn", "no reduction",
)


class StubStance:
    """Lexical heuristic. This is not a real stance signal. It only lets the harness run
    offline. Treat any stub-derived stance recall as a lower-bound placeholder."""

    name = "stub"

    def classify(self, gold: GoldEntry, candidate: Candidate) -> StanceLabel:
        text = f"{candidate.title} {candidate.abstract}".lower()
        hits = [c for c in _REFUTE_CUES if c in text]
        if hits:
            stance = Stance.REFUTES
            conf = min(0.3 + 0.1 * len(hits), 0.7)
            rationale = f"stub: refute cues {hits[:3]}"
        else:
            stance = Stance.NEUTRAL
            conf = 0.2
            rationale = "stub: no refute cues"
        return StanceLabel(
            claim_id=gold.id, candidate_ext_id=candidate.ext_id,
            stance=stance, confidence=conf, rationale=rationale, condition_match=None,
        )


_PROMPT = """You are assessing how a piece of medical evidence relates to a specific claim.

CLAIM (as asserted): {claim_text}
Structured: intervention={intervention}, outcome={outcome}, population={population}, \
comparator={comparator}, asserted direction={direction}

EVIDENCE:
Title: {title}
Publication types: {pub_types}
Year: {year}
Abstract: {abstract}

Decide the stance of the EVIDENCE toward the CLAIM. Critically, only judge "refutes" \
if the evidence concerns the SAME population, intervention and outcome the claim asserts \
AND contradicts its asserted direction. If the evidence tests a DIFFERENT condition \
(different population, comparator, dose, or setting), set condition_match=false and prefer \
"neutral" unless it still directly contradicts.

Respond with a JSON object as shown below and nothing else:
{{"stance": "supports|refutes|neutral", "confidence": 0.0-1.0, \
"condition_match": true|false, "rationale": "one sentence"}}"""


class LLMStance:
    """Condition-aware stance classifier backed by any ``LLMClient`` provider."""

    def __init__(self, client: LLMClient) -> None:
        self.name = f"stance:{client.name}"
        self._client = client

    def classify(self, gold: GoldEntry, candidate: Candidate) -> StanceLabel:
        n = gold.normalized
        prompt = _PROMPT.format(
            claim_text=gold.claim_text,
            intervention=n.intervention, outcome=n.outcome,
            population=n.population or "unspecified",
            comparator=n.comparator or "unspecified",
            direction=n.direction or "unspecified",
            title=candidate.title, pub_types=", ".join(candidate.pub_types) or "unknown",
            year=candidate.year or "unknown",
            abstract=(candidate.abstract or "")[:4000] or "(no abstract)",
        )
        text = self._client.complete(prompt, max_tokens=300)
        return self._parse(gold.id, candidate.ext_id, text)

    @staticmethod
    def _parse(claim_id: str, ext_id: str, text: str) -> StanceLabel:
        try:
            start, end = text.index("{"), text.rindex("}") + 1
            data = json.loads(text[start:end])
            stance = Stance(data.get("stance", "neutral"))
        except (ValueError, KeyError):
            stance, data = Stance.NEUTRAL, {}
        return StanceLabel(
            claim_id=claim_id, candidate_ext_id=ext_id, stance=stance,
            confidence=float(data.get("confidence", 0.0)),
            rationale=str(data.get("rationale", "")),
            condition_match=data.get("condition_match"),
        )


def get_stance_backend() -> StanceBackend:
    backend = os.environ.get("MEDFACT_STANCE_BACKEND", "stub").lower()
    if backend == "stub":
        return StubStance()
    provider = backend if backend in PROVIDERS else None
    return LLMStance(get_llm(provider))


def classify_batch(
    backend: StanceBackend,
    gold: GoldEntry,
    candidates: list[Candidate],
    max_workers: int | None = None,
) -> list[StanceLabel]:
    """Classify candidates concurrently, preserving input order.

    Stance calls against an LLM backend are independent network-bound requests, so they fan
    out across a thread pool (``MEDFACT_STANCE_CONCURRENCY``, default 8). The stub backend
    runs through the same path harmlessly.
    """
    if not candidates:
        return []
    workers = max_workers or int(os.environ.get("MEDFACT_STANCE_CONCURRENCY", "8"))
    workers = max(1, min(workers, len(candidates)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(lambda c: backend.classify(gold, c), candidates))
