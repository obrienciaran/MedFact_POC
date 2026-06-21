"""Turn stance labels into per-claim and per-paper truthfulness verdicts.

The score is grounded in evidence rather than a model opinion. Refutation strength is the
stance confidence weighted by the refuting study's evidence tier, so a high-tier RCT or
guideline that contradicts a claim moves the verdict far more than a case report. A paper
is judged as harshly as its most damning claim, because one confidently wrong central claim
is enough to want the paper down-weighted or dropped from training.

These functions are pure and unit-tested. Thresholds are module constants so the policy is
visible and tunable rather than buried.
"""

from __future__ import annotations

from ..schema import (
    Action,
    Candidate,
    ClaimVerdict,
    ExtractedClaim,
    PaperRecord,
    PaperVerdict,
    Stance,
    StanceLabel,
    Verdict,
)

REFUTE_THRESHOLD = 0.5  # tier-weighted refutation strength at/above which a claim is refuted

# Default action per verdict. Unverified is kept, because a missing refutation is not
# evidence of falsity. The continuous score is also written so the user can set a threshold.
_ACTION_BY_VERDICT = {
    Verdict.REFUTED: Action.DROP,
    Verdict.CONTESTED: Action.DOWNWEIGHT,
    Verdict.SUPPORTED: Action.KEEP,
    Verdict.UNVERIFIED: Action.KEEP,
}


def score_claim(
    claim: ExtractedClaim, candidates: list[Candidate], labels: list[StanceLabel]
) -> ClaimVerdict:
    """Weigh one claim's evidence into a verdict and a 0..1 truthfulness score."""
    tier = {c.ext_id: c.evidence_tier for c in candidates}
    refuting = [l for l in labels if l.stance is Stance.REFUTES]
    supporting = [l for l in labels if l.stance is Stance.SUPPORTS]

    refute_strength = max((tier.get(l.candidate_ext_id, 0.4) * l.confidence for l in refuting), default=0.0)
    support_strength = max((tier.get(l.candidate_ext_id, 0.4) * l.confidence for l in supporting), default=0.0)
    top_refuting_tier = max((tier.get(l.candidate_ext_id, 0.4) for l in refuting), default=0.0)

    score = _clamp(0.5 + 0.4 * support_strength - 0.8 * refute_strength)
    verdict = _claim_verdict(len(labels), bool(refuting), bool(supporting), refute_strength)

    return ClaimVerdict(
        claim_id=claim.claim_id, claim_text=claim.claim_text,
        n_evidence=len(labels), n_refuting=len(refuting), n_supporting=len(supporting),
        top_refuting_tier=top_refuting_tier, verdict=verdict, score=score,
        refuting_pmids=[l.candidate_ext_id for l in refuting],
        supporting_pmids=[l.candidate_ext_id for l in supporting],
    )


def _claim_verdict(n_evidence: int, has_refute: bool, has_support: bool, refute_strength: float) -> Verdict:
    if n_evidence == 0:
        return Verdict.UNVERIFIED
    if has_refute and has_support:
        return Verdict.CONTESTED
    if has_refute:
        return Verdict.REFUTED if refute_strength >= REFUTE_THRESHOLD else Verdict.CONTESTED
    if has_support:
        return Verdict.SUPPORTED
    return Verdict.UNVERIFIED  # only neutral evidence


def score_paper(paper: PaperRecord, claim_verdicts: list[ClaimVerdict]) -> PaperVerdict:
    """Roll per-claim verdicts up to the paper, judged by its most damning claim."""
    if not claim_verdicts:
        return PaperVerdict(
            pmid=paper.pmid, title=paper.title, verdict=Verdict.UNVERIFIED, score=0.5,
            action=Action.KEEP, n_claims=0, n_refuted_claims=0, top_refuting_tier=0.0,
            notes="no claims extracted",
        )

    score = min(cv.score for cv in claim_verdicts)
    verdict = _paper_verdict(claim_verdicts)
    refuting_pmids = sorted({p for cv in claim_verdicts for p in cv.refuting_pmids})
    return PaperVerdict(
        pmid=paper.pmid, title=paper.title, verdict=verdict, score=score,
        action=_ACTION_BY_VERDICT[verdict],
        n_claims=len(claim_verdicts),
        n_refuted_claims=sum(1 for cv in claim_verdicts if cv.verdict is Verdict.REFUTED),
        top_refuting_tier=max(cv.top_refuting_tier for cv in claim_verdicts),
        refuting_pmids=refuting_pmids, claim_verdicts=claim_verdicts,
    )


def _paper_verdict(claim_verdicts: list[ClaimVerdict]) -> Verdict:
    verdicts = {cv.verdict for cv in claim_verdicts}
    if Verdict.REFUTED in verdicts:
        return Verdict.REFUTED
    if Verdict.CONTESTED in verdicts:
        return Verdict.CONTESTED
    if Verdict.SUPPORTED in verdicts:
        return Verdict.SUPPORTED
    return Verdict.UNVERIFIED


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
