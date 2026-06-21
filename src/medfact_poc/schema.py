"""Pydantic models for the negative-evidence recall harness.

The flow these model: a GoldEntry feeds retrieval, which produces Candidates. Each
Candidate is judged by the stance step into a StanceLabel, and the per-claim outcome is
summarized as a ClaimReport.

A ``GoldEntry`` is ground truth we hand-authored. A ``Candidate`` is one piece of evidence
retrieval surfaced. A ``StanceLabel`` is how the stance step judged a candidate relative to
the claim. A ``ClaimReport`` is the per-claim verdict the metrics layer consumes.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ClaimStatus(str, Enum):
    """Ground-truth status of a gold claim."""

    REVERSED = "reversed"  # the world later contradicted/superseded this claim
    STILL_TRUE = "still_true"  # control: remains supported by current consensus


class Stance(str, Enum):
    """How a candidate relates to the claim as asserted."""

    SUPPORTS = "supports"
    REFUTES = "refutes"
    NEUTRAL = "neutral"


class FailureBucket(str, Enum):
    """Why the harness failed to surface contradiction for a reversed claim.

    This taxonomy is the actual deliverable: it localizes *where* recall dies so
    the fix is targeted rather than "improve retrieval" in the abstract.
    """

    NONE = "none"  # not a failure (contradiction surfaced, or claim is a control)
    NOT_INDEXED = "not_indexed"  # contradiction not findable via either source/query
    RETRIEVED_NOT_RECOGNIZED = "retrieved_not_recognized"  # found but stance != refutes
    ENTITY_MISS = "entity_miss"  # query failed to map intervention/outcome
    CONDITION_MISMATCH = "condition_mismatch"  # refutation tested a different condition
    TIER_INVERSION = "tier_inversion"  # only low-tier support; high-tier refutation missed


class NormalizedClaim(BaseModel):
    """A claim with its conditions kept attached.

    Stripping conditions is the dominant source of false refutations (e.g. HCQ
    "reduces COVID mortality" is only false *in hospitalized patients*), so the
    population/comparator/outcome are first-class fields, not free text.
    """

    intervention: str
    outcome: str
    population: str | None = None
    comparator: str | None = None
    direction: Literal["increases", "decreases", "no_effect", "causes", "prevents"] | None = None

    def as_query_terms(self) -> list[str]:
        """Salient terms for building a literature query."""
        terms = [self.intervention, self.outcome]
        if self.population:
            terms.append(self.population)
        if self.comparator:
            terms.append(self.comparator)
        return [t for t in terms if t]


class GoldEntry(BaseModel):
    """One hand-authored ground-truth claim."""

    id: str
    claim_text: str = Field(..., description="The claim as originally asserted in the literature.")
    normalized: NormalizedClaim
    status: ClaimStatus
    answer_key: list[str] = Field(
        default_factory=list,
        description="PMIDs/DOIs of the landmark refuting study/SR/guideline. "
        "Empty for still_true controls.",
    )
    reversal_year: int | None = None
    notes: str | None = None


class GoldSet(BaseModel):
    """The full gold set loaded from YAML."""

    entries: list[GoldEntry]

    @property
    def reversed_entries(self) -> list[GoldEntry]:
        return [e for e in self.entries if e.status is ClaimStatus.REVERSED]

    @property
    def control_entries(self) -> list[GoldEntry]:
        return [e for e in self.entries if e.status is ClaimStatus.STILL_TRUE]


class Candidate(BaseModel):
    """One piece of retrieved evidence."""

    source: Literal["pubmed", "europepmc"]
    ext_id: str  # PMID for pubmed, source-prefixed id for europepmc
    doi: str | None = None
    title: str = ""
    abstract: str = ""
    pub_types: list[str] = Field(default_factory=list)  # e.g. "Meta-Analysis", "Guideline"
    year: int | None = None
    # link relationships (PubMed ELink), a secondary contradiction signal
    is_retraction_of: list[str] = Field(default_factory=list)
    retracted_by: list[str] = Field(default_factory=list)
    # retrieval provenance
    retrieved_by: list[str] = Field(default_factory=list)  # "keyword" | "semantic" | "links"

    @property
    def evidence_tier(self) -> float:
        """Coarse GRADE-flavored tier weight from publication types.

        This is intentionally crude. It exists only so the harness can detect
        tier inversion (low-tier support masking high-tier refutation). It is not a
        scoring instrument and must not grow into one here.
        """
        pt = {p.lower() for p in self.pub_types}
        # PubMed labels the withdrawing notice "Retraction Notice" (and, on older records,
        # "Retraction of Publication"). Match both so a retraction is never mistiered as a
        # generic article.
        if pt & {"retraction of publication", "retraction notice"}:
            return 0.95  # a retraction is the strongest refutation of the work it withdraws
        if pt & {"guideline", "practice guideline"}:
            return 1.0
        if "systematic review" in pt:
            return 0.9
        if "meta-analysis" in pt:
            return 0.85
        if pt & {"randomized controlled trial", "controlled clinical trial"}:
            return 0.8
        if pt & {"observational study", "comparative study", "cohort studies"}:
            return 0.5
        if pt & {"case reports", "case report"}:
            return 0.2
        return 0.4  # unknown / generic journal article


class StanceLabel(BaseModel):
    """Stance verdict for one candidate against one claim."""

    claim_id: str
    candidate_ext_id: str
    stance: Stance
    confidence: float = 0.0
    rationale: str = ""
    # Whether the candidate addresses the SAME condition (population/comparator) the
    # claim asserts. None when the backend can't judge it (e.g. the stub). False here
    # with stance != refutes is the signature of a condition-mismatch false negative.
    condition_match: bool | None = None


class ClaimReport(BaseModel):
    """Per-claim outcome consumed by the metrics layer."""

    claim_id: str
    status: ClaimStatus
    n_candidates: int
    # retrieval-stage outcome (independent of stance)
    answer_key_retrieved: bool  # was a gold answer-key doc in the retrieved set?
    answer_key_rank: int | None = None  # best rank of an answer-key doc (1-based)
    # stance-stage outcome
    refuting_found: bool  # any candidate judged REFUTES?
    answer_key_recognized: bool  # an answer-key doc judged REFUTES?
    top_refuting_tier: float = 0.0
    failure_bucket: FailureBucket = FailureBucket.NONE
    # for controls: did we wrongly flag contradiction?
    false_contradiction: bool = False


# The data filter, the project's end goal, scores whole PubMed papers, not just the
# hand-authored gold claims. These models carry that flow: a parsed paper, the claims
# extracted from it, and the per-claim and per-paper verdicts written to the flat table.


class Verdict(str, Enum):
    """Evidence-grounded truthfulness verdict for a claim or a paper."""

    SUPPORTED = "supported"  # evidence backs it, no credible refutation
    CONTESTED = "contested"  # evidence both supports and refutes it
    REFUTED = "refuted"  # higher-tier evidence contradicts it
    UNVERIFIED = "unverified"  # no usable evidence found (absence is not falsity)


class Action(str, Enum):
    """Recommended training-data action derived from the verdict."""

    KEEP = "keep"
    DOWNWEIGHT = "downweight"
    DROP = "drop"


class PaperRecord(BaseModel):
    """One PubMed paper parsed from MEDLINE XML, the unit the filter scores."""

    pmid: str
    title: str = ""
    abstract: str = ""
    pub_types: list[str] = Field(default_factory=list)
    mesh: list[str] = Field(default_factory=list)
    year: int | None = None
    journal: str | None = None
    doi: str | None = None
    # CommentsCorrectionsList grouped by RefType (e.g. "RetractionIn", "CommentIn",
    # "ErratumIn"), each mapping to the PMIDs of works that retracted, corrected, or
    # debated this paper. These are the paper's own pointers to its refutations.
    comments_corrections: dict[str, list[str]] = Field(default_factory=dict)


class ExtractedClaim(BaseModel):
    """One checkable claim lifted from a paper, conditions kept attached."""

    paper_pmid: str
    index: int
    claim_text: str
    normalized: NormalizedClaim

    @property
    def claim_id(self) -> str:
        return f"{self.paper_pmid}#c{self.index}"

    def as_gold_entry(self) -> GoldEntry:
        """Adapt to a GoldEntry so the existing stance backends classify it unchanged."""
        return GoldEntry(
            id=self.claim_id, claim_text=self.claim_text, normalized=self.normalized,
            status=ClaimStatus.REVERSED, answer_key=[],
        )


class ClaimVerdict(BaseModel):
    """Verdict for one extracted claim after weighing its evidence."""

    claim_id: str
    claim_text: str
    n_evidence: int
    n_refuting: int
    n_supporting: int
    top_refuting_tier: float
    verdict: Verdict
    score: float  # 0 (refuted) .. 1 (well supported)
    refuting_pmids: list[str] = Field(default_factory=list)
    supporting_pmids: list[str] = Field(default_factory=list)


class PaperVerdict(BaseModel):
    """Per-paper rollup written as one row of the flat table."""

    pmid: str
    title: str
    verdict: Verdict
    score: float
    action: Action
    n_claims: int
    n_refuted_claims: int
    top_refuting_tier: float
    refuting_pmids: list[str] = Field(default_factory=list)
    claim_verdicts: list[ClaimVerdict] = Field(default_factory=list)
    notes: str = ""
