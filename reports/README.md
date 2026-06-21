# `reports/` — generated output

This folder holds the artifacts the tools write. Nothing here is hand-edited.

- `filter.csv` / `filter.html` — the **filter** output (`medfact-filter`): one row per paper
  with its truthfulness verdict, plus the same data as an interactive graph.
- `recall-<timestamp>.md` / `recall-<timestamp>.csv` — the **harness** output
  (`medfact-run`): how often the search found the known disproving evidence. These are
  validation metrics, not part of `filter.csv`; see the main `README.md`.
- `graph.html` / `graph_demo.html` — the evidence graph for the harness run / a seeded demo.

## `filter.csv` columns

One row per paper. The first two columns identify the paper, `verdict` and `action` are what
a downstream training pipeline acts on, and the rest are the evidence behind the verdict so a
curator can audit or override it.

| column | meaning |
|---|---|
| `pmid` | PubMed identifier of the paper. |
| `title` | Article title. |
| `verdict` | Evidence-grounded truthfulness verdict (see below). |
| `score` | Truthfulness score from `0.000` to `1.000`: `1` = well supported, `0` = refuted. |
| `action` | Recommended training-data action: `keep`, `downweight`, or `drop`. |
| `n_claims` | How many checkable claims were extracted from the paper. |
| `n_refuted_claims` | How many of those claims were judged `refuted`. |
| `top_refuting_tier` | Evidence strength (`0.00`–`1.00`) of the strongest study refuting any claim; `0.00` if nothing refuted it (see the tier table). |
| `refuting_pmids` | PMIDs of the works that refuted the paper's claims, separated by `;`; empty if none. |
| `notes` | Free-text notes, e.g. `no claims extracted`. |

### `verdict`

The verdict is grounded in the retrieved evidence and its strength, not in the model's
free-standing opinion. A paper is judged as harshly as its **most damning claim**, so one
confidently wrong central claim is enough to refute the paper.

| verdict | meaning |
|---|---|
| `supported` | Evidence backs the claims and nothing credible refutes them. |
| `contested` | Evidence both supports and refutes, or refutation exists but is not strong enough to be decisive. |
| `refuted` | Higher-tier evidence contradicts a claim. |
| `unverified` | No usable evidence was found. Absence of refutation is **not** evidence of falsity. |

### `action`

`action` is the default policy derived from `verdict`. The continuous `score` is also written
so you can set your own threshold instead of relying on these defaults.

| verdict | default action |
|---|---|
| `refuted` | `drop` |
| `contested` | `downweight` |
| `supported` | `keep` |
| `unverified` | `keep` (a missing refutation is not falsity) |

### `score`

`score` is continuous truthfulness, not a probability. Per claim it is
`0.5 + 0.4 × support_strength − 0.8 × refute_strength`, clamped to `0..1`, where each strength
is the stance confidence weighted by the evidence tier of the study making that point. The
paper's score is the **minimum** across its claims (its weakest, most-refuted claim).

### `top_refuting_tier` and the evidence tiers

The tier is a coarse, GRADE-flavoured weight read from a study's publication type. It exists
so a high-tier refutation outweighs low-tier support; it is not a precise scoring instrument.

| tier | publication type |
|---|---|
| `1.00` | Guideline / practice guideline |
| `0.95` | Retraction of publication |
| `0.90` | Systematic review |
| `0.85` | Meta-analysis |
| `0.80` | Randomized / controlled clinical trial |
| `0.50` | Observational / cohort / comparative study |
| `0.40` | Unknown or generic journal article |
| `0.20` | Case report |
