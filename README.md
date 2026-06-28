# 🏥 MedScreen — Evidence-Grounded Data Quality Filter for Medical Papers

## What this is

EXPERIMENTAL

A proof of concept for an evidence-based data quality filter for medical training data. Point it
at a corpus of PubMed papers (XML) and it produces a flat table, one row per paper, judging
whether each paper's claims hold up against trusted medical evidence. Downstream training can then
keep, down-weight, or drop papers based on that table.

The point is to filter on truth, not surface features. Rule-based filters, quality classifiers,
and LLM raters all judge a paper by how it reads (fluency, formatting, resemblance to a trusted
reference), which is exactly what confident misinformation imitates best. This filter instead
pulls out each claim and checks it against the evidence, so the verdict is discovered, not
guessed.

It is a POC, so it tests the one dependency the approach rests on; can retrieval find the evidence
that contradicts a wrong claim?

## ➡️ What it produces

Running the filter writes `reports/filter.csv`, one row per paper:

| paper | truthful | ... |
|---|---|---|
| paper identifier | do its claims hold up against the evidence | confidence, contradicting studies, evidence tier, other metadata |

Every column is documented in [`reports/README.md`](reports/README.md). The filter can also draw
an optional evidence graph over the same data (`reports/filter.html`), but the table is the
product.

<img width="1199" height="279" alt="Filter output table" src="https://github.com/user-attachments/assets/8ec9ed89-82d9-4dc7-a768-a57945c77b06" />

## 🛠️ How it works

The filter processes each paper on its own, judged only against external trusted evidence, so cost
grows linearly with the number of papers. For one paper:

1. Ingest: read the PubMed XML into a structured record (claim text, publication type, MeSH
   terms, retraction/comment links).
2. Extract claims: an LLM lifts the paper's specific claims out of its text.
3. Retrieve evidence: for each claim, fetch a small set of candidate studies that bear on it.
   This is the step the POC validates.
4. Judge stance: an LLM reads each candidate's title and abstract and labels it supporting,
   refuting, or neutral toward the claim.
5. Score: weigh those labels by evidence tier into a per-claim verdict, then take the paper's
   worst claim as its verdict (see below).

The whole pipeline also runs offline on stub backends (no network, no LLM) so you can check the
plumbing before spending anything. Stub output is a placeholder, not a real result.

The LLM's role is deliberately bounded to those two steps (extract and judge stance). It does not
run the search, score a paper, or decide which papers are kept.

### 🔎 How evidence is found

Evidence is not stored locally, and this is not a vector search over all of PubMed. For each claim
the filter builds a few keyword/boolean queries (`transformation/query.py`) and sends them to the
PubMed E-utilities and Europe PMC search endpoints, which return a capped, relevance-ranked list
of study IDs. It unions those with any dispute links from the paper's own XML, then fetches each
candidate's abstract and publication type.

Retrieval is tuned so a known refutation actually comes back:

- Several searches per claim: a broad one (intervention + outcome), one limited to strong study
  types (meta-analyses, reviews, RCTs, guidelines), and one looking for contradiction (`risk`,
  `harm`, `no benefit`).
- Two sources queried independently.
- Optional cache: set `MEDSCREEN_QUERY_CACHE` to a file path (DuckDB) to fetch repeated searches
  once across a corpus. Unset to always search live.

Embeddings (`transformation/semantic.py`) are used only to re-rank an already-fetched pool during
validation (`recall@k`); they cannot recover a study the queries never returned. Whether the
queries succeed is the central failure mode, and exactly what the validation measures.

> Status: proof of concept. Tested on 10 PubMed papers with Gemini 2.5 Flash Lite (6 kept, 4
> downweighted, 0 dropped; all 10 got a real verdict). Not yet run on a large or varied dataset;
> query construction, retrieval, and scoring all need refinement before production use.

## 📄 How a paper is scored

Scoring is mechanical, not a model opinion. Thresholds live in `transformation/scoring.py`.

1. Weigh each study by its publication-type evidence tier (guideline 1.0, retraction 0.95,
   systematic review 0.9, meta-analysis 0.85, RCT 0.8, observational 0.5, case report 0.2, else
   0.4). Its pull on a claim is that tier × the model's stance confidence.
2. Score the claim. Starting at 0.5, the strongest supporting pull raises the score and the
   strongest refuting pull lowers it (refutation weighs about twice as much). Verdict is
   `refuted`, `contested`, `supported`, or `unverified` (no usable evidence).
3. Roll up to the paper by its most damning claim: lowest score, worst verdict. `refuted` drops
   the paper, `contested` down-weights it, `supported` and `unverified` keep it. Unverified is
   kept on purpose, since a missing refutation is not proof a claim is false.

## ❓ Validation study: can the search find the evidence?

The filter is only as good as its search. A separate test (`medscreen-run`) checks that one step
using claims the field already knows were wrong, where the disproving study is recorded in
advance. It runs the filter's search and checks how often it finds the known study, scored in two
stages so a failure traces to the right one:

- Retrieval recall: of the disproving studies that exist, the fraction the search pulled back.
  Model-independent. The headline number.
- Stance recall: of those fetched, the fraction the model labelled refuting.
- Recall@k: retrieval recall within the top k results (k = 1, 5, 10, 20).
- False-contradiction rate: fraction of still-true controls wrongly flagged refuted (lower is
  better).

Each miss is tagged with a root cause (`not_indexed`, `entity_miss`, `retrieved_not_recognized`,
`condition_mismatch`, `tier_inversion`). Results print at the end of a run and save to
`reports/recall-<timestamp>.md` and `.csv`.

> Status: real measurement, two known gaps. A real run (`pritamdeka/S-PubMedBert-MS-MARCO`
> embeddings, Gemini 2.5 Flash Lite stance) on the 10-reversal/8-control gold slice scored 80%
> retrieval recall and a 25% false-contradiction rate (`reports/recall-20260621-194531.md`).
> Stance recall is 100% conditional on retrieval, so retrieval is the sole bottleneck. Both misses
> are `not_indexed` (the 1984 Marshall & Warren ulcer paper and the NICE-SUGAR trial), needing
> MeSH-based queries or alias expansion. A stronger stance model than Flash Lite would improve the
> false-contradiction rate.

## 🤔 Doesn't this exist already?

Why not trust well-cited sources, like a h-index? Reputation judges who is speaking, not whether they are right.
The belief that hormone replacement therapy prevents coronary heart disease was highly cited the
entire time it was wrong, until the 2002 Women's Health Initiative trial found the opposite.

Why not just count refuted claims? That assumes the hard part — finding and confirming the
refuting study — is already done. This POC tests that step instead of taking it for granted.

## ⚙️ Setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
# optional backends: real embeddings + one LLM provider for extraction + stance
uv pip install -e ".[dev,embed,gemini]"   # or anthropic / openai
```

Everything runs with no optional deps and no API key using stub backends. Set credentials via env
for real backends:

```bash
export NCBI_API_KEY=...        # optional, raises PubMed rate limit
export NCBI_EMAIL=you@org.com  # polite identification for E-utilities
export GEMINI_API_KEY=...      # or ANTHROPIC_API_KEY / OPENAI_API_KEY
```

## 🏃 Run the filter

```bash
# Offline stub backends — no key needed:
medscreen-filter --input path/to/pubmed_xml_dir

# Real backends (LLM extraction + stance, live retrieval):
MEDSCREEN_LLM_PROVIDER=gemini MEDSCREEN_EXTRACT_BACKEND=llm \
MEDSCREEN_STANCE_BACKEND=llm MEDSCREEN_RETRIEVER=live \
  medscreen-filter --input path/to/pubmed_xml_dir
```

Writes `reports/filter.csv` (per-paper table, documented in
[`reports/README.md`](reports/README.md)) and `reports/filter.html` (interactive graph). The other
HTML files in `reports/` are not a filter run — `graph_demo.html` is a shipped example and
`graph.html` is the validation visualization.

Input must be PubMed/MEDLINE XML, since it carries the `CommentsCorrectionsList` links, publication
types, and MeSH terms the filter relies on. The ingester reads any PubMed XML it can and skips
(with an error) any it cannot.

## 🏃 Run the validation

```bash
medscreen-build-cache      # fetch candidate evidence for the gold set into DuckDB
medscreen-run --use-cache  # score against the cache, write report to ./reports/
medscreen-graph            # render the evidence graph to reports/graph.html
pytest                   # unit tests (network tests are opt-in: pytest -m live)
```

`medscreen-run` defaults to stub backends (offline, no key). For a real measurement:

```bash
MEDSCREEN_EMBED_BACKEND=sbert MEDSCREEN_STANCE_BACKEND=llm MEDSCREEN_LLM_PROVIDER=gemini \
  medscreen-run --use-cache
```

`MEDSCREEN_EMBED_BACKEND=sbert` needs the `embed` extra; without it the numbers come from stubs and
are not a real measurement.

## 🌀 Visualization (optional)

A secondary aid, not the product. `medscreen-graph` renders a validation run to a self-contained
interactive page (`reports/graph.html`) — claims as boxes, retrieved studies as dots, line colour
for stance, with a legend, edge filters, a "recall gaps only" view, and hover/click focus. The
filter draws the same graph for its own results at `reports/filter.html`. Run `medscreen-run` (or
`medscreen-filter`) first so there is data to draw.

<img width="1499" height="768" alt="Evidence graph" src="https://github.com/user-attachments/assets/e0018172-eb28-4b29-b49c-bbc18d825967" />

## 🛣️ Roadmap

Not built yet: GRADE-based evidence weighting, feeding verdicts into training weights, UMLS
concept grounding, approximate-nearest-neighbour indexing, and a non-LLM stance classifier.
