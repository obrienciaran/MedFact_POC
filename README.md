# MedFact_POC — Evidence-Grounded Truth Filter for Medical Papers

## What this is

A proof of concept for a **truth-based data quality filter for medical training
data**. Point it at a corpus of PubMed medical papers in XML format and it produces a
flat table, one row per paper, judging whether the paper's claims hold up against
trusted medical evidence. Downstream training then keeps, down-weights, or drops each
paper based on that table.

The point is to filter on *truth*, not on surface features — and that is what sets it
apart from every existing kind of quality filter, because all of them judge a paper by
how it looks rather than whether it is right:

<img width="1501" height="776" alt="Screenshot 2026-06-21 at 03 45 19" src="https://github.com/user-attachments/assets/e221edca-404a-47de-abb0-a8157dade899" />


- **Rule-based filters** check form such as length, punctuation, repetition, language. They
  strip out gibberish and spam, but well-formed false claims pass through.
- **Quality classifiers** are trained to favour text that resembles a trusted reference
  like Wikipedia or textbooks. They reward resemblance to authority, not correctness.
  A wrong claim written in a journal's style also passes through.
- **Multi-dimensional rating** scores text across several style axes at once, such as
  readability, professionalism, educational value. More axes, but every one still
  measures how the text reads, not whether its claims are true.
- **LLM rating** asks a language model how good or educational a passage looks. Models
  tend to reward fluency and coherence, which is exactly what confident misinformation imitates best.

The shared blind spot is that each of these approahces answer "does this look trustworthy?"
while none answers "is this true?" 

This filter does not score appearance at all. It pulls out each claim and checks it
against the evidence, asking whether the research backs it up or contradicts it, and the
verdict is *discovered* from that evidence, not guessed from how the paper reads. What
exposes a wrong paper is the contradicting evidence itself.

This is a POC. It runs on a small curated slice and tests the one dependency the
whole proposed approach rests on. That is to check if the filter has the ability to *find* the evidence that contradicts a
wrong claim. We call this a "harness".
[Validation](#validation-does-the-search-find-the-evidence). If retrieval cannot find the
contradicting evidence, the filter would look reliable on famous topics, where the evidence is
easy to find, and be quietly wrong on rarer ones, where it is hard to find.

## What it produces

Running the filter writes `reports/filter.csv`, one row per paper:

| paper | truthful | ... |
|---|---|---|
| paper identifier | do its claims hold up against the evidence | confidence, contradicting studies, evidence tier, other metadata |

Every column is documented in [`reports/README.md`](reports/README.md). Downstream training
keeps, down-weights, or drops papers from that table. The evidence graph the filter can also
draw is an optional visual aid over the same data, not the product.

<img width="1174" height="388" alt="Screenshot 2026-06-21 at 03 48 03" src="https://github.com/user-attachments/assets/57568d0c-4dd7-47e9-a5db-6f6889f573a9" />

## How it works

The filter takes a directory of PubMed XML papers and processes **each paper on its own**. It
does not compare papers against each other; there is no pairwise cross-check. A paper is judged
only against external trusted evidence, never against the rest of your corpus, so the cost
grows linearly with the number of papers, not quadratically.

For one paper the steps are:

1. **Ingest** — read the PubMed XML into a structured record (claim text, publication type,
   MeSH terms, and any retraction/comment links).
2. **Extract claims** — an LLM lifts the paper's specific claims out of its text.
3. **Retrieve evidence** — for each claim, fetch a small set of candidate studies that bear on
   it (see below). This is the step the harness validates.
4. **Judge stance** — an LLM reads each candidate study's title and abstract and labels it as
   supporting, refuting, or neutral toward the claim.
5. **Score** — weigh those labels by evidence tier into a per-claim verdict, then take the
   paper's worst claim as the paper's verdict (see [How a paper is scored](#how-a-paper-is-scored)).

The result is one row per paper in `reports/filter.csv`.

### How evidence is found

The evidence is not in your corpus, and there is no local copy of the medical literature. **The
database search happens on PubMed's and Europe PMC's own servers, not here.** For each claim
the filter builds a few keyword/boolean queries (`transformation/query.py`) and sends them to
the PubMed E-utilities and Europe PMC REST search endpoints. Their search engines scan their
full databases and return only the matching study IDs, already ranked by relevance and capped
at a small limit (default 20 per query). The filter unions those matches with any dispute links
from the paper's own XML, then fetches each candidate's abstract and publication type. The
result is a small per-claim candidate pool, a few dozen studies, not the whole database.

The queries deliberately include contradiction-seeking and high-tier formulations. **Whether
they actually surface the disproving study is not assumed — it is the exact thing the harness
measures (retrieval recall), because a boolean query that misses the right terms is this
project's central failure mode.** When a query fails the harness tags it `entity_miss` (the
query missed the claim's terms) or `not_indexed` (no query or source returned the study). This
is **not** a vector search over all of PubMed; no such index exists here. Embeddings and cosine
similarity (`transformation/semantic.py`) appear only in the *harness*, where they re-rank an
already-fetched pool to measure how near the top the disproving study lands (recall@k).
Re-ranking cannot recover a study the API queries never returned.

### What it costs

The expense is dominated by network calls (PubMed/Europe PMC search and fetch) and LLM calls
(one extraction per paper, plus one stance judgement per candidate study per claim). Papers run
concurrently (`MEDFACT_FILTER_CONCURRENCY`, default 4). The whole pipeline also runs offline on
stub backends with no network and no LLM, for checking the plumbing before spending anything.

> **Status: this is a proof of concept, not a finished filter.** The goal is a truth-based data
> quality filter for medical training data. The code here demonstrates that the approach is
> viable on a small curated slice; it has not been tuned or validated at scale, and the query
> construction, retrieval, and scoring all need further refinement before production use. Treat
> every result as an early capability demonstration, not a reliable data filtering tool.

## Doesn't this exist already?

Two simpler ideas sound like they would do the same job. Neither does.

**Why not trust well-cited sources?** Filtering by reputation, i.e. keeping high-citation authors
and journals (the H-index and similar) judges who is speaking, not whether what they say is
true. A highly cited researcher in a top journal can still publish a claim that is later
overturned; the hormone-replacement-therapy belief was exactly that, and it carried a high
citation count the whole time. Reputation also has the blind spot where famous
work is easy to find, and obscure-but-correct work is not, so a reputation filter looks strong on
famous topics and fails on the rare ones.

**Why not just count refuted claims?** Counting refutations assumes the hard part is already
done. To count them you must first *find* the study that refutes each claim and confirm it
does, which is what this POC tests rather than takes for granted. You cannot count
refutations you cannot find.

## Where the language model fits

The language model has a deliberately bounded role. It does two jobs. It extracts each claim
from a paper's text, and it judges whether a retrieved study refutes or supports that claim.
It does not run the search, decide which papers are kept or dropped, or score a paper on its
own, those follow from the retrieved evidence and its tier.

## How a paper is scored

Scoring is mechanical and evidence-driven, not a model opinion. It runs per claim, then rolls
up to the paper. The thresholds all live in `transformation/scoring.py`.

1. **Weigh each study.** Every retrieved study carries an evidence tier set by its publication
   type (guideline 1.0, retraction 0.95, systematic review 0.9, meta-analysis 0.85, randomised
   trial 0.8, observational study 0.5, case report 0.2, anything else 0.4). A study's pull on
   the claim is that tier multiplied by the model's stance confidence, so a high-tier study the
   model is sure about moves the result far more than a weak or uncertain one.
2. **Score the claim.** The claim takes its single strongest refuting pull and its single
   strongest supporting pull and turns them into a 0 to 1 truthfulness score: it starts at 0.5,
   supporting evidence raises it, and refuting evidence lowers it (refutation weighs about
   twice as much). The claim's verdict is `refuted` (a strong high-tier refutation),
   `contested` (evidence both ways, or only a weak refutation), `supported` (support only), or
   `unverified` (no usable evidence found).
3. **Roll up to the paper.** A paper is judged by its most damning claim. Its score is the
   lowest of its claim scores and its verdict is the worst of its claim verdicts. The verdict
   maps to an action: `refuted` drops the paper, `contested` down-weights it, and `supported`
   or `unverified` keeps it. Unverified is kept on purpose, because a missing refutation is not
   proof a claim is false.

The continuous score sits next to the verdict in the table, so a curator can set a stricter or
looser cutoff than the default actions.

## Validation: can the search find the evidence?

The filter is only as good as its search. If the search cannot find the study that contradicts
a wrong claim, the filter cannot catch that claim. Everything rests on this one step, so a
**harness** (`medfact-run`) tests it on its own.

The test is simple. We take a set of claims the field already knows were wrong, where the study
that disproved each one is written down in advance. We run the filter's search over those
claims and check how often it brings the known disproving study back. This is the proof of
concept: a small, hand-curated check that the search works on cases where we already know the
answer.

> This is an experimental proof-of-concept project. It has not been run on a wider dataset.

The search runs in two steps, and the harness scores each step separately so that when
something goes wrong you can tell which step broke:

1. **Fetch** — did the search pull the disproving study back at all?
2. **Judge** — once fetched, did the model recognise it as refuting the claim?

Each score is a *recall*: of the studies that should have been found, the fraction that
actually were. Every claim counts as a pass or a fail, and the recall is the percentage that
pass. The harness reports four:

- **Retrieval recall** scores the fetch step: of the disproving studies we know exist, the
  fraction the search pulled back. It is checked against the written answer key, so it does not
  depend on the model at all. This is the headline number.
- **Stance recall** scores the judge step: of the disproving studies that were fetched, the
  fraction the model then labelled as refuting. Keeping the two apart pinpoints any failure —
  a low retrieval recall means the search missed the study, while a high retrieval recall but
  low stance recall means the search found it and the model failed to recognise it.
- **Recall@k** is retrieval recall counted only within the top k results by relevance (k = 1,
  5, 10, 20). It shows whether the disproving study ranked near the top or was buried deep in
  the pool.
- **False-contradiction rate** is a safety check on the control claims that are still true: the
  fraction the search wrongly flagged as refuted. A filter that flags everything is useless, so
  lower is better.

When a claim fails, the harness also tags *why*, so a miss can be traced to a root cause:

- `not_indexed`: the disproving study was never fetched by any query or source.
- `entity_miss`: the query missed the claim's terms, so almost nothing came back.
- `retrieved_not_recognized`: the study was fetched, but the model did not call it refuting.
- `condition_mismatch`: the study was fetched, but it tested a different population or setting.
- `tier_inversion`: only weak evidence was recognised while a stronger refutation was missed.

The percentage summaries and the failure-tag tally are printed at the end of the run and saved
to a markdown report (`reports/recall-<timestamp>.md`); the per-claim detail behind them, one
row per claim with its pass/fail flags and failure tag, goes to a CSV
(`reports/recall-<timestamp>.csv`). These files only validate the search and are separate from
the filter's own per-paper table (`reports/filter.csv`). If retrieval recall is poor, a
claim-scoring filter is confident about well-known cases and silently wrong about everything
that is under-indexed.

## Setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
# optional backends: real embeddings, and one LLM provider for claim extraction + stance
uv pip install -e ".[dev,embed,gemini]"   # or anthropic / openai
```

Everything runs end-to-end with **no optional deps and no API key** using built-in *stub*
backends. A stub is a placeholder that fakes a step's output deterministically with **no
LLM call and no network**, so you can validate the plumbing before spending anything. The
real run uses an LLM (see below). Stub output is a placeholder, not a real result.

Set credentials/keys via env when using real backends:

```bash
export NCBI_API_KEY=...        # optional, raises PubMed rate limit
export NCBI_EMAIL=you@org.com  # polite identification for E-utilities
export GEMINI_API_KEY=...      # or ANTHROPIC_API_KEY / OPENAI_API_KEY
```

## Run the filter

```bash
# Offline, synthetic (stub) backends — no key needed:
medfact-filter --input path/to/pubmed_xml_dir

# Real backends (claim extraction + stance via an LLM, live evidence retrieval):
MEDFACT_LLM_PROVIDER=gemini MEDFACT_EXTRACT_BACKEND=llm \
MEDFACT_STANCE_BACKEND=llm MEDFACT_RETRIEVER=live \
  medfact-filter --input path/to/pubmed_xml_dir
```

**Where to find the results.** The filter writes exactly two files, and these are what you
open to view a run:

- **`reports/filter.csv`** — the per-paper table (pmid, verdict, score, action, metadata),
  one row per paper. Every column is documented in [`reports/README.md`](reports/README.md).
- **`reports/filter.html`** — the interactive graph of the same run. Open it in a browser.

Two other HTML files in `reports/` are *not* a filter run and are easy to mistake for one:
`reports/graph_demo.html` is a prebuilt static example shipped with the repo, and
`reports/graph.html` is the **harness** visualization (written by `medfact-graph`), not the
filter. Only `reports/filter.html` and `reports/filter.csv` come from `medfact-filter`.

The input is PubMed/MEDLINE XML because that format carries the `CommentsCorrectionsList`
retraction and comment links, the publication types (evidence tier), and MeSH terms the filter
relies on.

The ingester accepts any PubMed XML it can read. It prints an error and skips a file it cannot
read or that is not a PubMed article set, and it prints a highlight for a paper that has no
`CommentsCorrectionsList`, which is the offline truthfulness signal. During live retrieval the
filter also queries Europe PMC as a second evidence source. That is retrieval, not input.

## Run the harness

```bash
medfact-build-cache      # fetch candidate evidence for the gold set into DuckDB
medfact-run              # run the measurement, write report to ./reports/
medfact-graph            # render the evidence graph to reports/graph.html
pytest                   # unit tests (network tests are opt-in: pytest -m live)
```

## Visualization (optional)

The graph is a secondary aid. `medfact-graph` renders a harness run to a
self-contained, interactive page (`reports/graph.html`); the page header explains how to read
it (claims as boxes, retrieved studies as dots, line colour for stance), and it adds a
summary, a legend, edge-type filters, a "recall gaps only" view, and hover/click focus. The
filter draws the same kind of graph for its own results at `reports/filter.html`. Run
`medfact-run` (or `medfact-filter`) first so there is data to draw.

## Roadmap

Not built yet: GRADE-based evidence weighting, feeding the verdicts into training weights,
UMLS concept grounding, approximate-nearest-neighbour indexing for faster search at scale, and
a non-LLM stance classifier.
