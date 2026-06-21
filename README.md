# 🏥 MedFact — Evidence-Grounded Data Quality Filter for Medical Papers

## What this is

A proof of concept for a **LLM powered, evidence-based, data quality filter for medical training data**. Point it
at a corpus of PubMed papers in XML format and it produces a flat table, one row per paper,
judging whether the paper's claims hold up against trusted medical evidence.

The point is to filter on truth, not on surface features. Existing approaches judge a paper by
how it looks, not whether it is right:

- **Rule-based filters** catch gibberish and spam, but a well-formed false claim passes through.
- **Quality classifiers** reward resemblance to a trusted reference like Wikipedia, not correctness.
- **Multi-dimensional rating** adds more style axes (readability, professionalism), but still
  measures how the text reads, not whether it is true.
- **LLM rating** rewards fluency and coherence, exactly what confident misinformation imitates best.

None of these ask "is this true?" This filter does. It pulls out each claim and checks it against
the evidence, so the verdict is discovered, not guessed from how the paper reads.

This is a POC. It tests the one dependency the whole approach rests on. Can retrieval *find* the
evidence that contradicts a wrong claim?

## ➡️ What it produces

Running the filter writes `reports/filter.csv`, one row per paper:

| paper | truthful | ... |
|---|---|---|
| paper identifier | do its claims hold up against the evidence | confidence, contradicting studies, evidence tier, other metadata |

Every column is documented in [`reports/README.md`](reports/README.md). Downstream training then
keeps, down-weights, or drops papers based on that table. The filter can also draw an evidence
graph as an optional visual aid over the same data (see [Visualization](#-visualization-optional)),
but the table is the product.

<img width="1174" height="388" alt="Screenshot 2026-06-21 at 03 48 03" src="https://github.com/user-attachments/assets/57568d0c-4dd7-47e9-a5db-6f6889f573a9" />

## 🛠️ How it works

The filter processes **each paper on its own**, never comparing papers against each other. A
paper is judged only against external trusted evidence, so the cost grows linearly with the
number of papers, not quadratically.

For one paper the steps are:

1. **Ingest** — read the PubMed XML into a structured record (claim text, publication type,
   MeSH terms, and any retraction/comment links).
2. **Extract claims** — an LLM lifts the paper's specific claims out of its text.
3. **Retrieve evidence** — for each claim, fetch a small set of candidate studies that bear on
   it (see below). This is the main step this POC validates.
4. **Judge stance** — an LLM reads each candidate study's title and abstract and labels it as
   supporting, refuting, or neutral toward the claim.
5. **Score** — weigh those labels by evidence tier into a per-claim verdict, then take the
   paper's worst claim as the paper's verdict (see [How a paper is scored](#-how-a-paper-is-scored)).

### 🔎 How evidence is found

The evidence is not stored locally; search happens on PubMed's and Europe PMC's own servers, not
here. For each claim the filter builds a few keyword/boolean queries (`transformation/query.py`)
and sends them to the PubMed E-utilities and Europe PMC REST search endpoints, which return a
capped, relevance-ranked list of matching study IDs (default 20 per query). The filter unions
those matches with any dispute links from the paper's own XML, then fetches each candidate's
abstract and publication type, giving a small per-claim candidate pool.

The queries deliberately seek out contradicting, high-tier evidence, but whether they succeed is
not assumed. That is the exact thing this POC measures (retrieval recall), since a query that
misses the right terms is the project's central failure mode. A miss is tagged `entity_miss` (the
query missed the claim's terms) or `not_indexed` (no query or source returned the study). This is
**not** a vector search over all of PubMed. Embeddings and cosine similarity are used
(`transformation/semantic.py`) only to re-rank an already-fetched pool, in the validation test, to
measure how near the top the disproving study lands (`recall@k`). Re-ranking cannot recover a
study the queries never returned.

### 🧱 Retrieval engine

Retrieval is plain keyword search, tuned so a known refutation actually comes back, and built to
stay cheap as the corpus grows. Three pieces matter (`transformation/query.py`,
`scraping/pubmed.py`, `scraping/querycache.py`):

- **Query ladder, not one rigid query.** Each claim becomes three PubMed rungs whose results are
  unioned: a loose `core` rung (intervention + outcome), a `high-tier` rung that filters to the
  publication types that overturn consensus (meta-analysis, systematic review, randomised trial,
  guideline), and a contradiction-seeking rung (`risk`, `harm`, `no benefit`, ...). Europe PMC
  mirrors the core and high-tier rungs. Terms are passed **unquoted** and stripped of parenthetical
  annotations first, because quoting a long phrase forces an exact-string match that descriptive
  claim text almost never satisfies. A per-rung contribution check on the gold slice set the rung
  count: `core` and `high-tier` each recover every answer key a wider set did, so extra rungs were
  dropped to keep the query count (and retrieval time) per claim low.
- **Batched fetch.** A claim's unioned PMIDs are fetched in batches (`efetch`, 150 per request) so a
  large pool cannot overflow the request URL (NCBI returns HTTP 414 for long URLs).
- **Source resilience.** The two evidence sources are queried independently, so if one (say Europe
  PMC) is temporarily down, the filter falls back to the other rather than failing the whole paper
  to `unverified`.
- **DuckDB query cache (opt-in).** Across a large corpus the same claim recurs constantly and
  normalizes to the same search. With `MEDFACT_QUERY_CACHE` set to a file path (or `1` for the
  default `data/cache/query_cache.duckdb`), retrieval is **cache-first**: it looks in DuckDB before
  every call and only touches the PubMed or Europe PMC API on a miss, writing the result back so the
  next paper that needs it is served from disk. Two things are cached, so a study seen once is
  neither searched nor fetched again: a `query_results` table maps `(source, query, page_size)` to
  its hits, and a `records` table maps a study id to its fetched title/abstract/publication types.
  A distinct query and a distinct study each hit the network only once for the whole corpus. The
  cache is thread-safe (the filter and harness fan papers out across threads) and fails open: if the
  file cannot be opened, retrieval just runs live instead of aborting. Leaving the variable unset
  preserves live-every-time behaviour. This is separate from the validation tool's own DuckDB store
  (`store.py`), which caches candidates, embeddings, and stance for repeatable offline
  `medfact-run --use-cache` scoring.

### 💵 What it costs

The expense is dominated by network calls (PubMed/Europe PMC search and fetch) and LLM calls
(one extraction per paper, plus one stance judgement per candidate study per claim). Papers run
concurrently (`MEDFACT_FILTER_CONCURRENCY`, default 4), and `MEDFACT_QUERY_CACHE` (see
[Retrieval engine](#-retrieval-engine)) collapses repeated searches across a large corpus. The
whole pipeline also runs offline on stub backends with no network and no LLM, for checking the
plumbing before spending anything.

> **Status: proof of concept, not a finished tool.** It was tested on a small batch of 10 PubMed
> papers using Gemini 2.5 Flash Lite: 6 papers kept, 4 downweighted, 0 dropped
> (`reports/filter.csv`), with all 10 receiving a real verdict (no fall-back to `unverified`).
> It has **not** been run on a large or varied dataset, and query construction, retrieval, and
> scoring all need further refinement before production use.

## 🤔 Doesn't this exist already?

Two simpler ideas sound like they would do the same job. Neither does.

**Why not trust well-cited sources?** Reputation (citation count, H-index, journal prestige)
judges who is speaking, not whether what they say is true. The belief that hormone replacement
therapy prevents coronary heart disease in healthy postmenopausal women was highly cited the
entire time it was wrong: it rested on observational data, and stayed mainstream until the 2002
Women's Health Initiative randomized trial found the opposite, an increased risk of coronary
heart disease. Reputation has the same blind spot as fame generally; well-known work is easy to
check, obscure-but-correct work is not.

**Why not just count refuted claims?** That assumes the hard part, finding the study that refutes
each claim and confirming it does, is already done. This POC tests that step rather than taking
it for granted; you cannot count refutations you cannot find.

## 🤖 Where the language model fits

The language model has a deliberately bounded role. It does two jobs. It extracts each claim
from a paper's text, and it judges whether a retrieved study refutes or supports that claim. It
does not run the search, decide which papers are kept or dropped, or score a paper on its own.
Those follow from the retrieved evidence and its tier.

## 📄 How a paper is scored

Scoring is mechanical and evidence-driven, not a model opinion. It runs per claim, then rolls up
to the paper. The thresholds all live in `transformation/scoring.py`.

1. **Weigh each study.** Every retrieved study carries an evidence tier set by its publication
   type (guideline 1.0, retraction 0.95, systematic review 0.9, meta-analysis 0.85, randomised
   trial 0.8, observational study 0.5, case report 0.2, anything else 0.4). A study's pull on
   the claim is that tier multiplied by the model's stance confidence — a 0 to 1 number the
   model reports alongside its supports/refutes/neutral label, for how sure it is in that
   judgement — so a high-tier study the model is sure about moves the result far more than a
   weak or uncertain one.
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

## ❓ Validation: can the search find the evidence?

The filter is only as good as its search. If it cannot find the study that contradicts a wrong
claim, it cannot catch that claim. A separate validation test (`medfact-run`) checks this one
step on its own, using claims the field already knows were wrong, where the disproving study is
recorded in advance. It runs the filter's search over those claims and checks how often it finds
the known study — a hand-curated check on cases where the answer is already known.

The search runs in two steps, scored separately so a failure can be traced to the right one:

1. **Fetch** — did the search pull the disproving study back at all?
2. **Judge** — once fetched, did the model recognise it as refuting the claim?

Each score is a *recall*: the fraction of claims that passed out of those that should have. The
test reports four:

- **Retrieval recall** — of the disproving studies that exist, the fraction the search pulled
  back. Checked against the written answer key, so it does not depend on the model. The headline
  number.
- **Stance recall** — of the disproving studies that were fetched, the fraction the model then
  labelled as refuting. Splitting fetch from judge pinpoints which step failed.
- **Recall@k** — retrieval recall counted only within the top k results by relevance (k = 1, 5,
  10, 20), showing whether the disproving study ranked near the top or was buried.
- **False-contradiction rate** — fraction of still-true control claims wrongly flagged as
  refuted. Lower is better.

A failed claim is also tagged with why, so a miss traces to a root cause:

- `not_indexed` — never fetched by any query or source.
- `entity_miss` — the query missed the claim's terms.
- `retrieved_not_recognized` — fetched, but the model did not call it refuting.
- `condition_mismatch` — fetched, but it tested a different population or setting.
- `tier_inversion` — only weak evidence was recognised while a stronger refutation was missed.

Results print at the end of the run and save to a markdown report
(`reports/recall-<timestamp>.md`) plus a per-claim CSV (`reports/recall-<timestamp>.csv`). These
validate the search only, separate from the filter's own per-paper table (`reports/filter.csv`).

> **Status: real measurement, retrieval still has two known gaps.** A real run
> (`pritamdeka/S-PubMedBert-MS-MARCO` embeddings, Gemini 2.5 Flash Lite stance) against the
> 10-reversal/8-control gold slice scored **80% retrieval recall** and a **25%
> false-contradiction rate** (`reports/recall-20260621-194531.md`). When the disproving study
> reaches the pool the model recognises it as refuting every time (stance recall is 100%
> conditional on retrieval), so retrieval is the sole remaining bottleneck. The two misses are
> `not_indexed`: the 1984 Marshall & Warren ulcer paper and the NICE-SUGAR glycaemic-control
> trial, both of which need MeSH-based queries or alias expansion to surface. An earlier run
> scored 40% recall and a 62% false-contradiction rate; both numbers improved once query
> construction stopped forcing exact-phrase matches and switched to a small loose-to-targeted
> query ladder (`transformation/query.py`); retrieval recall held at 80% after the ladder was
> trimmed to its three highest-yield rungs. Retrieval recall is deterministic; the
> false-contradiction rate varies run to run (seen between 12% and 25%) because the stance judge
> is mildly non-deterministic on one or two borderline controls.

## ⚙️ Setup

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

## 🏃 Run the filter

```bash
# Offline, synthetic (stub) backends — no key needed:
medfact-filter --input path/to/pubmed_xml_dir

# Real backends (claim extraction + stance via an LLM, live evidence retrieval):
MEDFACT_LLM_PROVIDER=gemini MEDFACT_EXTRACT_BACKEND=llm \
MEDFACT_STANCE_BACKEND=llm MEDFACT_RETRIEVER=live \
  medfact-filter --input path/to/pubmed_xml_dir
```

**Where to find the results.** The filter writes exactly two files:

- **`reports/filter.csv`** — the per-paper table (pmid, verdict, score, action, metadata),
  one row per paper. Every column is documented in [`reports/README.md`](reports/README.md).
- **`reports/filter.html`** — the interactive graph of the same run. Open it in a browser.

Two other HTML files in `reports/` are *not* a filter run: `reports/graph_demo.html` is a static
example shipped with the repo, and `reports/graph.html` is the **validation** visualization
(written by `medfact-graph`). Only `reports/filter.html` and `reports/filter.csv` come from
`medfact-filter`.

The input is PubMed/MEDLINE XML because it carries the `CommentsCorrectionsList` retraction and
comment links, publication types (evidence tier), and MeSH terms the filter relies on. The
ingester accepts any PubMed XML it can read, skipping (with an error) any file it cannot, and
flags a paper with no `CommentsCorrectionsList` as a highlight, since that absence is itself the
offline truthfulness signal. During live retrieval the filter also queries Europe PMC as a
second evidence source.

## 🏃 Run the validation

```bash
medfact-build-cache      # fetch candidate evidence for the gold set into DuckDB
medfact-run --use-cache  # read that cache instead of re-fetching live, write report to ./reports/
medfact-graph            # render the evidence graph to reports/graph.html
pytest                   # unit tests (network tests are opt-in: pytest -m live)
```

`medfact-build-cache` always queries the live PubMed/Europe PMC APIs (no API key needed for
retrieval itself). `medfact-run` can also be run with no `--use-cache` flag, which re-fetches
the same evidence live instead of reading the cache; that works too, just slower.

By default `medfact-run` scores with stub embeddings and a stub (lexical) stance check, so it
runs offline with no API key. For a real measurement, set real backends the same way as the
filter:

```bash
MEDFACT_EMBED_BACKEND=sbert MEDFACT_STANCE_BACKEND=llm MEDFACT_LLM_PROVIDER=gemini \
  medfact-run --use-cache
```

`MEDFACT_EMBED_BACKEND=sbert` needs the `embed` extra installed (see Setup); without it the
recall numbers come from the stub backends and are not a real measurement.

## 🌀 Visualization (optional)

The graph is a secondary aid, not the product. `medfact-graph` renders a validation run to a
self-contained, interactive page (`reports/graph.html`); its header explains how to read it
(claims as boxes, retrieved studies as dots, line colour for stance), with a summary, legend,
edge-type filters, a "recall gaps only" view, and hover/click focus. The filter draws the same
kind of graph for its own results at `reports/filter.html`. Run `medfact-run` (or
`medfact-filter`) first so there is data to draw.

<img width="1501" height="776" alt="Screenshot 2026-06-21 at 03 45 19" src="https://github.com/user-attachments/assets/e221edca-404a-47de-abb0-a8157dade899" />

## 🛣️ Roadmap

Not built yet: GRADE-based evidence weighting, feeding the verdicts into training weights,
UMLS concept grounding, approximate-nearest-neighbour indexing for faster search at scale, and
a non-LLM stance classifier.
