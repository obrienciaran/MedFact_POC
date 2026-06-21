# CLAUDE.md

## Project Context

### What this project is

A truth based data quality filter. It is used for curating high quality PubMed papers for use for medical AI training data by exmaining if a paper's claims hold up against scientific evidence.

Given a corpus of PubMed papers in XML format, the filter produces a flat table whose key columns are the paper identifier and whether the paper's claims are truthful given the evidence, plus supporting metadata. Downstream
training can then keep, drop, or down-weight papers based on that table. The filter must generalise
across medical papers, not just a fixed list of known cases. 

Truth is discovered by checking each paper's XML file, then checking its claims against trusted evidence, not inferred
from surface features (fluency, formatting, citations) or hand-written rules.  Confident medical misinformation imitates those surface features well, so a classifier or rule-based filter rewards the wrong thing while this evidence-grounded approach does not. 

Retrieval must be able to find the evidence that contradicts a wrong claim. If it cannot, we risk providing untruthful or
false data to the training pipeline, which will increase the degree of hallucinations in medical AI models. This repository builds that filter, and includes a validation tool (or "harness") that measures this single dependency on a slice where we already know the answers, so we know the filter can work. The filter is the final product; the validation tool is simply the testing and scaffolding that validates it, and the graph/visualisation is a secondary aid.

We use an LLM in two bounded places; to extract the claim from a paper's text; and to judge whether a retrieved study supports or refutes it (unless we are in development mode and using generated data). The LLM does not run retrieval and does not decide the keep/drop verdict.

The validation tool's core question is for medical claims the field already knows were reversed, can
the retrieval process surface the contradicting or superseding evidence? Its output is a recall number plus
an error taxonomy showing where retrieval fails. The filter's output is the per-paper truthfulness table.


### Scope

- The data filter (`medfact-filter`) is the product: ingest PubMed XML, extract claims, retrieve
  evidence, judge stance, aggregate to a per-paper verdict, and write a flat CSV plus an
  HTML graph. It runs fully on offline stubs and swaps in real backends via env.
- `medfact-run` is the validation arm: it measures retrieval recall on the
  consensus-reversal gold slice, the dependency the filter's accuracy rests on.
- PubMed XML only. XML over .txt because it carries the
  CommentsCorrectionsList retraction/comment links, publication types, and MeSH.
- Claim-level scoring. Verdict is supported / contested / refuted / unverified; action is keep / downweight / drop.
- Sources: PubMed E-utilities and Europe PMC REST. No Retraction Watch, UMLS, or licensed corpus.
- Slice: consensus-reversal cases (HRT and WHI, CAST, H. pylori, hospitalized COVID and hydroxychloroquine).
- DuckDB is the single store, exact brute-force cosine, no ANN index yet.
- Python 3.12, packaged with uv.

### Roadmap
- On the roadmap, not built yet: GRADE aggregation, training-weight integration, UMLS
  grounding, ANN indexing, a non-LLM (NLI) stance backend.

### Scoring (the filter's verdict)

The filter's truthfulness scoring is mechanical and evidence-driven, not an LLM opinion. It
runs per claim, then rolls up to the paper (`transformation/scoring.py`).

- Each retrieved study has an evidence tier from its publication type (guideline 1.0,
  retraction 0.95, systematic review 0.9, meta-analysis 0.85, RCT 0.8, observational 0.5,
  case report 0.2, anything else 0.4). A study's pull on a claim is its tier times the stance
  judge's confidence.
- A claim's 0 to 1 score starts at 0.5, rises with the strongest supporting pull, and falls with
  the strongest refuting pull (refutation weighs roughly twice as much). Its verdict is
  refuted (a strong high-tier refutation), contested (evidence both ways, or a weak
  refutation), supported (support only), or unverified (no usable evidence).
- A paper is judged by its most damning claim: its score is the lowest claim score and its
  verdict is the worst claim verdict. The verdict maps to an action: refuted drops the paper,
  contested down-weights it, supported or unverified keeps it. Unverified is kept on purpose,
  because absent refutation is not proof a claim is false.

### Metrics (the validation tool's recall)

Each metric is a fraction over the gold claims. For every claim the validation tool records a pass or
fail, then reports the percentage that pass.

- Retrieval recall: fraction of reversed claims whose recorded disproving study landed in the
  retrieved pool. Ground-truth anchored and stance-independent, so the most trustworthy number.
- Stance recall: of the reversed claims where that study was retrieved, the fraction the stance
  judge labelled as refuting.
- Recall@k: fraction of reversed claims whose disproving study ranked in the top k by semantic
  similarity. False-contradiction rate: fraction of still-true controls wrongly flagged as
  refuted (lower is better).
- Error taxonomy per miss: not_indexed, retrieved_not_recognized, entity_miss,
  condition_mismatch, tier_inversion.

### Architecture (`src/medfact_poc/`)

The package is organised by role. Base classes live in `base/`, network
fetchers in `scraping/`, data changes in `transformation/`, and output in `reporting/`.

Shared core:
- `schema.py`: Pydantic models for both data filter, and the validation tool.
- `llm.py`: LLM clients (stub, Anthropic, OpenAI, Gemini), lazily
  imported. The single place a generative provider is chosen, used by stance and extract.
  The `LLMClient` Protocol lives in `base/llm.py`.
- `store.py`: DuckDB cache for the validation tool (candidates, embeddings, claim_retrieval, stance).

`base/` holds the Protocols, one module per concept: `Source`, `Embedder`, `StanceBackend`,
`ClaimExtractor`, `Retriever`, and `LLMClient`. Implementations live in their role folder and
import the Protocol from here.

`scraping/` (network fetchers): `pubmed.py` and `europepmc.py` evidence providers,
`sources.py` (the `get_sources` registry), `links.py` (pool expansion via PubMed retraction
links), `evidence.py` (the filter's `Retriever`, where the stub uses the paper's own comment
and retraction links and the live backend reuses the sources), and `http.py` (shared httpx
client, rate limiting, TLS handling: `MEDFACT_INSECURE_TLS`, `MEDFACT_CA_BUNDLE`,
`NCBI_EMAIL`, `NCBI_API_KEY`).

`transformation/`: `medline.py` (leaf extractors for efetch XML, shared by
`scraping/pubmed.py` and `transformation/ingest.py` so the two parsers cannot drift),
`ingest.py` (parses PubMed XML into `PaperRecord` and validates input, the pure parser is
tested), `query.py` (builds queries), `semantic.py` (re-ranks behind the `Embedder` Protocol),
`extract.py` (lifts claims behind `ClaimExtractor`, stub and LLM), `stance.py` (stance
classification behind `StanceBackend`, `classify_batch` runs calls concurrently, stub lexical
and `LLMStance`), and `scoring.py` (weighs stance into per-claim and per-paper verdicts, pure
and tested).

`reporting/` (output): `metrics.py` and `report.py` (validation tool aggregation, markdown plus CSV),
`flat_report.py` (the filter's flat CSV), and `graph.py` (`build_graph_data` and
`build_paper_graph_data` are pure and tested, `render_html` writes a self-contained
vis-network HTML file with a data-driven legend and summary, edge filters, and hover/click
focus).

`orchestration/` wires the layers together: `harness.py` (the validation arm, with
failure-bucket assignment) and `pipeline.py` (the data filter itself, scoring papers concurrently).

`cli/`: `medfact-build-cache`, `medfact-run`, `medfact-graph`, and `medfact-filter`.

Europe PMC (`scraping/europepmc.py`) is an evidence retrieval source. It is
queried to find studies that contradict or debate a claim. This data filter is for PubMed XML only.
The ingester accepts any PubMed XML it can read. It prints an error and skips a file it cannot
read or that is not a PubMed article set, and it prints a highlight for a paper that has no
`CommentsCorrectionsList`, which is the offline truthfulness signal.

Swappable plug points share the same Protocol shape: `Source`, `Embedder`, `StanceBackend`,
`ClaimExtractor`, `Retriever`, and `LLMClient`. Each has a dependency-free stub (an offline
placeholder that fakes the step with no LLM and no network) so the filter and validation tool run
offline, plus a real backend. Real runs need `MEDFACT_LLM_PROVIDER` in {anthropic, openai,
gemini}, `MEDFACT_EXTRACT_BACKEND=llm`, `MEDFACT_STANCE_BACKEND=llm`, and
`MEDFACT_RETRIEVER=live` (and `MEDFACT_EMBED_BACKEND=sbert` for the validation tool). Stub output is a
placeholder, not a real result.

### Data

`data/gold/consensus_reversals.yaml` is the gold dataset. Reversed claims carry the PMIDs of
the studies that overturned them (the known disproving studies). Controls carry none. All
those PMIDs were verified against PubMed esummary. It is a high-precision seed (ten reversals, eight controls).
It is the most accuracy-critical artifact, so change it with care.

### Invariants

- Factuality and writing quality are separate axes. The filter judges factuality. Do not
  merge them into one score, and never let writing quality stand in for truth.
- Keep claim conditions attached (population, comparator, dose, setting). Condition-stripped
  claims manufacture false verdicts.
- Retrieval recall stays ground-truth anchored and independent of the stance judge. A paper's truthfulness
  verdict can only be as reliable as the evidence retrieval surfaced for it.
- The truthfulness verdict is anchored in retrieved trusted evidence and its tier, not the
  LLM's freestanding opinion.
- Negative-evidence recall is a risk. Absence of contradicting evidence is not proof
  of truth. Do not assume retrieval found everything.

## Codebase Architecture 

### The Principle of Least Abstraction

Your goal is clarity over cleverness. Follow the "Keep it simple stupid" principle.

### Duplication vs. Abstraction

Avoid hasty abstractions. Duplication is often better than the wrong abstraction.

### Coding principles

First consider clarity and simplicity. If architecture changes must be made to the codebase, follow DRY and SOLID principles as an general guideline. You may be pragmatic if needed and do not need to be strictly bound to these principles if it does not make sense.

## Important Rules

- *Never* run git commands without asking for user permission, even if 'auto-accept' is selected during a Claude Command session.
- *Never* run files which use an LLM API without asking for user permission,  even if 'auto-accept' is selected during a Claude Command session.
- *Never* attempt to re-engineer the code or alter data without asking for user permission.
- There is no need to check UI changes in HTML files. I will manually examine.
- *Never* make assumptions. Ask for more information and wait for the user response.
- Do *not* use numerical prefixes when writing comments.
- Do *not* use newline characters in print statements.
- Use double quotation marks instead of single quotation marks when possible.
- Use type hinting.
- Favor modular, resuable code.
- Favor vectorised code.
- Favor lazy loading.
- Use abstract base classes or Protocols to ensure code stability.
- Use Pydantic where appropriate.
- Testing should be implemented with Pytest where appropriate (i.e. critical code).
- Use concurrency when processing data with an LLM API.
- Read existing files before writing any output.
- Do not re-read files unless they have been changed.
- Avoid AI style language like the overuse of emdashes or semi-colons, or overuse of patterns like "not x, but y" or "short label: longer explanation", or typical AI phrases like "honest review", "here's the catch", or "pragmatic approach".