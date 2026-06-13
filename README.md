# Argument Quality Analysis

A research codebase that studies what makes an argument persuasive on Reddit's
r/changemyview (CMV), and then uses those signals to drive an agentic
refinement loop that improves a candidate argument against a given post.

The project is organized around three core parts, plus a retrieval-corpus
pipeline and an offline test suite:

1. **Preprocessing** — building a clean pair-wise dataset of delta-awarded vs.
   non-delta CMV arguments.
2. **Models** — baseline TF-IDF classifiers, a zero-shot GPT-5.4-nano
   pair-wise baseline, and a QLoRA fine-tuned Qwen3-8B pair-wise ranker.
3. **Agents** — a LangGraph workflow that refines two opposing arguments
   against each other using Adaptive-RAG, Self-RAG, Reflective-RAG, and
   Reflexion patterns, with the Qwen ranker as the reward signal.

Supporting these are **RAG** (`rag/`), the pro-Israel retrieval-corpus pipeline
that scrapes, classifies, and ingests CMV arguments into Chroma for the agents;
the **harvester** (`harvester/`), which detects anti-Israel posts live across
Reddit/Lemmy/PieFed and drafts rebuttals with the agent workflow; and **tests**
(`tests/`), a fully offline suite covering graph wiring, helpers, and the package
layout.

## Repository layout

```
preprocessing/   # generic pair-wise data pipelines (Webis-CMV-20, winning-args-corpus)
rag/             # pro-Israel RAG-corpus pipeline (scrape -> classify -> ingest)
models/          # TF-IDF baselines + Qwen3-8B pair-wise ranker
agents/          # LangGraph refinement workflow + retrieval backends + generate entrypoint
harvester/       # live multi-platform detector -> draft -> notify (see "Harvester")
tests/           # offline graph-wiring, helper-unit, and import smoke tests
schemas.py       # Pydantic types shared across the pipelines
graph.png        # rendered topology of the agentic workflow (see "Agentic refinement")
data/            # generated artifacts (gitignored)
.chroma/         # Chroma vector store for the retriever (gitignored)
```

## Setup

This project uses [uv](https://github.com/astral-sh/uv) and requires Python
3.11+.

```bash
uv sync
```

Create a `.env` file at the repo root with whichever keys you need:

```
OPENAI_API_KEY=...
TAVILY_API_KEY=...          # optional, enables the web-search retrieval arm
HF_TOKEN=...                # optional, for dataset/model uploads
RANKER_PATH=...             # Qwen ranker checkpoint, for the agentic graph
```

### Observability (optional)

Both are opt-in and no-op unless the keys are present:

- **LangSmith** traces every node / LLM call of the agentic graph. Enable with
  `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY=ls__...`, and optionally
  `LANGSMITH_PROJECT=argument-quality`.
- **Weights & Biases** logs the Qwen QLoRA fine-tuning run. Enable with
  `WANDB_API_KEY=...` (and optionally `WANDB_PROJECT=qwen-qlora-ranker`) before
  running `uv run python -m models.qwen`.

### Tests

The `tests/` suite is fully offline — no API keys, network, or model loading
(every LLM / retrieval / Qwen boundary is stubbed). Run it with:

```bash
uv run pytest
```

It covers the graph's loop caps and termination (`test_graph_offline.py`), the
deterministic string/parse helpers (`test_helpers.py`), the package layout /
shared entrypoint (`test_layout_imports.py`), and the harvester's Fediverse
adapter parsing (`test_fediverse_adapters.py`).

## Data pipeline

The preprocessing pipeline produces a unified pair-wise argument quality
dataset from two sources:

- [Webis-CMV-20](https://webis.de/data/webis-cmv-20.html)
- [winning-args-corpus (Tan et al., 2016)](https://convokit.cornell.edu/documentation/winning.html)

```bash
uv run python -m preprocessing.preprocess
```

A semantic-similarity filter then removes off-topic and near-duplicate pairs.
OpenAI embeddings give three cosine similarities per row
(post↔delta, post↔nodelta, delta↔nodelta); thresholds keep only rows where
both arguments are on-topic with respect to the original post and the two arguments
are not near-duplicates of each other.

```bash
uv run python -m preprocessing.filter_by_similarity
```

A final token-length filter drops pairs whose arguments fall outside
`[MIN_TOKENS, MAX_TOKENS]` (per the ranker's tokenizer), caps the original-post
length, and enforces a max delta/nodelta length ratio so neither baseline nor
ranker can exploit raw length. This produces the `filtered_v2` split the models
train on.

```bash
uv run python -m preprocessing.filter_by_tokens
```

For the Israel-focused RAG corpus, the `rag/` package scrapes CMV threads via
arctic-shift, classifies each argument's stance, and ingests the high-confidence
pro-Israel arguments (plus a few legal primary sources) into Chroma:

```bash
uv run python -m rag.scrape_cmv_israel      # -> data/cmv_israel_rag.parquet
uv run python -m rag.classify_stance        # -> data/cmv_israel_rag_pro.parquet
uv run python -m rag.ingest_rag             # -> .chroma/ pro_israel_corpus
uv run python -m rag.ingest_legal_sources   # -> Palmer/San Remo legal chunks
```

## Baselines

Run TF-IDF + Logistic Regression / Random Forest / XGBoost over the pair-wise
dataset. The two arguments share a single TF-IDF vocabulary and enter the
classifier as one *signed difference* vector — `tfidf(arg_a) - tfidf(arg_b)` —
rather than as two concatenated blocks (context fields are vectorized as-is).
Swapping the two arguments therefore just negates the argument features, so the
model sees only the contrast between them and can't learn a "slot A is usually
the delta" positional shortcut. This holds the baselines to the same
order-invariant bar as the Qwen ranker, which scores each argument
independently. The shared featurizer lives in `models.tfidf_features.PairTfidf`.

```bash
uv run python -m models.main
```

## GPT-5.4-nano zero-shot baseline

Zero-shot pair-wise prompting of GPT-5.4-nano (no fine-tuning). Both
arguments are shown in a single prompt; the model picks `A` or `B`, and the
probability of `A` is calibrated from the top-logprob distribution at the
answer token.

```bash
uv run python -m models.gpt_5_4_nano
```

## Qwen3-8B pair-wise ranker

QLoRA (4-bit NF4) fine-tuning of Qwen3-8B with a margin ranking loss over
delta vs. non-delta arguments. Scoring is order-invariant — each argument
gets an independent forward pass and the higher score wins.

```bash
uv run python -m models.qwen
```

## Agentic refinement

The `agents` package wires a LangGraph workflow that drafts two candidate
arguments, eliminates the weaker one up front on the Qwen ranker, then refines
the survivor against retrieved evidence. Refinement is governed by four
independent loops, each with its own cap:

- **Grounding loop** (`hallucination_check -> refine`): re-refines while the
  draft is ungrounded, up to `MAX_GROUND_RETRIES` times **per refinement pass**
  (reset by the router each pass).
- **Refinement loop** (`stance_check -> router`): if the draft is on-topic but
  not yet a clearly pro-Israel reply, it reroutes for another refinement pass,
  up to `MAX_REFINE_ITERS` passes per generation.
- **Regeneration loops** (`stance_check`/`early_stance_check ->
  generate_initial`): if the draft is off-topic or anti-Israel, both drafts are
  thrown out and regenerated. The early gate regenerates up to
  `MAX_EARLY_REGEN_ITERS` times **per generation** (its budget reset on each
  fresh generation); the late gate regenerates up to `MAX_LATE_REGEN_ITERS`
  times across the whole run (each regeneration resets the refinement counter).
  When both the refinement and late-regeneration budgets are spent the late
  `stance_check` records `gave_up=True` and reports it honestly rather than
  shipping a non-pro-Israel argument. (The early gate only regenerates; it never
  decides give-up.)

There are **two stance gates**: a cheap `early_stance_check` on the raw survivor
(before any refinement — catches off-topic/anti survivors and regenerates
immediately, so a hopeless draft never burns a full refinement loop) and the
authoritative `stance_check` after `hallucination_check` (every argument that
ships via the late gate has passed through the grounding pass).

> **On "grounded".** The hallucination check verifies a claim is supported by
> the *retrieved evidence*, and the web arm is restricted to a pro-Israel /
> advocacy domain allow-list (`WEB_ALLOWED_DOMAINS`). So `grounded=True` means
> "consistent with the retrieved (one-sided, by design) sources", **not**
> "independently fact-checked / neutral". The `local` and `none` retrieval arms
> have no factual evidence to check against, so they mark the draft grounded
> *without running the grader*; `final_scores["grounding_verified"]`
> distinguishes a grader-verified pass (`True`) from an assumed one (`False`).
> Keep this in mind for any analysis that leans on the `grounded` flag.

Four patterns are fused into the graph:

- **Adaptive RAG** — the router picks between local Chroma and Tavily web
  search each pass.
- **Self-RAG** — two layers: `grade_docs` grades each retrieved chunk for
  relevance and keeps only the relevant subset (triggering a web search if none
  survive), and `hallucination_check` verifies the refined draft is grounded in
  the evidence.
- **Reflective RAG** — `reflect` grounds the critique in retrieved evidence.
- **Reflexion** — every critique is accumulated into a running
  `critique_history` that the refiner consumes in full, so it stops repeating
  fixed mistakes; the Qwen ranker supplies the one-time A-vs-B reward.

### Graph nodes

- **generate_initial** — drafts the two initial candidate arguments (`arg_a`,
  `arg_b`) from the topic and original post.
- **eliminate_loser** — runs one Qwen pairwise comparison on the two raw
  initial drafts and keeps the winner as the active side; only the survivor
  iterates from here. This is the only A-vs-B decision in the graph, made when
  both drafts are at equal polish. Elimination is permanent and not
  stance-aware: a wrongly-eliminated-but-salvageable draft is only recoverable
  via a full regeneration (which replaces both drafts).
- **early_stance_check** — cheap pre-refinement stance gate on the raw
  survivor. Catches only the off-topic / anti-Israel case (which refinement
  cannot rescue) and regenerates immediately *while the regeneration budget
  lasts*; on-topic survivors fall through to the router. It has just two
  out-edges (`generate_initial` | `router`) and never routes to `finalize`:
  once the regen budget is spent it hands the draft to the router so the late
  `stance_check` is always reached. Give-up is decided solely by the late gate.
- **router** — Adaptive-RAG: picks `local`, `web`, or `none` (skip retrieval —
  the current draft is strong enough to refine without new evidence) for the
  active side this pass; when it picks `web`, the same call also plans the
  search queries that target the critique's evidence gaps. It also resets the
  grounding-retry budget and clears any stale grounding verdict at the start of
  each pass. (Set `FORCE_RETRIEVAL_MODE={local|web}` in the env to pin the route
  for debugging.)
- **retrieve_local** — queries the Chroma `pro_israel_corpus` (delta-awarded
  CMV-Israel arguments) using the topic + post as the query.
- **retrieve_web** — runs the router's planned queries through Tavily,
  restricted to a curated domain allow-list (`WEB_ALLOWED_DOMAINS`).
- **skip_retrieval** — the router's `none` arm: adds no new documents and
  preserves the existing pool, so refinement still runs (and can cite
  previously-grounded facts) without fetching fresh evidence this pass.
- **grade_docs** — Self-RAG per-chunk relevance grading. Keeps the relevant
  subset of retrieved chunks for `reflect`; if none survive, the pool is dropped
  and `web_search=True` routes to the web-search fallback first.
- **web_search** — fallback triggered by an irrelevant grade: runs a Tavily
  search (planning queries from the critique if the router didn't) and fills
  the citation pool with web results before reflecting.
- **reflect** — generates a critique of the current draft grounded in the
  retrieved evidence, comparing it against the original post (what's missing /
  superfluous). Each critique is appended to `critique_history` (Reflexion).
- **refine** — rewrites the active side's draft using the full critique history
  plus any **fix notes** (ungrounded-claim issues from `hallucination_check`
  and/or a stance reason from `stance_check`), attaching inline `[n]` citations
  to evidence-backed claims. Drops an empty `### Sources` footer if the model
  cited nothing.
- **hallucination_check** — Self-RAG binary groundedness check. Skipped when
  retrieval was `local` or `none` (no factual evidence to check against — those
  passes are marked grounded but `grounding_verified=False`). Otherwise grades
  the draft; only a genuinely fabricated fact counts as ungrounded
  (citation-marker problems are tolerated). An ungrounded verdict spends one
  grounding retry and loops back to `refine` until the per-pass budget is spent.
- **stance_check** — the authoritative gate: classifies the refined draft as
  `pro_israel` (→ finalize), `neutral_needs_refine` (→ router, refinement loop)
  or `off_topic_or_anti` (→ generate_initial, regeneration loop). It records a
  `regen_reason` the next pass must fix, and sets `gave_up=True` (→ finalize)
  when both the refinement and regeneration budgets are exhausted.
- **finalize** — declares the surviving side the winner, publishes the
  result as `generation`, surfaces any grounded / pro-Israel / gave-up warnings
  (and whether grounding was verified), and prints the run's trajectory.

Run end-to-end:

```bash
uv run python -m agents.graph.builder \
  --topic "CMV: ..." \
  --post  "The original post body..."
```

Graph topology:

![graph](graph.png)

## Harvester

The `harvester/` package applies the agent workflow to live data: it searches
**Reddit, Lemmy, and PieFed** for recent anti-Israel posts, drafts a pro-Israel
rebuttal with the same `agents.generate` entrypoint, and pushes it to the operator
via ntfy. It is **read + draft + notify only** — it never posts back; a human
reviews and decides whether to post.

```
        INGESTION                  PIPELINE (per post)            POST-GENERATION
  ┌─────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────┐
  │ orchestrate.py      │   │ classify.py              │   │ notify.py  (ntfy)    │
  │  search 3 platforms │──►│  keyword + LLM stance    │──►│ tracking.py (SQLite) │
  │  dedup + age + sort │   │ ─► agents.generate       │   │                      │
  └─────────────────────┘   └──────────────────────────┘   └──────────────────────┘
            ▲
   fediverse/ adapters (Lemmy, PieFed, Reddit) — also exposed read-only via fediverse_mcp.py
```

### Run it

`orchestrate.py` is the single entrypoint and a **one-shot** — it runs once and
exits; schedule it externally (cron, a systemd timer, or on AWS an EventBridge
schedule → a Fargate task):

```bash
# Defaults: ≤3 answers, only posts from the last 24h, Reddit-first then newest.
uv run python -m harvester.orchestrate
uv run python -m harvester.orchestrate --dry-run                # search+classify, no spend
uv run python -m harvester.orchestrate --platforms lemmy,piefed --query "israel gaza"
```

Notifier setup: set `NTFY_TOPIC=cmv-<random>` in `.env` and subscribe to that
topic in the ntfy app (optional `NTFY_SERVER`, `NTFY_TOKEN`). The agent graph
also needs its usual keys (`OPENAI_API_KEY`, `TAVILY_API_KEY`, `RANKER_PATH`,
`HF_TOKEN`).

Each run searches every platform, drops posts older than `--max-age-hours`
(default 24) or already answered, and — if anything new is in window — answers up
to `--max-generations` (default 3), Reddit-first then newest: classify → draft →
notify. If nothing is new, it does nothing.

#### Dedup guarantee

Each post is answered **at most once, ever**. On first sight — before any
classify/generate work — its **canonical id** (the ActivityPub `ap_id`, or the
Reddit permalink) is claimed in a SQLite `seen` ledger (atomic `INSERT OR
IGNORE`, race-safe). The *same federated post on Lemmy and PieFed* shares a
canonical id, so it's answered once. `--dry-run` only peeks, never consumes the
ledger. The ledger persists in `harvester_tracking.db` (`HARVESTER_DB`); delete
it to reset.

### Files

| File | Role |
|------|------|
| `orchestrate.py` | The entrypoint. Search → dedup/age/sort → classify → generate → notify. |
| `fetch.py` | `fetch_from_rss()`: read the live Reddit Atom feed into `Post` objects (HTML → text). |
| `fediverse/` | Platform adapters behind one `Platform` interface (`base.py`, `lemmy.py`, `piefed.py`, `reddit.py`); `get_platform(name)` registry. |
| `fediverse_mcp.py` | **MCP server** (read-only): `search_posts` / `get_thread` over the adapters. |
| `classify.py` | Cheap keyword prefilter, then an LLM anti-Israel stance classifier. |
| `notify.py` | Send one ntfy push per generated response (the only outbound write). |
| `tracking.py` | SQLite: the `seen` dedup ledger + a `responses` store of generated rebuttals. |

**Cross-section dependency:** the only calls outside `harvester/` are to
`agents/` (`orchestrate.py` → `agents.generate`, `classify.py` → `agents.llm`);
everything else is self-contained.

### The Fediverse

Reddit's data API is approval-gated, but its public `/r/changemyview/new/.rss`
feed and the Lemmy/PieFed APIs are free to read unauthenticated. Lemmy and PieFed
are Reddit-like federated platforms where anti-Israel content is abundant. (Mbin
was evaluated and skipped — its search requires OAuth.)

`fediverse_mcp.py` exposes the read tools (`search_posts`, `get_thread`) over MCP.
The tools are read-only; the orchestrator itself currently drives them as a
deterministic loop.

### Out of scope: auto-posting

The pipeline ends at **notifying you** — it never posts to Reddit/Lemmy/PieFed.
Auto-posting political rebuttals violates these platforms' rules and risks bans,
so a human reviews and posts manually.

## Security

The harvester ingests **attacker-authored** text (any Reddit/Lemmy/PieFed
user can craft a post), so it ships with input-trust-boundary defenses:

- **Prompt injection** — untrusted title/body are fenced and neutralized
  (`harvester/classify.py::_neutralize`) before reaching the classifier and
  `agents.generate`; the graph's own `stance_check` / `hallucination_check` gates
  are the output-validation backstop, so a flipped draft is never shipped.
- **SSRF** — `harvester/fediverse/base.py::assert_safe_url()` blocks outbound
  requests to private / loopback / link-local / reserved IPs (every resolved
  address checked, defeating DNS rebinding), guarding the cloud metadata endpoint.
- **Cost abuse** — `--max-generations` and `--max-age-hours` bound paid work per
  run, and the `seen` ledger prevents re-answering.

`tests/test_security.py` covers the SSRF guard and the injection neutralizer.
