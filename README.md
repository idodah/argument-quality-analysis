# Argument Quality Analysis

This project's goal is to **find posts advancing antisemitic tropes on three social
networks (e.g. reddit) and automatically generate a well-grounded, persuasive
factual refutation of each one.** To do that well, it first studies
what makes an argument persuasive at all and trains a model that, given two
arguments, predicts which one changed the reader's view, then builds an agent
that drafts a rebuttal and refines it against retrieved evidence.

### Background: r/changemyview and deltas

The persuasion signal comes from Reddit's [r/changemyview
(CMV)](https://www.reddit.com/r/changemyview/), a forum where someone posts an
opinion they hold and explicitly invites others to change their mind. When a
reply genuinely shifts the original poster's view, the OP (the person who created the r/changemyview post) awards it a **delta**
(Δ). A
delta is therefore a human-labeled marker of a *persuasive* argument: among all
the replies to a post, the delta-awarded ones are the comments that demonstrably
worked. We treat **delta vs. non-delta** as the ground-truth label for argument
quality and learn to rank arguments accordingly.

## The three core parts

1. **Preprocessing** (`preprocessing/`) — builds a clean pair-wise dataset of
   delta-awarded vs. non-delta CMV arguments to learn persuasiveness from.
2. **Models** (`models/`) — TF-IDF baselines, a zero-shot GPT-5.4-nano baseline,
   and a QLoRA fine-tuned Qwen3-8B ranker, all trained to predict which of two
   arguments changed the reader's view.
3. **Agents** (`agents/`) — a LangGraph workflow that drafts **two** candidate
   refutations of a given post's trope, then iteratively refines the stronger
   one against retrieved evidence, using Adaptive-RAG, Self-RAG, Corrective-RAG
   (CRAG), and Reflexion patterns.

Supporting these are **RAG** (`rag/`), a two-part retrieval corpus — authoritative
trope-refutation articles (USHMM, Wikipedia) for the factual record, plus
delta-awarded CMV comments for persuasive form — ingested into Chroma so the
agent can ground its rebuttals; the **harvester**
(`harvester/`), which detects trope-advancing posts live across the three social
networks (Reddit, and the Fediverse platforms Lemmy and PieFed) and drafts
rebuttals with the agent workflow; **infra** (`infra/`), the Terraform + GitHub
Actions stack that runs the harvester on AWS as a scheduled, serverless job; and
**tests** (`tests/`), a fully offline suite covering graph wiring, helpers, and
the package layout.

## Repository layout

```
preprocessing/   # pair-wise data pipelines (Webis-CMV-20, winning-args-corpus)
rag/             # trope-refutation retrieval corpus
models/          # TF-IDF baselines, GPT-5.4-nano baseline, Qwen3-8B ranker
agents/          # LangGraph refinement workflow
harvester/       # live Reddit/Lemmy/PieFed detector -> draft -> notify
infra/           # AWS deployment: Terraform stack + SageMaker packaging
tests/           # offline graph-wiring, and helper-unit
schemas.py       # Pydantic types shared across the pipelines
```

## Setup

This project uses [uv](https://github.com/astral-sh/uv) and requires Python
3.11+.

```bash
uv sync
```

Create a `.env` file at the repo root with whichever keys are needed:

```
OPENAI_API_KEY=...          # OpenAI embeddings (retrieval / similarity filter)
AWS_REGION=...              # Bedrock region for the LLM (e.g. eu-west-1)
BEDROCK_MODEL_ID=...        # optional, defaults to amazon.nova-2-lite-v1:0
TAVILY_API_KEY=...          # optional, enables the web-search retrieval arm
HF_TOKEN=...                # optional, for dataset/model uploads
RANKER_PATH=...             # Qwen ranker checkpoint, for the agentic graph
```

The agent graph's LLM calls go through **Amazon Bedrock** (Nova 2 Lite by
default). Set `LLM_BACKEND=openai` to use OpenAI instead (no AWS needed).

### Observability (optional)

Both are opt-in and no-op unless the keys are present:

- **LangSmith** traces every node / LLM call of the agentic graph. Enable with
  `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY=ls__...`.
- **Weights & Biases** logs the Qwen QLoRA fine-tuning run. Enable with
  `WANDB_API_KEY=...` before
  running `uv run python -m models.qwen`.

### Tests

The `tests/` suite is fully offline — no API keys, network, or model loading. 

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
ranker can exploit raw length. This produces the split the models
train on.

```bash
uv run python -m preprocessing.filter_by_tokens
```

## Baselines

Simple bag-of-words baselines: TF-IDF features into Logistic Regression /
Random Forest / XGBoost, predicting which of the two arguments earned the delta.

The two arguments share one TF-IDF vocabulary, and the classifier sees their
*difference*, `tfidf(arg_a) - tfidf(arg_b)`, rather than the two vectors
side by side (the topic and post are vectorized separately as context).

```bash
uv run python -m models.tfidf_main
```

## GPT-5.4-nano zero-shot baseline

Zero-shot pair-wise prompting of GPT-5.4-nano (no fine-tuning). Both
arguments are shown in a single prompt; the model picks `A` or `B`.

```bash
uv run python -m models.gpt_5_4_nano
```

## Qwen3-8B ranker

QLoRA fine-tuning of Qwen3-8B. It is **trained pair-wise but scores
point-wise**: training uses a margin ranking loss over (delta, non-delta) pairs
to push `score(delta) > score(non-delta)`, but at inference each argument gets
its own independent forward pass to a single scalar — there is no A/B token and
the model never sees both arguments at once. Ranking two candidates is then just
comparing their two scalars, so scoring is order-invariant by construction (the
higher score wins regardless of order).

```bash
uv run python -m models.qwen
```

## Results

Test-split metrics for each model (best run per model):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| gpt_5_4_nano | 0.5324 | 0.5337 | 0.6274 | 0.5768 | 0.5308 |
| tfidf_logreg | 0.5604 | 0.5800 | 0.4880 | 0.5300 | 0.5967 |
| tfidf_random_forest | 0.5275 | 0.5438 | 0.4327 | 0.4819 | 0.5487 |
| tfidf_xgboost | 0.5238 | 0.5312 | 0.5312 | 0.5312 | 0.5400 |
| qwen_qlora_rank | **0.6716** | **0.6824** | **0.6611** | **0.6716** | **0.7236** |

## Agentic refinement

The `agents` package wires a LangGraph workflow that drafts two candidate
arguments, eliminates the weaker one up front on the Qwen ranker, then refines
the survivor against retrieved evidence. The rest of this section builds up from
the bottom: first the individual **graph nodes**, then how they wire into the
**graph topology**, and finally the **loops**, **stance gates**, and **RAG
patterns** that the wiring implements.

### Graph nodes

- **generate_initial** — drafts the two initial candidate arguments (`arg_a`,
  `arg_b`) from the topic and original post, in plain prose.
- **eliminate_loser** — runs one Qwen pairwise comparison on the two raw
  initial drafts and keeps the winner as `argument`; only that survivor
  iterates from here (the loser is dropped).
- **early_stance_check** — pre-refinement stance gate on the raw
  survivor. Catches only the off-topic / ad-hominem / political-drift case and regenerates immediately; on-topic survivors fall through to the router.
- **router** — Adaptive-RAG: each pass picks `local`, `web`, or `none` (skip
  retrieval — the current draft is strong enough to refine without new
  evidence); when it picks `web`, the same call also plans the search queries
  that target the critique's evidence gaps. The `none` route is not a node — it
  edges straight to `reflect`, adding no new documents and preserving the
  existing pool, so refinement still runs and can cite previously-grounded facts
  without fetching fresh evidence.
- **retrieve_local** — queries the Chroma `trope_refutation_corpus` (both halves:
  reference articles and delta-awarded CMV comments) using the topic + post as
  the query. Deliberately unfiltered by `source` — filtering to one source value
  would hide the reference corpus and leave `hallucination_check` with nothing
  local to verify against.
- **retrieve_web** — runs the router's planned queries through Tavily,
  restricted to a curated domain allow-list (`WEB_ALLOWED_DOMAINS`): Holocaust
  museums and research institutes (USHMM, Yad Vashem, IHRA), antisemitism
  monitors (ADL, JPR, Kantor Center), mainstream fact-checkers (AFP, Reuters,
  AP, Snopes, PolitiFact, Full Fact), and general reference (Wikipedia,
  Britannica, JSTOR).
- **grade_docs** — CRAG per-chunk relevance grading. Keeps the relevant
  subset of retrieved chunks for `reflect`; if none survive, the pool is dropped
  and `web_search=True` routes to the web-search fallback first.
- **web_search** — CRAG corrective action, triggered by an irrelevant grade:
  runs a Tavily search (planning queries from the critique if the router didn't)
  and fills the citation pool with web results before reflecting.
- **reflect** — generates a critique of the current draft grounded in the
  retrieved evidence, comparing it against the original post (what's missing /
  superfluous). Each critique is appended to `critique_history` (Reflexion).
- **refine** — rewrites the current draft using the full critique history
  plus any **fix notes** (ungrounded-claim issues from `hallucination_check`
  and/or a stance reason from `stance_check`), attaching inline citations
  to evidence-backed claims.
- **hallucination_check** — Self-RAG binary groundedness check.
- **stance_check** — classifies the refined draft as `refutes_trope` (→ finalize),
  `neutral_needs_refine` (→ router, refinement loop), or `off_topic_or_anti`
  (→ generate_initial, regeneration loop). It records a `regen_reason` the next
  pass must fix, and sets `gave_up=True` (→ finalize) when both the refinement
  and regeneration budgets are exhausted.
- **finalize** — terminal node: publishes the refined draft as `generation`, surfaces any
  grounded / refutation-quality / gave-up warnings (and whether grounding was verified),
  and prints the run's trajectory.

### Graph topology

The nodes above wire into the graph below. Run it end-to-end with:

```bash
uv run python -m agents.graph.builder \
  --topic "CMV: ..." \
  --post  "The original post body..."
```

![graph](graph.png)

### Refinement loops

Refinement is governed by four independent loops, each with its own cap:

- **Grounding loop** (`hallucination_check -> refine`): re-refines while the
  draft is ungrounded, up to `MAX_GROUND_RETRIES = 2` (2 grounding retries) per
  refinement pass.
- **Refinement loop** (`stance_check -> router`): if the draft is on-topic but
  not yet a substantive refutation, it reroutes for another refinement pass,
  up to `MAX_REFINE_ITERS = 3` (3 passes — the first pass plus 2 reroutes) per
  generation.
- **Early-regeneration loop** (`early_stance_check -> generate_initial`): if the
  raw survivor is off-topic or drifts into political advocacy, both drafts are
  thrown out and regenerated before any refinement, up to
  `MAX_EARLY_REGEN_ITERS = 2` (2 regeneration retries) per generation.
- **Late-regeneration loop** (`stance_check -> generate_initial`): if a refined
  draft is still off-topic or politically drifting, both drafts are regenerated, up to
  `MAX_LATE_REGEN_ITERS = 1` (1 regeneration retry) across the whole run (each
  regeneration resets the refinement counter).

### RAG patterns

Four patterns are fused into the graph:

- **Adaptive RAG** — each pass the router picks one of three routes: local
  Chroma retrieval, Tavily web search, or `none` (skip retrieval entirely and
  refine the current draft on the evidence already in hand).
- **Corrective RAG (CRAG)** — `grade_docs` grades each retrieved chunk for
  relevance and keeps only the relevant subset; when no chunk survives, the pool
  is dropped and the run takes the corrective action of falling back to a
  `web_search` before reflecting, rather than generating on bad evidence.
- **Self-RAG** — `hallucination_check` verifies the refined draft is actually
  grounded in the retrieved evidence, re-refining while it is not.
- **Reflexion** — every critique is accumulated into a running
  `critique_history` that the refiner consumes in full, so it stops repeating
  fixed mistakes.

## Harvester

The `harvester/` package applies the agent workflow to live data: it searches
three social networks — **Reddit, Lemmy, and PieFed** — for recent posts
advancing an antisemitic trope, drafts a factual refutation
with the same `agents.generate` entrypoint, and pushes it to the operator
via ntfy. It is **read + draft + notify only** — it never posts back; a human
reviews and decides whether to post.

Detection is deliberately conservative: a cheap keyword prefilter (trope
vocabulary, not geopolitics) gates an LLM classifier that is instructed to
answer "no" whenever a post is criticism of a government or state, is quoting
or debunking a trope, or is simply ambiguous. Combined with the human-in-the-loop
boundary above, a false positive costs an operator one discarded draft rather
than publicly labelling someone a bigot.

```
        INGESTION                  PIPELINE (per post)            POST-GENERATION
  ┌─────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────┐
  │ orchestrate.py      │   │ classify.py              │   │ notify.py  (ntfy)    │
  │  search 3 platforms │──►│  keyword + LLM trope     │──►│ tracking.py          │
  │  dedup + age + sort │   │ ─► agents.generate       │   │  (SQLite / DynamoDB) │
  └─────────────────────┘   └──────────────────────────┘   └──────────────────────┘
            ▲
   fediverse/ adapters (Lemmy, PieFed, Reddit) — also exposed read-only via fediverse_mcp.py
```

### Run it

`orchestrate.py` is the single entrypoint and a **one-shot** — it runs once and
exits.

```bash
# Defaults: ≤3 answers, only posts from the last 24h, Reddit-first then newest.
uv run python -m harvester.orchestrate
uv run python -m harvester.orchestrate --dry-run                # search+classify, no spend
uv run python -m harvester.orchestrate --platforms lemmy,piefed --query 'rothschild "blood libel"'
```

**Notifier setup.** Notifications go to **ntfy** — the only backend, and the
only outbound write the harvester makes.

```
NTFY_TOPIC=cmv-<something-long-and-random>
NTFY_SERVER=https://ntfy.sh    # optional, this is the default
NTFY_TOKEN=tk_...              # optional, Bearer auth for a protected topic
```

Then subscribe to that topic in the [ntfy app](https://ntfy.sh/app) or at
`https://ntfy.sh/<topic>`. That is the whole setup — no account, no bot, no
password.

Long refutations are **not** split. ntfy stores any payload over 4096 bytes as
a `.txt` attachment and links to it from the notification, so the full text
survives in one push. (An earlier version of this module chunked at 4000 bytes
into several independent pushes, which arrived unordered and cut mid-sentence —
that is the bug the single-request approach fixes.)

The agent graph also needs its usual keys (`OPENAI_API_KEY`, `TAVILY_API_KEY`,
`RANKER_PATH`, `HF_TOKEN`).

Each run searches every platform and drops posts older than `--max-age-hours`
(default 24) or already answered. Whatever new posts remain within that window it
answers, Reddit-first then newest: classify → draft → notify. If nothing new is
in window, it does nothing.

**Searching is one query per term, not one query for the whole phrase.** Lemmy
and PieFed pass `q` straight to their APIs, which match a multi-word query as a
phrase to be found in full — so a six-term query matched *nothing* on either
platform across a week of live runs, while each term on its own returned results
immediately. `--query` is therefore split into terms (shell-style quoting keeps
a multi-word term whole: `--query 'zog "blood libel"'`) and each is searched
separately, with results merged and deduped on `canonical_id`.

Two consequences worth knowing:

- Keep the term list short. Each term is one API call per platform per run; the
  26-term prefilter in `classify.py` does the real narrowing once posts are
  fetched, so the query only has to be broad enough to surface candidates.
- Reddit fetches its feed **once per run**, not once per term. It has no
  unauthenticated search API, so the adapter filters a locally-fetched `/new`
  RSS feed — and re-fetching it per term tripped Reddit's rate limiter (429 on
  five of six terms, leaving the Reddit arm empty). The feed is identical for
  every term, so it is cached for the life of the adapter instance.

#### Dedup guarantee

Each post is answered **at most once, ever**. On first sight — before any
classify/generate work — its **canonical id** (the ActivityPub `ap_id`, or the
Reddit permalink) is recorded in a `seen` store.

### Files

| File | Role |
|------|------|
| `orchestrate.py` | The entrypoint. Search → dedup/age/sort → classify → generate → notify. |
| `fetch.py` | `fetch_from_rss()`: read the live Reddit feed into `Post` objects (HTML → text). |
| `fediverse/` | Platform adapters behind one `Platform` interface (`base.py`, `lemmy.py`, `piefed.py`, `reddit.py`); `get_platform(name)` registry. Lemmy/PieFed query their search APIs; Reddit filters a cached `/new` RSS feed locally. |
| `fediverse_mcp.py` | **MCP server** (read-only): `search_posts` / `get_thread` over the adapters. |
| `classify.py` | keyword prefilter, then an LLM antisemitic-trope classifier (criticism of a government is excluded by design). |
| `notify.py` | Message shape (`format_result`) + re-exports the ntfy transport (the only outbound write). |
| `notify_ntfy.py` | ntfy transport — one request per message; ntfy stores an oversized body as a linked `.txt`. |
| `tracking.py` | The `seen` dedup store + a `responses` store of generated rebuttals. |

### The Fediverse

Reddit's data API is approval-gated, but its public `/r/changemyview/new/.rss`
feed and the Lemmy/PieFed APIs are free to read unauthenticated. Lemmy and PieFed
are Reddit-like federated platforms.

`fediverse_mcp.py` exposes the read tools (`search_posts`, `get_thread`) over MCP.
The tools are read-only; the orchestrator itself currently drives them as a
deterministic loop.

## Deployment

The harvester runs on AWS as a scheduled, serverless job, defined entirely as
code in `infra/` (Terraform + GitHub Actions — no click-ops).

```
EventBridge Scheduler  ──►  ECS Fargate (one-shot task)  ──►  Amazon Bedrock (Nova 2 Lite)
   (rate(1 hour))            harvester image from ECR           DynamoDB (seen / responses)
                                    │                           SageMaker async GPU endpoint
                                    └─ secrets from Secrets Manager   (Qwen ranker, optional)
```

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
