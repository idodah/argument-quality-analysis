# CLAUDE.md

Guidance for Claude Code (and humans) working in this repo.

## What this is

A research codebase on argument persuasiveness on Reddit r/changemyview (CMV).
Three parts, each runnable on its own:

- `preprocessing/` — builds the pair-wise (delta vs. non-delta) dataset from
  Webis-CMV-20 and winning-args-corpus.
- `models/` — TF-IDF baselines, a zero-shot GPT-5.4-nano baseline, and a QLoRA
  fine-tuned Qwen3-8B pair-wise ranker.
- `agents/` — a LangGraph refinement workflow (Adaptive-RAG + Self-RAG +
  Reflective-RAG + Reflexion) that drafts two opposing arguments, eliminates the
  weaker one on the Qwen ranker, then refines the survivor against retrieved
  evidence.
- `rag/` — the pro-Israel retrieval-corpus pipeline (scrape → classify → ingest).
- `webapp/` — a Gradio UI over the same `agents.generate` entrypoint as the CLI.

See `README.md` for the full architecture and the graph topology (`graph.png`).

## Conventions

- **Package manager is `uv`.** Run everything through it: `uv run python -m ...`,
  `uv add <pkg>`, `uv sync`. Don't call `pip` or a bare `python`.
- **Entry points are modules**, not loose scripts: `uv run python -m models.main`,
  `uv run python -m agents.graph.builder --topic ... --post ...`,
  `uv run python -m webapp.app`. (`run.py` is a local-only convenience and is
  gitignored.)
- **Secrets live in `.env`** (gitignored): `OPENAI_API_KEY`, `TAVILY_API_KEY`,
  `HF_TOKEN`, `RANKER_PATH`. Never hard-code keys or echo them.
- **Generated data is gitignored**: `data/`, `.chroma/`, `checkpoints/`, `*.csv`,
  `*.parquet`, `*.xlsx`. Don't commit artifacts.

## Tests

The `tests/` suite is **fully offline** — no API keys, no network, no model
loading. Every LLM / retrieval / Qwen boundary is stubbed (see
`test_graph_offline.py`'s `_make_scenario`). Keep it that way: a test that needs
a key or a GPU does not belong here.

```bash
uv run pytest          # whole suite
```

When changing the agent graph, run `test_graph_offline.py` — it drives the real
compiled graph and asserts the loop caps and termination still hold.

## Design invariants (don't break these without good reason)

- **Loop caps.** Two independent caps gate refinement:
  `MAX_OUTER_ITERS` (stance passes) × `MAX_GROUND_RETRIES` (grounding re-refines
  per pass), worst case 3 × 2 = 6 grounding refines. They live in
  `agents/graph/state.py`. If you change them, update the README and the
  graph-offline test (which derives expectations from the constants).
- **The ranker is order-invariant** and used **once**, at `eliminate_loser`.
  It scores each argument independently; there is no A/B token and no
  per-iteration ranking.
- **Citations are stripped before the ranker sees an argument**
  (`strip_citations` in `state.py`) — it was trained on uncited prose.
- **Retrieval is deliberately pro-Israel only** (`WEB_ALLOWED_DOMAINS` in
  `state.py`). This is an intentional design choice for a one-sided generator,
  documented at length there — not an oversight.
- **`looks_like_critique` is a safety guard**, not a nicety: if the refiner
  emits commentary instead of an argument, refine no-ops deterministically
  because the ranker has been observed to score critique-shaped text higher.

## Working style in this repo

- Verify, don't assume: run the tests after changing graph logic; check that a
  referenced module/symbol actually exists before documenting it.
- Keep commits tight and explain *why* in the message. This repo uses
  `Co-Authored-By: Claude` trailers — keep them; AI assistance here is explicit.
- Prefer editing a module's docstring when behavior changes, so docs don't drift
  (e.g. the `docs/` → `.chroma/` retriever migration).
