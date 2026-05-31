# CLAUDE.md

Guidance for Claude Code (and humans) working in this repo.

## What this is

A research codebase on argument persuasiveness on Reddit r/changemyview (CMV),
in a few self-contained parts:

- `preprocessing/` — builds the pair-wise (delta vs. non-delta) argument dataset.
- `models/` — TF-IDF baselines, a zero-shot LLM baseline, and a fine-tuned
  pair-wise ranker.
- `agents/` — a LangGraph workflow that drafts two opposing arguments, keeps the
  stronger one, and refines it against retrieved evidence.
- `rag/` — the retrieval-corpus pipeline (scrape → classify → ingest).
- `webapp/` — a Gradio UI over the same generation entrypoint as the CLI.

See `README.md` for the full architecture and the graph topology (`graph.png`).

## Conventions

- **Package manager is `uv`.** Run everything through it: `uv run python -m ...`,
  `uv add <pkg>`, `uv sync`. Don't call `pip` or a bare `python`.
- **Entry points are modules**, not loose scripts (`uv run python -m <package>`).
- **Secrets live in `.env`** (gitignored). Never hard-code keys or echo them.
- **Generated data and artifacts are gitignored** (`data/`, `.chroma/`,
  `checkpoints/`, spreadsheets, etc.). Don't commit them.

## Tests

The `tests/` suite is **fully offline** — no API keys, no network, no model
loading (LLM / retrieval / model boundaries are stubbed). Keep it that way: a
test that needs a key or a GPU does not belong here.

```bash
uv run pytest
```

When changing the agent graph, run the offline graph test — it drives the real
compiled graph and checks that the refinement loops stay bounded and terminate.

## Working style

- Verify, don't assume: run the tests after changing logic, and confirm a
  referenced module or symbol actually exists before documenting it.
- Keep commits tight and explain *why* in the message. This repo uses
  `Co-Authored-By: Claude` trailers — AI assistance here is explicit.
- When behavior changes, update the nearest docstring/README so the docs don't
  drift from the code.
