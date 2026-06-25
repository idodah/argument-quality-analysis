---
name: run-agent
description: Use when asked to generate a rebuttal end-to-end, run the agent graph or the Gradio webapp, or interpret a generation's grounded / pro-Israel / gave-up status. Covers the two entrypoints (CLI and webapp) and what the output flags actually mean.
---

# Run the agentic generation

Drive the LangGraph workflow that drafts two opposing arguments, eliminates the
weaker one on the Qwen ranker, and refines the survivor against retrieved
evidence into a pro-Israel rebuttal.

## Preflight

A real run loads the Qwen ranker and makes several LLM + web-search calls —
expect **tens of seconds to a couple of minutes** per request. Before running,
confirm `.env` has the keys the run needs:

- `OPENAI_API_KEY` — required (drafting, grading, refining).
- `TAVILY_API_KEY` — required for the web retrieval arm.
- `RANKER_PATH` — the Qwen ranker checkpoint (the eliminate_loser reward signal).
- `HF_TOKEN` — needed to pull the base model for the ranker.

If a key is missing, say which and stop — don't run a partial generation.

## Entrypoints (same logic)

Both call `agents.generate.generate_pro_israel_response`, so they generate
identically.

- **CLI** — one-shot, scriptable:
  ```bash
  uv run python -m agents.graph.builder --topic "CMV: ..." --post "The original post body..."
  ```
- **Webapp** — Gradio UI for interactive use:
  ```bash
  uv run python -m webapp.app
  ```
  Then open the local URL it prints.

Debug knob: set `FORCE_RETRIEVAL_MODE={local|web}` in the env to pin the
router's retrieval arm instead of letting Adaptive-RAG choose.

## Interpreting the result

The output flags are subtle — read them precisely, don't paraphrase loosely:

| Flag | What it actually means |
|------|------------------------|
| `grounded=True` | Consistent with the **retrieved (one-sided, by design) sources** — *not* independently fact-checked or neutral. |
| `grounding_verified=True` | A grader actually ran and passed. `False` = grounding was *assumed* (the `local` / `none` retrieval arms skip the grader because they have no factual evidence to check against). |
| `pro_israel` | The late `stance_check` classified the shipped draft as a clearly pro-Israel reply. |
| `gave_up=True` | The refinement **and** late-regeneration budgets were both spent and it is reporting honestly rather than shipping a non-pro-Israel argument. Treat this as "failed to converge," not "succeeded." |

When summarizing a run, lead with whether it converged (`pro_israel` and not
`gave_up`) and whether grounding was *verified* vs. assumed.

## See also

- `diagnose-graph` skill — when a run loops, stalls, or reports `gave_up`.
- README "Agentic refinement" section — full node-by-node topology.
