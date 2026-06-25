---
name: diagnose-graph
description: Use when an agent run loops, stalls, repeatedly regenerates, or reports gave_up, to trace which of the four loop budgets was hit against the printed trajectory. Diagnoses termination behavior of the LangGraph refinement workflow.
---

# Diagnose the agent graph

The refinement workflow is governed by **four independent loops**, each with its
own cap. A run that loops, stalls, or gives up is almost always one budget being
spent. Find which one.

## Read the real budgets first

Do **not** assume the values — read them from
[agents/graph/state.py](../../../agents/graph/state.py) each time, because they
change and the termination contract depends on them. As currently set:

| Constant | Loop | Edge | Value |
|----------|------|------|-------|
| `MAX_GROUND_RETRIES` | Grounding | `hallucination_check -> refine` | 2 retries **per refinement pass** (reset each pass by the router) |
| `MAX_REFINE_ITERS` | Refinement | `stance_check -> router` | 3 passes per generation |
| `MAX_EARLY_REGEN_ITERS` | Early regeneration | `early_stance_check -> generate_initial` | 2 retries per generation (budget reset on each fresh generation) |
| `MAX_LATE_REGEN_ITERS` | Late regeneration | `stance_check -> generate_initial` | 1 retry across the whole run (each resets the refinement counter) |

## The two stance gates

- **`early_stance_check`** — cheap, runs on the raw survivor before any
  refinement. **Only regenerates** (off-topic / anti-Israel survivors) while its
  budget lasts; otherwise falls through to the router. It **never** decides
  give-up and never routes to finalize.
- **`stance_check`** — the authoritative late gate after `hallucination_check`.
  Routes `pro_israel -> finalize`, `neutral_needs_refine -> router`,
  `off_topic_or_anti -> generate_initial`. It is the **only** node that sets
  `gave_up=True` (when both refinement and late-regen budgets are exhausted).

## Symptom -> cause

Read the printed trajectory and match:

- **Stuck re-refining the same draft** -> grounding loop
  (`hallucination_check -> refine`), bounded by `MAX_GROUND_RETRIES` per pass.
- **Repeated full regenerations early** -> `early_stance_check` rejecting the raw
  survivor as off-topic/anti, bounded by `MAX_EARLY_REGEN_ITERS`.
- **Repeated refinement passes without converging** -> `stance_check` returning
  `neutral_needs_refine`, bounded by `MAX_REFINE_ITERS`.
- **`gave_up=True`** -> the refinement budget *and* the single late-regen budget
  are both spent; the late gate shipped honestly rather than a non-pro-Israel
  reply. This is convergence failure, not success.

## Verify the contract still holds

`tests/test_graph_offline.py` drives the real compiled graph and encodes the
loop caps + termination guarantee. If you changed any `MAX_*` constant, that
test **and** the README "Agentic refinement" section must change with it. Run
the offline suite to confirm the graph still terminates:

```bash
uv run pytest -q tests/test_graph_offline.py
```
