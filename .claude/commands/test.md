---
description: Run the offline test suite and summarize pass/fail
---

Run the project's offline test suite and report the result.

Steps:
1. Run `uv run pytest -q`.
2. Report the total number of tests and whether they all passed.
3. If anything failed, show the failing test names and the assertion detail,
   then briefly diagnose the likely cause (is it a test that needs updating, or
   a real regression in the code under test?). Do not change code without
   confirming the intended behavior first.

Notes:
- The suite is fully offline (no API keys, network, or model loading). A failure
  about a missing key or GPU means a test broke that contract — flag it.
- `test_graph_offline.py` drives the real compiled graph; its expectations are
  derived from `MAX_OUTER_ITERS` / `MAX_GROUND_RETRIES` in
  `agents/graph/state.py`. If those constants changed, the test and the README
  must change with them.

$ARGUMENTS
