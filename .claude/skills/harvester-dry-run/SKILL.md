---
name: harvester-dry-run
description: Use to safely exercise the live multi-platform harvester (Reddit / Lemmy / PieFed anti-Israel post detection) without spending on LLM calls or consuming the dedup ledger. Explains the dry-run path, the at-most-once dedup guarantee, and the no-auto-post boundary.
---

# Harvester dry run

The harvester searches Reddit, Lemmy, and PieFed for recent anti-Israel posts,
drafts a rebuttal with `agents.generate`, and pushes it via ntfy. Use a dry run
to see what it *would* answer without spending or mutating state.

## Dry run (no spend, no ledger writes)

```bash
uv run python -m harvester.orchestrate --dry-run
```

`--dry-run` searches and **peeks** at the candidate set: it does **not** classify
or generate (zero LLM spend) and **never consumes** the `seen` ledger. Safe to
run repeatedly.

Useful flags:

- `--platforms lemmy,piefed` — restrict the search (default: all three).
- `--query "israel gaza"` — override the search query.
- `--max-age-hours 24` — drop posts older than this (default 24).
- `--max-generations 3` — cap answers per run (default 3); ignored under
  `--dry-run` since nothing is generated.

## The at-most-once dedup guarantee

Each post is answered **at most once, ever**. On first sight — before any
classify/generate work — the post's **canonical id** (the ActivityPub `ap_id`,
or the Reddit permalink) is claimed in a SQLite `seen` ledger via an atomic
`INSERT OR IGNORE` (`tracking.mark_seen`). Consequences:

- No new posts in window -> everything skipped, **zero LLM calls**.
- The *same federated post on Lemmy and PieFed* shares a canonical id, so it is
  answered **once**.
- Race-safe (PRIMARY KEY). `--dry-run` only peeks; it never consumes the ledger.

The ledger persists in `harvester_tracking.db` (`HARVESTER_DB`). Delete that file
to reset and re-answer everything.

## Boundary: it never auto-posts

The pipeline ends at **notifying the operator** (one ntfy push per draft). It is
**read + draft + notify only** — it never posts back to Reddit/Lemmy/PieFed,
because auto-posting political rebuttals violates platform rules and risks bans.
A human reviews and decides whether to post. Detection + drafting (the
automatable part) is the system's job; the decision to post stays with a person.

## The MCP placement

`harvester/fediverse_mcp.py` exposes read-only `search_posts` / `get_thread`
tools over MCP across the swappable Fediverse adapters — the *justified* MCP
placement (a decision point over heterogeneous-but-uniform backends). The fixed
RSS -> generate pipeline uses no MCP; MCP belongs at decision points, not fixed
plumbing.

## Notifier setup (for a real run)

Set `NTFY_TOPIC=cmv-<random>` in `.env` and subscribe to it in the ntfy app.
A real (non-dry) run also needs the agent graph's keys: `OPENAI_API_KEY`,
`TAVILY_API_KEY`, `RANKER_PATH`, `HF_TOKEN`.
