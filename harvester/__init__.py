"""Harvester: detect posts advancing antisemitic tropes (Reddit/Lemmy/PieFed),
draft factual refutations.

READ + DRAFT + NOTIFY only — it never posts, comments, or votes; the only outbound
write is a single Discord webhook push to the operator. The generation logic itself lives in
`agents/` and is reused here.

Layout (see the project README for the full map):

  Ingestion
    orchestrate.py   - the entrypoint: search 3 platforms, dedup/age/sort, run pipeline
    fetch.py         - fetch_from_rss(): read the live Reddit Atom feed
    fediverse/       - platform adapters (Lemmy/PieFed/Reddit) behind one interface
    fediverse_mcp.py - read-only MCP server (search_posts / get_thread)

  Pipeline (per post)
    classify.py  - keyword prefilter + LLM antisemitic-trope classifier
    core.py      - re-exports the generation entrypoint from agents.generate

  Post-generation (notify + persist)
    notify.py    - message shape + send via Discord (the only outbound write)
    tracking.py  - SQLite: dedup ledger + responses store
"""
