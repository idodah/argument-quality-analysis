"""SQLite store: the dedup ledger + a record of generated counter-arguments.

Two responsibilities:
  - `seen` ledger (`mark_seen` / `is_seen`): claim each examined post by canonical
    id so it is never answered twice (the orchestrator's dedup guarantee).
  - `responses` table (`record`): persist each generated rebuttal + quality flags.

SQLite on purpose: no server, no API key, one file. The access layer is a few
functions, so a different backend later (e.g. DynamoDB for AWS) means reimplementing
this one module behind the same `mark_seen` / `is_seen` / `record` API.

DB path: HARVESTER_DB env var, else `harvester_tracking.db` in the cwd.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    id              TEXT PRIMARY KEY,         -- reddit post id (dedup key)
    title           TEXT NOT NULL,
    topic           TEXT,                     -- the CMV topic (= the post title)
    url             TEXT,
    original_post   TEXT,                     -- the anti-Israel post body we replied to
    generation      TEXT NOT NULL,            -- the generated rebuttal
    sources         TEXT,                     -- JSON array of source urls
    grounded        INTEGER NOT NULL,         -- 0/1
    pro_israel      INTEGER NOT NULL,         -- 0/1
    stance          TEXT,
    gave_up         INTEGER NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL             -- epoch seconds
);
CREATE INDEX IF NOT EXISTS idx_responses_created ON responses(created_at);

-- Every post the orchestrator has EVER examined, claimed on first sight (before
-- any classify/generate work). This is the dedup ledger that guarantees a post is
-- answered at most once. Keyed on the canonical id (ActivityPub ap_id / permalink)
-- so the same federated post on two instances is deduped.
CREATE TABLE IF NOT EXISTS seen (
    id             TEXT PRIMARY KEY,        -- canonical post id
    first_seen_at  REAL NOT NULL            -- epoch seconds
);
"""


def _db_path() -> Path:
    return Path(os.environ.get("HARVESTER_DB") or "harvester_tracking.db")


def _connect(path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path or _db_path()))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def record(result: dict, db_path: Path | None = None) -> bool:
    """Insert one generated result. Returns True if inserted, False if the id was
    already present (idempotent — safe to call on re-delivered posts).

    `result` is the dict from `generate_pro_israel_response`, augmented with
    `id`/`title`/`url`/`original_post`. The CMV topic is the post title.
    """
    conn = _connect(db_path)
    try:
        title = result.get("title") or ""
        conn.execute(
            """INSERT INTO responses
               (id, title, topic, url, original_post, generation, sources,
                grounded, pro_israel, stance, gave_up, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                str(result.get("id") or ""),
                title,
                result.get("topic") or title,   # topic defaults to the CMV title
                result.get("url") or "",
                result.get("original_post") or result.get("body") or "",
                result.get("generation") or "",
                json.dumps(result.get("sources") or []),
                int(bool(result.get("grounded", True))),
                int(bool(result.get("pro_israel_reply", True))),
                result.get("stance") or "",
                int(bool(result.get("gave_up", False))),
                time.time(),
            ),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # duplicate id
    finally:
        conn.close()


def mark_seen(post_id: str, db_path: Path | None = None) -> bool:
    """Atomically claim a post id in the `seen` ledger.

    Returns True if this is the FIRST time we've seen this id (claim succeeded —
    the caller should process it), or False if it was already seen (skip it).

    `INSERT OR IGNORE` + the PRIMARY KEY make this race-free: if two runs examine
    the same id, exactly one gets True, so a post is never processed twice even
    with overlapping/concurrent polls.
    """
    if not post_id:
        return False
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "INSERT OR IGNORE INTO seen (id, first_seen_at) VALUES (?, ?)",
            (post_id, time.time()),
        )
        conn.commit()
        return cur.rowcount == 1  # 1 = inserted (new), 0 = already present
    finally:
        conn.close()


def is_seen(post_id: str, db_path: Path | None = None) -> bool:
    """Read-only check: has this id been examined before? (Does NOT claim it.)
    Used by dry-runs, which must not consume the ledger."""
    if not post_id:
        return False
    conn = _connect(db_path)
    try:
        return conn.execute("SELECT 1 FROM seen WHERE id = ?", (post_id,)).fetchone() is not None
    finally:
        conn.close()
