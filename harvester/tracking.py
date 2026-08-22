"""Dedup ledger + record of generated counter-arguments.

Two responsibilities, exposed as three module-level functions the orchestrator
calls — `mark_seen`, `is_seen`, `record`:
  - `seen` ledger (`mark_seen` / `is_seen`): claim each examined post by canonical
    id so it is never answered twice (the orchestrator's dedup guarantee).
  - `responses` (`record`): persist each generated rebuttal + quality flags.

Two interchangeable backends behind that one API, chosen by environment:
  - **DynamoDB** (deployed on AWS): set `DDB_SEEN_TABLE` and `DDB_RESPONSES_TABLE`.
    `mark_seen` is a conditional `PutItem` (`attribute_not_exists`) — the same
    atomic, race-free claim the SQLite PRIMARY KEY gave us.
  - **SQLite** (local / offline / tests): no env needed; one file at
    `HARVESTER_DB` (default `harvester_tracking.db`). This is the default so the
    test suite and local runs need no AWS.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

from botocore.exceptions import ClientError

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def _use_dynamo() -> bool:
    return bool(os.environ.get("DDB_SEEN_TABLE") and os.environ.get("DDB_RESPONSES_TABLE"))


# ---------------------------------------------------------------------------
# SQLite backend (default)
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    id              TEXT PRIMARY KEY,         -- reddit post id (dedup key)
    title           TEXT NOT NULL,
    topic           TEXT,                     -- the CMV topic (= the post title)
    url             TEXT,
    original_post   TEXT,                     -- the trope-advancing post body we replied to
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


def _row_from_result(result: dict) -> dict:
    """Normalize a generation result into the column set both backends store."""
    title = result.get("title") or ""
    return {
        "id": str(result.get("id") or ""),
        "title": title,
        "topic": result.get("topic") or title,   # topic defaults to the CMV title
        "url": result.get("url") or "",
        "original_post": result.get("original_post") or result.get("body") or "",
        "generation": result.get("generation") or "",
        "sources": result.get("sources") or [],
        "grounded": bool(result.get("grounded", True)),
        "pro_israel": bool(result.get("pro_israel_reply", True)),
        "stance": result.get("stance") or "",
        "gave_up": bool(result.get("gave_up", False)),
        "created_at": time.time(),
    }


def _sqlite_record(result: dict, db_path: Path | None = None) -> bool:
    conn = _connect(db_path)
    try:
        r = _row_from_result(result)
        conn.execute(
            """INSERT INTO responses
               (id, title, topic, url, original_post, generation, sources,
                grounded, pro_israel, stance, gave_up, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                r["id"], r["title"], r["topic"], r["url"], r["original_post"],
                r["generation"], json.dumps(r["sources"]),
                int(r["grounded"]), int(r["pro_israel"]), r["stance"],
                int(r["gave_up"]), r["created_at"],
            ),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # duplicate id
    finally:
        conn.close()


def _sqlite_mark_seen(post_id: str, db_path: Path | None = None) -> bool:
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


def _sqlite_is_seen(post_id: str, db_path: Path | None = None) -> bool:
    conn = _connect(db_path)
    try:
        return conn.execute("SELECT 1 FROM seen WHERE id = ?", (post_id,)).fetchone() is not None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# DynamoDB backend
# ---------------------------------------------------------------------------

_DDB_TABLES: dict = {}


def _ddb_table(name_env: str):
    """Cached DynamoDB Table handle for the table named by env var `name_env`
    (cached because boto3 client construction is too costly to repeat per call)."""
    table_name = os.environ[name_env]
    if table_name not in _DDB_TABLES:
        import boto3

        region = os.environ.get("DDB_REGION") or os.environ.get("AWS_REGION")
        _DDB_TABLES[table_name] = boto3.resource("dynamodb", region_name=region).Table(table_name)
    return _DDB_TABLES[table_name]


def _ddb_record(result: dict) -> bool:
    r = _row_from_result(result)
    # Store sources as a JSON string for parity with SQLite; booleans map to DDB
    # BOOL, created_at to a Number.
    item = {
        "id": r["id"], "title": r["title"], "topic": r["topic"], "url": r["url"],
        "original_post": r["original_post"], "generation": r["generation"],
        "sources": json.dumps(r["sources"]),
        "grounded": r["grounded"], "pro_israel": r["pro_israel"],
        "stance": r["stance"], "gave_up": r["gave_up"],
        "created_at": int(r["created_at"]),
    }
    try:
        _ddb_table("DDB_RESPONSES_TABLE").put_item(
            Item=item,
            ConditionExpression="attribute_not_exists(id)",  # idempotent like SQLite
        )
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return False  # duplicate id
        raise


def _ddb_mark_seen(post_id: str) -> bool:
    """Atomic claim via conditional PutItem (fails if the id exists), so exactly
    one concurrent caller gets True — the DynamoDB analogue of the SQLite
    PRIMARY KEY guarantee."""

    try:
        _ddb_table("DDB_SEEN_TABLE").put_item(
            Item={"id": post_id, "first_seen_at": int(time.time())},
            ConditionExpression="attribute_not_exists(id)",
        )
        return True  # claim succeeded (first sight)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return False  # already seen
        raise


def _ddb_is_seen(post_id: str) -> bool:
    resp = _ddb_table("DDB_SEEN_TABLE").get_item(Key={"id": post_id})
    return "Item" in resp


# ---------------------------------------------------------------------------
# Public API — dispatches to the active backend
# ---------------------------------------------------------------------------

def record(result: dict, db_path: Path | None = None) -> bool:
    """Insert one generated result (the generation dict augmented with
    id/title/url/original_post). Returns True if inserted, False if the id was
    already present — idempotent, so safe on re-delivered posts."""
    if _use_dynamo():
        return _ddb_record(result)
    return _sqlite_record(result, db_path)


def mark_seen(post_id: str, db_path: Path | None = None) -> bool:
    """Atomically claim a post id in the `seen` ledger. Returns True on first
    sight (caller should process it), False if already seen (skip it). Race-free
    on both backends, so a post is never processed twice even under concurrent
    polls."""
    if not post_id:
        return False
    if _use_dynamo():
        return _ddb_mark_seen(post_id)
    return _sqlite_mark_seen(post_id, db_path)


def is_seen(post_id: str, db_path: Path | None = None) -> bool:
    """Read-only check: has this id been examined before? (Does NOT claim it.)
    Used by dry-runs, which must not consume the ledger."""
    if not post_id:
        return False
    if _use_dynamo():
        return _ddb_is_seen(post_id)
    return _sqlite_is_seen(post_id, db_path)
