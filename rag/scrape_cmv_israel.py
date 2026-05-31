"""
Scrape Israel-related CMV threads from arctic-shift (Pushshift successor).

Pulls submissions from r/changemyview where the title matches an Israel-related
keyword, fetches their comment trees, identifies delta-awarded top-level
comments, and emits one RagArgument record per delta comment.

Output (local; the downstream rag.classify_stance step reads the parquet):
    data/cmv_israel_rag.parquet  /  .jsonl

Usage:
    python -m rag.scrape_cmv_israel             # scrape -> parquet + jsonl
    python -m rag.scrape_cmv_israel --smoke     # 5 threads, dump raw JSON
"""

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from schemas import RagArgument

ARCTIC_BASE = "https://arctic-shift.photon-reddit.com/api"
SUBREDDIT = "changemyview"
START_DATE = datetime(2023, 10, 1, tzinfo=timezone.utc)
END_DATE = datetime.now(timezone.utc)

KEYWORDS = [
    r"\bisrael\w*\b", r"\bpalestin\w*\b", r"\bgaza\w*\b", r"\bhamas\b",
    r"\bzionis\w*\b", r"\bidf\b", r"\bwest bank\b", r"\bintifada\b",
    r"\bjewish state\b", r"\bnetanyahu\b", r"\bhezbollah\b", r"\boctober 7\b",
]
KEYWORD_RE = re.compile("|".join(KEYWORDS), re.IGNORECASE)

PAGE_LIMIT = 100
COMMENT_PAGE_LIMIT = 100
REQUEST_TIMEOUT = 60
SLEEP_BETWEEN = 1.0

OUT_DIR = Path("data")
OUT_RAG_PARQUET = OUT_DIR / "cmv_israel_rag.parquet"
OUT_RAG_JSONL = OUT_DIR / "cmv_israel_rag.jsonl"
SMOKE_DUMP = OUT_DIR / "cmv_israel_smoke.json"


def _get(path: str, params: dict) -> dict:
    for attempt in range(5):
        try:
            r = requests.get(f"{ARCTIC_BASE}{path}", params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    return {}


def fetch_submissions(max_matched: int | None = None) -> list[dict]:
    """Page through the date window, keeping title-matched submissions.

    `max_matched` stops early once that many matches are collected (used by the
    smoke test); None pages through the whole window.
    """
    matched: list[dict] = []
    after = int(START_DATE.timestamp())
    end_ts = int(END_DATE.timestamp())
    total_seen = 0

    while after < end_ts:
        data = _get("/posts/search", {
            "subreddit": SUBREDDIT, "after": after, "before": end_ts,
            "limit": PAGE_LIMIT, "sort": "asc", "sort_type": "created_utc",
        })
        rows = data.get("data") or []
        if not rows:
            break
        for sub in rows:
            total_seen += 1
            if KEYWORD_RE.search(sub.get("title") or ""):
                matched.append(sub)
                if max_matched is not None and len(matched) >= max_matched:
                    return matched
        after = int(rows[-1]["created_utc"]) + 1
        at = datetime.fromtimestamp(after, timezone.utc).date()
        print(f"  paged {total_seen} submissions, kept {len(matched)} (now at {at})")
        time.sleep(SLEEP_BETWEEN)

    return matched


def fetch_comments(submission_id: str) -> list[dict]:
    """Flat list of the submission's comments, each annotated with a `_children`
    list of direct replies (so we can walk delta-bot confirmations)."""
    all_rows: list[dict] = []
    after: int | None = None
    while True:
        params = {
            "link_id": submission_id,
            "limit": COMMENT_PAGE_LIMIT,
            "sort": "asc",
            "sort_type": "created_utc",
        }
        if after is not None:
            params["after"] = after
        data = _get("/comments/search", params)
        rows = data.get("data") or []
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < COMMENT_PAGE_LIMIT:
            break
        after = int(rows[-1]["created_utc"]) + 1

    by_name = {f"t1_{c['id']}": c for c in all_rows}
    for c in all_rows:
        c["_children"] = []
    for c in all_rows:
        parent = c.get("parent_id") or ""
        if parent in by_name:
            by_name[parent]["_children"].append(c)
    return all_rows


def _valid_body(body: str | None) -> bool:
    return bool(body) and body not in ("[deleted]", "[removed]")


def _is_delta_confirmation(comment: dict) -> bool:
    if comment.get("author") != "DeltaBot":
        return False
    body = (comment.get("body") or "").lower()
    return "confirmed:" in body and "delta awarded" in body


def _has_delta(comment: dict) -> bool:
    """True if a DeltaBot 'Confirmed: ...' message appears within two levels
    below this comment (OP replies '!delta'/'Δ'; DeltaBot confirms on that
    reply, which is a child of the comment that earned the delta)."""
    if comment.get("author") == "DeltaBot":
        return False
    for reply in comment.get("_children", []) or []:
        if _is_delta_confirmation(reply):
            return True
        for grandchild in reply.get("_children", []) or []:
            if _is_delta_confirmation(grandchild):
                return True
    return False


def _is_top_level(comment: dict, submission_id: str) -> bool:
    return (comment.get("parent_id") or "") == f"t3_{submission_id}"


def build_rag_records(submission: dict, comments: list[dict]) -> list[RagArgument]:
    """Emit one RagArgument per delta-awarded top-level comment."""
    sid = submission["id"]
    top_level = [c for c in comments if _is_top_level(c, sid) and _valid_body(c.get("body"))]
    deltas = [c for c in top_level if _has_delta(c)]
    if not deltas:
        return []

    created = submission.get("created_utc")
    sub_date = datetime.fromtimestamp(created, timezone.utc).date() if created else None
    return [
        RagArgument(
            thread_id=sid,
            comment_id=d["id"],
            topic=submission.get("title", ""),
            original_post=submission.get("selftext", "") or "",
            argument=d["body"],
            score=int(d.get("score") or 0),
            date=sub_date,
        )
        for d in deltas
    ]


def smoke_test(n: int = 5) -> None:
    """Fetch n matched threads, dump raw submission + comment JSON for inspection."""
    OUT_DIR.mkdir(exist_ok=True)
    print(f"[smoke] Fetching first {n} Israel-matched submissions...")
    matched = fetch_submissions(max_matched=n)

    dump = []
    for sub in matched:
        comments = fetch_comments(sub["id"])
        records = build_rag_records(sub, comments)
        dump.append({
            "submission_id": sub["id"],
            "title": sub.get("title"),
            "date": datetime.fromtimestamp(sub["created_utc"], timezone.utc).isoformat(),
            "n_comments": len(comments),
            "n_top_level": sum(1 for c in comments if _is_top_level(c, sub["id"])),
            "n_rag_records": len(records),
            "submission": sub,
            "comments": comments,
        })
        time.sleep(SLEEP_BETWEEN)

    SMOKE_DUMP.write_text(json.dumps(dump, indent=2, default=str))
    print(f"[smoke] Wrote {SMOKE_DUMP}")
    for d in dump:
        print(f"  {d['submission_id']}  comments={d['n_comments']}  rag={d['n_rag_records']}"
              f"  | {d['title'][:80]}")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Fetching CMV submissions {START_DATE.date()} -> {END_DATE.date()}")
    submissions = fetch_submissions()
    print(f"\nMatched {len(submissions)} Israel-related submissions. Fetching comment trees...")

    records: list[RagArgument] = []
    for i, sub in enumerate(submissions, 1):
        try:
            records.extend(build_rag_records(sub, fetch_comments(sub["id"])))
        except Exception as e:
            print(f"  [skip {sub.get('id')}] {e}")
        if i % 10 == 0:
            print(f"  processed {i}/{len(submissions)} threads, {len(records)} records so far")
        time.sleep(SLEEP_BETWEEN)

    if not records:
        print("No records extracted.")
        return

    df = pd.DataFrame([r.model_dump() for r in records])
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(OUT_RAG_PARQUET, index=False)
    OUT_RAG_JSONL.write_text("".join(r.model_dump_json() + "\n" for r in records))
    print(f"\nSaved {len(df)} RAG records to {OUT_RAG_PARQUET} and {OUT_RAG_JSONL}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Fetch 5 threads, dump raw JSON")
    args = parser.parse_args()
    if args.smoke:
        smoke_test()
    else:
        main()
