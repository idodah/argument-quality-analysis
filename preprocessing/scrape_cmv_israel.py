"""
Scrape Israel-related CMV threads from arctic-shift (Pushshift successor).

Pulls submissions from r/changemyview where the title matches an Israel-related
keyword, fetches their comment trees, identifies delta-awarded top-level
comments. In RAG mode (default), emits one RagArgument record per delta
comment. In --pairs mode, also picks a matched non-delta sibling and emits
ArgumentPair records (legacy ranker-training format).

Output:
    RAG:    data/cmv_israel_rag.parquet  /  .jsonl
    Pairs:  data/cmv_israel.parquet      /  .jsonl

Usage:
    python -m preprocessing.scrape_cmv_israel             # RAG mode, push to Hub
    python -m preprocessing.scrape_cmv_israel --pairs     # legacy pair-wise mode
    python -m preprocessing.scrape_cmv_israel --smoke     # 5 threads, dump raw JSON
    python -m preprocessing.scrape_cmv_israel --no-hub    # skip Hub upload
"""

import argparse
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from datasets import Dataset, Features, Value
from dotenv import load_dotenv

from models.data import HF_REPO_ID
from schemas import ArgumentPair, RagArgument

ARCTIC_BASE = "https://arctic-shift.photon-reddit.com/api"
SUBREDDIT = "changemyview"
START_DATE = datetime(2023, 10, 1, tzinfo=timezone.utc)
END_DATE = datetime.now(timezone.utc)

KEYWORDS = [
    r"\bisrael\w*\b",
    r"\bpalestin\w*\b",
    r"\bgaza\w*\b",
    r"\bhamas\b",
    r"\bzionis\w*\b",
    r"\bidf\b",
    r"\bwest bank\b",
    r"\bintifada\b",
    r"\bjewish state\b",
    r"\bnetanyahu\b",
    r"\bhezbollah\b",
    r"\boctober 7\b",
]
KEYWORD_RE = re.compile("|".join(KEYWORDS), re.IGNORECASE)

LENGTH_WEIGHT = 1.0
SCORE_WEIGHT = 0.5

NODELTA_MIN_BODY_CHARS = 100
NODELTA_MIN_SCORE = 1

PAGE_LIMIT = 100
COMMENT_PAGE_LIMIT = 100
REQUEST_TIMEOUT = 60
SLEEP_BETWEEN = 1.0

OUT_DIR = Path("data")
OUT_PARQUET = OUT_DIR / "cmv_israel.parquet"
OUT_JSONL = OUT_DIR / "cmv_israel.jsonl"
OUT_RAG_PARQUET = OUT_DIR / "cmv_israel_rag.parquet"
OUT_RAG_JSONL = OUT_DIR / "cmv_israel_rag.jsonl"
SMOKE_DUMP = OUT_DIR / "cmv_israel_smoke.json"

HUB_SPLIT = "israel_2023_plus"
HUB_FEATURES = Features({
    "thread_id": Value("large_string"),
    "topic": Value("large_string"),
    "original_post": Value("large_string"),
    "delta_argument": Value("large_string"),
    "nodelta_argument": Value("large_string"),
    "date": Value("date32"),
})

HUB_RAG_SPLIT = "israel_2023_plus_rag"
HUB_RAG_FEATURES = Features({
    "thread_id": Value("large_string"),
    "comment_id": Value("large_string"),
    "topic": Value("large_string"),
    "original_post": Value("large_string"),
    "argument": Value("large_string"),
    "score": Value("int32"),
    "date": Value("date32"),
})


def _get(path: str, params: dict) -> dict:
    for attempt in range(5):
        try:
            r = requests.get(f"{ARCTIC_BASE}{path}", params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    return {}


def fetch_submissions() -> list[dict]:
    """Page through submissions in the date window, keep title-matched ones."""
    matched = []
    after = int(START_DATE.timestamp())
    end_ts = int(END_DATE.timestamp())
    total_seen = 0

    while after < end_ts:
        data = _get("/posts/search", {
            "subreddit": SUBREDDIT,
            "after": after,
            "before": end_ts,
            "limit": PAGE_LIMIT,
            "sort": "asc",
            "sort_type": "created_utc",
        })
        rows = data.get("data") or []
        if not rows:
            break

        for sub in rows:
            total_seen += 1
            title = sub.get("title") or ""
            if KEYWORD_RE.search(title):
                matched.append(sub)

        after = int(rows[-1]["created_utc"]) + 1
        print(f"  paged through {total_seen} submissions, kept {len(matched)} (now at {datetime.fromtimestamp(after, timezone.utc).date()})")
        time.sleep(SLEEP_BETWEEN)

    return matched


def fetch_comments(submission_id: str) -> list[dict]:
    """
    Returns a flat list of comments for the submission, each annotated with a
    `_children` list of direct replies (so we can walk delta-bot confirmations).
    """
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
    """
    A comment earned a delta if any DeltaBot 'Confirmed: ...' message exists
    whose parent is a descendant of this comment. (OP / awarder posts a reply
    with '!delta' or 'Δ', DeltaBot then confirms with parent_id pointing to
    that awarder reply — which is a child of the comment that actually
    earned the delta.)
    """
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
    parent = comment.get("parent_id") or ""
    return parent == f"t3_{submission_id}"


def _match_score(d_len, d_score, c_len, c_score) -> float:
    len_dist = abs(math.log1p(d_len) - math.log1p(c_len))
    score_dist = abs(math.log1p(max(d_score, 0)) - math.log1p(max(c_score, 0)))
    return LENGTH_WEIGHT * len_dist + SCORE_WEIGHT * score_dist


def build_rag_records(submission: dict, comments: list[dict]) -> list[RagArgument]:
    """Emit one RagArgument per delta-awarded top-level comment."""
    sid = submission["id"]
    top_level = [c for c in comments if _is_top_level(c, sid) and _valid_body(c.get("body"))]
    deltas = [c for c in top_level if _has_delta(c)]
    if not deltas:
        return []

    created = submission.get("created_utc")
    sub_date = datetime.fromtimestamp(created, timezone.utc).date() if created else None
    out = []
    for d in deltas:
        out.append(RagArgument(
            thread_id=sid,
            comment_id=d["id"],
            topic=submission.get("title", ""),
            original_post=submission.get("selftext", "") or "",
            argument=d["body"],
            score=int(d.get("score") or 0),
            date=sub_date,
        ))
    return out


def build_pairs(submission: dict, comments: list[dict]) -> list[ArgumentPair]:
    sid = submission["id"]
    top_level = [c for c in comments if _is_top_level(c, sid) and _valid_body(c.get("body"))]
    if not top_level:
        return []

    deltas = [c for c in top_level if _has_delta(c)]
    nodeltas = [
        c for c in top_level
        if not _has_delta(c)
        and c.get("author") != submission.get("author")
        and len(c.get("body") or "") >= NODELTA_MIN_BODY_CHARS
        and float(c.get("score") or 0) >= NODELTA_MIN_SCORE
    ]
    if not deltas or not nodeltas:
        return []

    used_nodelta_ids = set()
    pairs = []
    for d in deltas:
        d_len = len(d["body"])
        d_score = float(d.get("score") or 0)
        best = None
        best_dist = math.inf
        for c in nodeltas:
            if c["id"] in used_nodelta_ids:
                continue
            dist = _match_score(d_len, d_score, len(c["body"]), float(c.get("score") or 0))
            if dist < best_dist:
                best_dist = dist
                best = c
        if best is None:
            continue
        used_nodelta_ids.add(best["id"])

        created = submission.get("created_utc")
        pairs.append(ArgumentPair(
            thread_id=sid,
            topic=submission.get("title", ""),
            original_post=submission.get("selftext", "") or "",
            delta_argument=d["body"],
            nodelta_argument=best["body"],
            date=datetime.fromtimestamp(created, timezone.utc).date() if created else None,
        ))
    return pairs


def smoke_test(n: int = 5) -> None:
    """Fetch n matched threads, dump raw submission + comment JSON for inspection."""
    OUT_DIR.mkdir(exist_ok=True)
    print(f"[smoke] Fetching first {n} Israel-matched submissions...")

    matched = []
    after = int(START_DATE.timestamp())
    end_ts = int(END_DATE.timestamp())
    while after < end_ts and len(matched) < n:
        data = _get("/posts/search", {
            "subreddit": SUBREDDIT,
            "after": after,
            "before": end_ts,
            "limit": PAGE_LIMIT,
            "sort": "asc",
            "sort_type": "created_utc",
        })
        rows = data.get("data") or []
        if not rows:
            break
        for sub in rows:
            if KEYWORD_RE.search(sub.get("title") or ""):
                matched.append(sub)
                if len(matched) >= n:
                    break
        after = int(rows[-1]["created_utc"]) + 1
        time.sleep(SLEEP_BETWEEN)

    dump = []
    for sub in matched:
        comments = fetch_comments(sub["id"])
        pairs = build_pairs(sub, comments)
        dump.append({
            "submission_id": sub["id"],
            "title": sub.get("title"),
            "date": datetime.fromtimestamp(sub["created_utc"], timezone.utc).isoformat(),
            "n_comments": len(comments),
            "n_top_level": sum(1 for c in comments if _is_top_level(c, sub["id"])),
            "n_pairs_built": len(pairs),
            "submission": sub,
            "comments": comments,
        })
        time.sleep(SLEEP_BETWEEN)

    SMOKE_DUMP.write_text(json.dumps(dump, indent=2, default=str))
    print(f"[smoke] Wrote {SMOKE_DUMP}")
    for d in dump:
        print(f"  {d['submission_id']}  comments={d['n_comments']}  top_level={d['n_top_level']}  pairs={d['n_pairs_built']}  | {d['title'][:80]}")


def main(mode: str = "rag", push: bool = True) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Fetching CMV submissions {START_DATE.date()} -> {END_DATE.date()} (mode={mode})")
    submissions = fetch_submissions()
    print(f"\nMatched {len(submissions)} Israel-related submissions. Fetching comment trees...")

    records: list = []
    for i, sub in enumerate(submissions, 1):
        try:
            comments = fetch_comments(sub["id"])
            if mode == "pairs":
                records.extend(build_pairs(sub, comments))
            else:
                records.extend(build_rag_records(sub, comments))
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

    if mode == "pairs":
        df.to_parquet(OUT_PARQUET, index=False)
        with OUT_JSONL.open("w") as f:
            for r in records:
                f.write(r.model_dump_json() + "\n")
        print(f"\nSaved {len(df)} pairs to {OUT_PARQUET} and {OUT_JSONL}.")
    else:
        df.to_parquet(OUT_RAG_PARQUET, index=False)
        with OUT_RAG_JSONL.open("w") as f:
            for r in records:
                f.write(r.model_dump_json() + "\n")
        print(f"\nSaved {len(df)} RAG records to {OUT_RAG_PARQUET} and {OUT_RAG_JSONL}.")

    if push:
        push_to_hub(df, mode=mode)


def push_to_hub(df: pd.DataFrame, mode: str = "rag") -> None:
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN not set; skipping Hub upload.")
        return

    hub_df = df.copy()
    hub_df["date"] = pd.to_datetime(hub_df["date"]).dt.date
    if mode == "pairs":
        features, split = HUB_FEATURES, HUB_SPLIT
    else:
        features, split = HUB_RAG_FEATURES, HUB_RAG_SPLIT
    ds = Dataset.from_pandas(hub_df, preserve_index=False, features=features)
    ds.push_to_hub(HF_REPO_ID, token=hf_token, split=split, private=False)
    print(f"Pushed {len(ds)} rows to https://huggingface.co/datasets/{HF_REPO_ID} (split='{split}').")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Fetch 5 threads, dump raw JSON")
    parser.add_argument("--pairs", action="store_true", help="Emit ArgumentPair records (legacy mode)")
    parser.add_argument("--no-hub", action="store_true", help="Skip pushing to Hugging Face Hub")
    args = parser.parse_args()
    if args.smoke:
        smoke_test()
    else:
        main(mode="pairs" if args.pairs else "rag", push=not args.no_hub)