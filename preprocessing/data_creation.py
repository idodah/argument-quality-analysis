"""Build the unified pair-wise dataset from Webis-CMV-20 and the
winning-args-corpus into ArgumentPair records."""

import os
import json
import bz2
import math
import pandas as pd
from collections import defaultdict
from convokit import download as convokit_download
from zenodo_get import download as zenodo_download
from datetime import datetime, timezone
from pathlib import Path

from schemas import ArgumentPair

WEBIS_DIR = Path("./Webis-CMV-20")
WINNING_DIR = Path("./winning-args-corpus/winning-args-corpus")

LENGTH_WEIGHT = 1.0
SCORE_WEIGHT = 0.5


if not os.path.exists("./Webis-CMV-20"):
    zenodo_download("3778298", output_dir="./Webis-CMV-20")

if not os.path.exists("./winning-args-corpus/winning-args-corpus"):
    convokit_download("winning-args-corpus", data_dir="./winning-args-corpus")


def _valid_body(body: str | None) -> bool:
    return bool(body) and body not in ("[deleted]", "[removed]")


def _match_score(delta_len: int, delta_score: float, cand_len: int, cand_score: float) -> float:
    """Lower is better. Combines log-length distance and log-score distance."""
    len_dist = abs(math.log1p(delta_len) - math.log1p(cand_len))
    score_dist = abs(math.log1p(max(delta_score, 0)) - math.log1p(max(cand_score, 0)))
    return LENGTH_WEIGHT * len_dist + SCORE_WEIGHT * score_dist


def _get_delta_comment(comment_entry, submission_id):
    for c in comment_entry.get("comments", []):
        if c.get("delta") is True and c.get("level") == 0 and c.get("parent_id") == submission_id:
            if _valid_body(c.get("body", "")):
                return c
    return None


def _get_matched_nodelta_comment(comment_entry, submission_id, delta_comment):
    """Pick the top-level non-delta reply whose (length, score) is closest to the delta."""
    delta_len = len(delta_comment.get("body", ""))
    delta_score = float(delta_comment.get("score") or 0)

    best = None
    best_dist = math.inf
    for c in comment_entry.get("comments", []):
        if c.get("delta") is True:
            continue
        if c.get("level") != 0 or c.get("parent_id") != submission_id:
            continue
        if not _valid_body(c.get("body", "")):
            continue
        cand_len = len(c["body"])
        cand_score = float(c.get("score") or 0)
        dist = _match_score(delta_len, delta_score, cand_len, cand_score)
        if dist < best_dist:
            best_dist = dist
            best = c
    return best


def build_webis_raw() -> list[ArgumentPair]:
    rows = []
    with bz2.open(WEBIS_DIR / "pairs.jsonl.bz2", "rt") as f:
        for line in f:
            pair = json.loads(line)
            submission = pair["submission"]
            title = submission.get("title", "")
            selftext = submission.get("selftext", "")
            created_utc = submission.get("created_utc")
            submission_id = pair["submission_id"]

            if not title or not selftext or selftext in ("[deleted]", "[removed]", ""):
                continue

            try:
                date = datetime.fromtimestamp(int(created_utc), tz=timezone.utc).date() if created_utc else None
            except (ValueError, TypeError):
                date = None

            delta_c = _get_delta_comment(pair["delta_comment"], submission_id)
            if not delta_c:
                continue
            nodelta_c = _get_matched_nodelta_comment(pair["nodelta_comment"], submission_id, delta_c)
            if not nodelta_c:
                continue

            rows.append(ArgumentPair(
                thread_id=submission_id,
                topic=title,
                original_post=selftext,
                delta_argument=delta_c["body"],
                nodelta_argument=nodelta_c["body"],
                date=date,
            ))
    return rows


def build_winning_raw() -> list[ArgumentPair]:
    utterances: dict[str, dict] = {}
    with open(WINNING_DIR / "utterances.jsonl") as f:
        for line in f:
            u = json.loads(line)
            utterances[u["id"]] = u

    with open(WINNING_DIR / "conversations.json") as f:
        conversations = json.load(f)

    thread_min_ts: dict[str, int | None] = defaultdict(lambda: None)
    for u in utterances.values():
        ts = u.get("timestamp")
        if ts and u["root"] != u["id"]:
            root = u["root"]
            if thread_min_ts[root] is None or ts < thread_min_ts[root]:
                thread_min_ts[root] = ts

    pairs: dict[str, list[str]] = defaultdict(list)
    for uid, u in utterances.items():
        for pid in (u["meta"].get("pair_ids") or []):
            pairs[pid].append(uid)

    rows = []
    for pid, uids in pairs.items():
        uids = [uid for uid in uids if uid in utterances]
        if not uids:
            continue
        root_id = utterances[uids[0]]["root"]
        conv = conversations.get(root_id, {})
        title = conv.get("op-title", "")
        op_text = conv.get("op-text-body", "")

        if not title or not op_text:
            continue

        min_ts = thread_min_ts[root_id]
        date = None
        if min_ts:
            try:
                date = datetime.fromtimestamp(min_ts, tz=timezone.utc).date()
            except (ValueError, TypeError):
                pass

        direct_replies = [uid for uid in uids if utterances[uid]["reply-to"] == utterances[uid]["root"]]
        delta_uids = [uid for uid in direct_replies if utterances[uid]["meta"].get("success") == 1]
        nodelta_uids = [uid for uid in direct_replies if utterances[uid]["meta"].get("success") == 0]

        # Each delta argument is matched to its single closest nodelta argument
        # by (length, score) — avoids the cross-product blow-up of the old code,
        # which paired every delta with every nodelta and biased the data.
        for d_uid in delta_uids:
            delta_u = utterances[d_uid]
            delta_body = delta_u["text"]
            if not _valid_body(delta_body):
                continue
            delta_len = len(delta_body)
            delta_score = float(delta_u["meta"].get("score") or 0)

            best_nd = None
            best_dist = math.inf
            for nd_uid in nodelta_uids:
                nd_u = utterances[nd_uid]
                nd_body = nd_u["text"]
                if not _valid_body(nd_body):
                    continue
                cand_len = len(nd_body)
                cand_score = float(nd_u["meta"].get("score") or 0)
                dist = _match_score(delta_len, delta_score, cand_len, cand_score)
                if dist < best_dist:
                    best_dist = dist
                    best_nd = nd_u

            if best_nd is None:
                continue

            rows.append(ArgumentPair(
                thread_id=root_id,
                topic=title,
                original_post=op_text,
                delta_argument=delta_body,
                nodelta_argument=best_nd["text"],
                date=date,
            ))
    return rows


if __name__ == "__main__":
    pairs = build_webis_raw() + build_winning_raw()
    df = pd.DataFrame([p.model_dump() for p in pairs])
    df = df[df["delta_argument"] != df["nodelta_argument"]]
    df.to_csv("data/argument_quality_dataset.csv", index=False)
    print(f"Dataset saved: {len(df)} rows, {df['thread_id'].nunique()} unique threads")
