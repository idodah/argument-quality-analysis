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


# Both source corpora are fetched on first import (multi-GB, cached on disk).
# Importing this module — e.g. via preprocess.py — will trigger the downloads
# if the directories are absent.
if not os.path.exists("./Webis-CMV-20"):
    zenodo_download("3778298", output_dir="./Webis-CMV-20")

if not os.path.exists("./winning-args-corpus/winning-args-corpus"):
    convokit_download("winning-args-corpus", data_dir="./winning-args-corpus")


def _valid_body(body: str | None) -> bool:
    """True if the comment body is real text (not empty or a Reddit deletion placeholder)."""
    return bool(body) and body not in ("[deleted]", "[removed]")


def _match_score(delta_len: int, delta_score: float, cand_len: int, cand_score: float) -> float:
    """Weighted log-distance between a delta comment and a non-delta candidate on
    length and Reddit score (lower = better match), used to pick per-delta counterparts
    that differ in persuasive quality."""
    len_dist = abs(math.log1p(delta_len) - math.log1p(cand_len))
    score_dist = abs(math.log1p(max(delta_score, 0)) - math.log1p(max(cand_score, 0)))
    return LENGTH_WEIGHT * len_dist + SCORE_WEIGHT * score_dist


def _get_delta_comment(comment_entry, submission_id):
    """Return the top-level, non-deleted reply to ``submission_id`` that earned a
    delta; deeper replies are ignored since they don't rebut the OP
    directly."""
    for c in comment_entry.get("comments", []):
        if c.get("delta") is True and c.get("level") == 0 and c.get("parent_id") == submission_id:
            if _valid_body(c.get("body", "")):
                return c
    return None


def _closest_nodelta(delta_body: str, delta_score: float, candidates: list[tuple[str, float]]) -> int | None:
    """Index of the candidate ``(body, score)`` closest to the delta by :func:`_match_score`.

    Candidates with an invalid body are skipped; returns ``None`` if none qualify.
    """
    delta_len = len(delta_body)
    best_idx = None
    best_dist = math.inf
    for i, (body, score) in enumerate(candidates):
        if not _valid_body(body):
            continue
        dist = _match_score(delta_len, delta_score, len(body), float(score or 0))
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def _get_matched_nodelta_comment(comment_entry, submission_id, delta_comment):
    """Pick the top-level non-delta reply whose (length, score) is closest to the delta."""
    delta_score = float(delta_comment.get("score") or 0)
    replies = [
        c for c in comment_entry.get("comments", [])
        if c.get("delta") is not True and c.get("level") == 0 and c.get("parent_id") == submission_id
    ]
    idx = _closest_nodelta(
        delta_comment.get("body", ""), delta_score,
        [(c.get("body", ""), c.get("score")) for c in replies],
    )
    return replies[idx] if idx is not None else None


def build_webis_raw() -> list[ArgumentPair]:
    """Build ``ArgumentPair`` rows from Webis-CMV-20, keeping one delta plus its
    closest non-delta top-level reply per submission and skipping any thread
    where the OP or either side is missing or deleted."""
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


def _load_winning_corpus():
    """Load the winning-args-corpus and derive what :func:`build_winning_raw` needs.

    Returns ``(utterances, conversations, pair_groups, thread_date)`` where
    ``pair_groups`` maps each ``pair_id`` to the utterance ids the corpus authors
    paired up, and ``thread_date`` maps each thread root to the date of its
    earliest reply (both computed in a single pass over the utterances).
    """
    utterances: dict[str, dict] = {}
    with open(WINNING_DIR / "utterances.jsonl") as f:
        for line in f:
            u = json.loads(line)
            utterances[u["id"]] = u

    with open(WINNING_DIR / "conversations.json") as f:
        conversations = json.load(f)

    pair_groups: dict[str, list[str]] = defaultdict(list)
    thread_min_ts: dict[str, int] = {}
    for uid, u in utterances.items():
        for pid in (u["meta"].get("pair_ids") or []):
            pair_groups[pid].append(uid)
        ts = u.get("timestamp")
        if ts and u["root"] != u["id"]:
            root = u["root"]
            if root not in thread_min_ts or ts < thread_min_ts[root]:
                thread_min_ts[root] = ts

    thread_date = {}
    for root, ts in thread_min_ts.items():
        try:
            thread_date[root] = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        except (ValueError, TypeError):
            thread_date[root] = None

    return utterances, conversations, pair_groups, thread_date


def build_winning_raw() -> list[ArgumentPair]:
    """Build ``ArgumentPair`` rows from the ConvoKit winning-args-corpus, emitting
    one pair per successful direct reply matched 1-to-1 with its closest
    unsuccessful sibling."""
    utterances, conversations, pair_groups, thread_date = _load_winning_corpus()

    rows = []
    for uids in pair_groups.values():
        uids = [uid for uid in uids if uid in utterances]
        if not uids:
            continue
        root_id = utterances[uids[0]]["root"]
        conv = conversations.get(root_id, {})
        title = conv.get("op-title", "")
        op_text = conv.get("op-text-body", "")

        if not title or not op_text:
            continue

        date = thread_date.get(root_id)

        direct_replies = [uid for uid in uids if utterances[uid]["reply-to"] == utterances[uid]["root"]]
        delta_uids = [uid for uid in direct_replies if utterances[uid]["meta"].get("success") == 1]
        nodelta_uids = [uid for uid in direct_replies if utterances[uid]["meta"].get("success") == 0]

        # Each delta argument is matched to its single closest nodelta argument
        # by (length, score) — avoids the cross-product blow-up of the old code,
        # which paired every delta with every nodelta and biased the data.
        nodelta_us = [utterances[uid] for uid in nodelta_uids]
        candidates = [(u["text"], u["meta"].get("score")) for u in nodelta_us]
        for d_uid in delta_uids:
            delta_u = utterances[d_uid]
            delta_body = delta_u["text"]
            if not _valid_body(delta_body):
                continue
            delta_score = float(delta_u["meta"].get("score") or 0)

            idx = _closest_nodelta(delta_body, delta_score, candidates)
            if idx is None:
                continue

            rows.append(ArgumentPair(
                thread_id=root_id,
                topic=title,
                original_post=op_text,
                delta_argument=delta_body,
                nodelta_argument=nodelta_us[idx]["text"],
                date=date,
            ))
    return rows


def main() -> None:
    """Build the raw pair-wise dataset from both sources and write it to CSV."""
    pairs = build_webis_raw() + build_winning_raw()
    df = pd.DataFrame([p.model_dump() for p in pairs])
    df = df[df["delta_argument"] != df["nodelta_argument"]]
    df.to_csv("data/argument_quality_dataset.csv", index=False)
    print(f"Dataset saved: {len(df)} rows, {df['thread_id'].nunique()} unique threads")


if __name__ == "__main__":
    main()
