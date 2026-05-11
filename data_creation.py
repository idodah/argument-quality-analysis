
import os
import json
import bz2
import pandas as pd
from collections import defaultdict
from convokit import download as convokit_download
from zenodo_get import download as zenodo_download
from datetime import datetime
from pathlib import Path

from schemas import ArgumentPair

WEBIS_DIR = Path("./Webis-CMV-20")
WINNING_DIR = Path("./winning-args-corpus/winning-args-corpus")


if not os.path.exists("./Webis-CMV-20"):
    zenodo_download("3778298", output_dir="./Webis-CMV-20")

if not os.path.exists("./winning-args-corpus/winning-args-corpus"):
    convokit_download("winning-args-corpus", data_dir="./winning-args-corpus")


def _get_delta_comment(comment_entry, submission_id):
    for c in comment_entry.get("comments", []):
        if c.get("delta") is True and c.get("level") == 0 and c.get("parent_id") == submission_id:
            body = c.get("body", "")
            if body and body not in ("[deleted]", "[removed]"):
                return c
    return None


def _get_nodelta_comment(comment_entry, submission_id):
    for c in comment_entry.get("comments", []):
        if c.get("delta") is True:
            continue
        if c.get("level") == 0 and c.get("parent_id") == submission_id:
            body = c.get("body", "")
            if body and body not in ("[deleted]", "[removed]"):
                return c
    return None


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
                date = datetime.utcfromtimestamp(int(created_utc)).date() if created_utc else None
            except (ValueError, TypeError):
                date = None

            delta_c = _get_delta_comment(pair["delta_comment"], submission_id)
            nodelta_c = _get_nodelta_comment(pair["nodelta_comment"], submission_id)

            if not delta_c or not nodelta_c:
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
                date = datetime.utcfromtimestamp(min_ts).date()
            except (ValueError, TypeError):
                pass

        direct_replies = [uid for uid in uids if utterances[uid]["reply-to"] == utterances[uid]["root"]]
        delta_uids = [uid for uid in direct_replies if utterances[uid]["meta"].get("success") == 1]
        nodelta_uids = [uid for uid in direct_replies if utterances[uid]["meta"].get("success") == 0]

        for d_uid in delta_uids:
            for nd_uid in nodelta_uids:
                rows.append(ArgumentPair(
                    thread_id=root_id,
                    topic=title,
                    original_post=op_text,
                    delta_argument=utterances[d_uid]["text"],
                    nodelta_argument=utterances[nd_uid]["text"],
                    date=date,
                ))
    return rows


if __name__ == "__main__":
    pairs = build_webis_raw() + build_winning_raw()
    df = pd.DataFrame([p.model_dump() for p in pairs])
    df = df[df["delta_argument"] != df["nodelta_argument"]]
    df.to_csv("argument_quality_dataset.csv", index=False)
    print(f"Dataset saved: {len(df)} rows, {df['thread_id'].nunique()} unique threads")