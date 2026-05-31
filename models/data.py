"""
Shared data loading, splitting, and pair-shuffling utilities.
"""

import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results.csv"
RESULTS_COLUMNS = ["model", "split", "n_train", "accuracy", "precision", "recall", "f1", "roc_auc", "timestamp"]

RANDOM_SEED = 42
HF_REPO_ID = "idodah/argument-quality-cmv"


def load_data() -> pd.DataFrame:
    ds = load_dataset(HF_REPO_ID, split="filtered_v2")
    df = ds.to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date", "topic", "original_post", "delta_argument", "nodelta_argument"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def split_by_date(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Rank threads by their earliest date, then cut at 80/90% of unique threads.
    thread_min_date = df.groupby("thread_id")["date"].min().sort_values()
    n = len(thread_min_date)
    train_threads = set(thread_min_date.iloc[: int(n * 0.80)].index)
    val_threads = set(thread_min_date.iloc[int(n * 0.80) : int(n * 0.90)].index)

    split = df["thread_id"].map(
        lambda tid: "train" if tid in train_threads else ("val" if tid in val_threads else "test")
    )
    return df[split == "train"], df[split == "val"], df[split == "test"]


def shuffle_pairs(df: pd.DataFrame, seed: int = RANDOM_SEED) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    For each row, randomly assign delta/nodelta to position 0 or 1.
    Returns:
      fields — dict with keys "topic", "original_post", "arg_a", "arg_b",
               each an (N,) array of strings
      labels — (N,) array where 1 means arg_a is the delta argument
    """
    rng = random.Random(seed)
    topics, posts, arg_as, arg_bs, labels = [], [], [], [], []
    for _, row in df.iterrows():
        if rng.random() < 0.5:
            arg_a, arg_b, label = row["delta_argument"], row["nodelta_argument"], 1
        else:
            arg_a, arg_b, label = row["nodelta_argument"], row["delta_argument"], 0
        topics.append(row["topic"])
        posts.append(row["original_post"])
        arg_as.append(arg_a)
        arg_bs.append(arg_b)
        labels.append(label)
    fields = {
        "topic": np.array(topics),
        "original_post": np.array(posts),
        "arg_a": np.array(arg_as),
        "arg_b": np.array(arg_bs),
    }
    labels_arr = np.array(labels)
    pos_rate = labels_arr.mean() if len(labels_arr) else 0.0
    print(f"shuffle_pairs: n={len(labels_arr)}, P(label=1)={pos_rate:.3f}")
    return fields, labels_arr


SYSTEM_PROMPT = (
    "You are an expert at evaluating argument quality. "
    "Given a topic, a post expressing a position, and two responses, "
    "decide which response is more persuasive and successfully changed the author's mind."
)

RANK_SYSTEM_PROMPT = (
    "You are an expert at evaluating argument quality. "
    "Given a topic, a post expressing a position, and a response, "
    "judge how persuasive the response is at changing the author's mind."
)


def make_rank_prompt(topic: str, post: str, argument: str) -> dict:
    """Single-argument prompt used by the pair-wise ranking head.

    The '### Post\\n' and '\\n\\n### Response' markers below are parsed by
    qwen._trim_prompt_to_length to locate the post for length-trimming — keep
    them in sync if you change this format, or trimming raises ValueError.
    """
    user = (
        f"### Topic\n{topic}\n\n"
        f"### Post\n{post}\n\n"
        f"### Response\n{argument}\n\n"
        "### Score\n"
    )
    return {"system": RANK_SYSTEM_PROMPT, "user": user}


def make_prompt(topic: str, post: str, arg_a: str, arg_b: str, label: int | None = None) -> dict:
    """
    Build system/user prompt dict for the LLM.
    If label is provided, appends the answer to the user message (for training).
    Label 1 = Argument A is the persuasive one, 0 = Argument B is the persuasive one.
    Returns {"system": str, "user": str} — callers format into chat template as needed.
    """
    user = (
        f"### Topic\n{topic}\n\n"
        f"### Post\n{post}\n\n"
        f"### Response A\n{arg_a}\n\n"
        f"### Response B\n{arg_b}\n\n"
        "Which response is more persuasive? Answer with only 'A' or 'B'.\n\n"
        "### Answer\n"
    )
    if label is not None:
        user += "A" if label == 1 else "B"
    return {"system": SYSTEM_PROMPT, "user": user}


def save_results(results: list[dict], path: Path = RESULTS_PATH) -> pd.DataFrame:
    """
    Persist each model's latest run to a shared CSV. Rows for any model present
    in `results` are replaced; rows for other models are preserved.
    """
    timestamp = datetime.now().isoformat(timespec="seconds")
    new_df = pd.DataFrame([{**r, "timestamp": timestamp} for r in results])

    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[~existing["model"].isin(new_df["model"].unique())]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.reindex(columns=RESULTS_COLUMNS)
    combined = combined.sort_values(["model", "split"]).reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    result = {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
    }
    if y_prob is not None:
        result["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
    return result
