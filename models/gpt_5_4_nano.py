"""
Zero-shot evaluation of GPT-5.4-nano on the test set (no fine-tuning).

Pair-wise prompting: both arguments are shown in a single prompt and the model
chooses "A" or "B". Probability of A is calibrated from the top-logprob
distribution at the answer token.

Usage:
    python -m models.gpt_5_4_nano
"""

import math
import os
import time

import numpy as np
from openai import OpenAI

from models.data import RANDOM_SEED, evaluate, load_data, make_prompt, save_results, shuffle_pairs, split_by_date

MODEL_ID = "gpt-5.4-nano"
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
# Confidence nudge for the no-logprobs fallback: enough to break the A/B tie in
# the right direction, small enough to stay near-neutral in roc_auc_score.
_FALLBACK_MARGIN = 0.01


def _predict_one(client: OpenAI, prompt: dict) -> tuple[int, float]:
    """Return (pred, P(A)) for a single pairwise prompt."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                ],
                max_completion_tokens=2,
                temperature=0.0,
                logprobs=True,
                top_logprobs=5,
                seed=RANDOM_SEED,
            )
            return _expected_choice(resp)
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_BACKOFF * (2 ** attempt))
    raise RuntimeError("unreachable: retry loop exited without returning or raising")


def _expected_choice(resp) -> tuple[int, float]:
    """Probability of 'A' from the top-logprob mass on A vs B at the first content token."""
    choice = resp.choices[0]
    text = (choice.message.content or "").strip().upper()

    logprobs = getattr(choice, "logprobs", None)
    if logprobs is None or not getattr(logprobs, "content", None):
        return _parse_choice_text(text)

    first = logprobs.content[0]
    p_a, p_b = 0.0, 0.0
    for cand in first.top_logprobs:
        tok = cand.token.strip().upper()
        if tok == "A":
            p_a += math.exp(cand.logprob)
        elif tok == "B":
            p_b += math.exp(cand.logprob)
    total = p_a + p_b
    if total == 0.0:
        return _parse_choice_text(text)
    p_a_norm = p_a / total
    return (1 if p_a_norm >= 0.5 else 0, p_a_norm)


def _parse_choice_text(text: str) -> tuple[int, float]:
    """Fallback when no logprobs are available: we have a hard label but no
    calibrated confidence. Return a probability only marginally off 0.5 in the
    decided direction so the pred is tie-broken correctly while the value stays
    near-neutral in roc_auc_score (a fabricated 0.0/1.0 would let a confident-
    but-wrong fallback distort the AUC ranking as if the model were certain)."""
    first = next((c for c in text if c in ("A", "B")), None)
    if first == "A":
        return 1, 0.5 + _FALLBACK_MARGIN
    if first == "B":
        return 0, 0.5 - _FALLBACK_MARGIN
    return 0, 0.5


def _predict(client: OpenAI, fields: dict, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    preds, probs = [], []
    n = len(labels)
    for i in range(n):
        prompt = make_prompt(
            fields["topic"][i],
            fields["original_post"][i],
            fields["arg_a"][i],
            fields["arg_b"][i],
        )
        pred, p_a = _predict_one(client, prompt)
        preds.append(pred)
        probs.append(p_a)
        if (i + 1) % 50 == 0:
            print(f"  predicted {i + 1}/{n}")
    return np.array(preds), np.array(probs)


def run(test_fields, test_labels, **_) -> list[dict]:
    model_name = "gpt_5_4_nano"
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    y_pred, y_prob = _predict(client, test_fields, test_labels)
    return [{**evaluate(model_name, test_labels, y_pred, y_prob), "split": "test"}]


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    df = load_data()
    _, _, test_df = split_by_date(df)
    test_fields, test_labels = shuffle_pairs(test_df, seed=RANDOM_SEED + 2)

    results = run(test_fields, test_labels)
    save_results(results)
    for r in results:
        print(r)
