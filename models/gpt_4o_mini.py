"""
Zero-shot evaluation of GPT-4o-mini on the test set (no fine-tuning).

Usage:
    python -m models.gpt_4o_mini
"""

import os
import time

import numpy as np
from openai import OpenAI

from models.data import RANDOM_SEED, evaluate, load_data, make_prompt, save_results, shuffle_pairs, split_by_date

MODEL_ID = "gpt-4o-mini"
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0


def _predict_one(client: OpenAI, prompt: dict) -> int:
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                ],
                max_tokens=1,
                temperature=0.0,
                logit_bias=None,
                seed=RANDOM_SEED,
            )
            text = (resp.choices[0].message.content or "").strip().upper()
            print(text)
            if text.startswith("A"):
                return 1
            if text.startswith("B"):
                return 0
            return 0
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_BACKOFF * (2 ** attempt))


def _predict(client: OpenAI, fields: dict, labels: np.ndarray, use_summary: bool) -> np.ndarray:
    preds = []
    n = len(labels)
    for i in range(n):
        post = fields["summary"][i] if use_summary else fields["original_post"][i]
        prompt = make_prompt(
            fields["topic"][i],
            post,
            fields["arg_a"][i],
            fields["arg_b"][i],
        )
        preds.append(_predict_one(client, prompt))
        if (i + 1) % 50 == 0:
            print(f"  predicted {i + 1}/{n}")
    return np.array(preds)


def run(test_fields, test_labels, use_summary: bool = False, **_) -> list[dict]:
    model_name = "gpt_4o_mini_summary" if use_summary else "gpt_4o_mini"
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    y_pred = _predict(client, test_fields, test_labels, use_summary)
    return [{**evaluate(model_name, test_labels, y_pred), "split": "test"}]


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    df = load_data()
    _, _, test_df = split_by_date(df)
    test_fields, test_labels = shuffle_pairs(test_df, seed=RANDOM_SEED + 2)

    results = run(test_fields, test_labels, use_summary=False)
    save_results(results)
    for r in results:
        print(r)