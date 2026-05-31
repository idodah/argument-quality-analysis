"""Lazy singleton wrapper around the fine-tuned Qwen pairwise ranker."""

from __future__ import annotations

import os

_RANKER: "RankerWrapper | None" = None


class RankerWrapper:
    """Loads the QLoRA-fine-tuned Qwen ranker once and exposes score_pair."""

    def __init__(self, model_path: str | None = None):
        from models import qwen

        path = model_path or os.environ.get("RANKER_PATH")
        if not path:
            raise EnvironmentError(
                "Set RANKER_PATH to a local checkpoint dir or HF repo id "
                "for the fine-tuned ranker (e.g. ./checkpoints/qwen_qlora_rank/final)."
            )
        print(f"[ranker] loading from {path} ...")
        self.model, self.tokenizer = qwen.load_model(path)
        self._score_pair = qwen.score_pair

    def score_pair(self, topic: str, post: str, arg_a: str, arg_b: str) -> dict:
        return self._score_pair(self.model, self.tokenizer, topic, post, arg_a, arg_b)


def get_ranker() -> RankerWrapper:
    global _RANKER
    if _RANKER is None:
        _RANKER = RankerWrapper()
    return _RANKER