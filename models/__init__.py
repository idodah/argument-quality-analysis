"""Argument-quality models over the pair-wise (delta vs. non-delta) CMV dataset.

Each model decides which of two arguments earned the delta:
  - data           — shared data loading, splitting, prompts, and metrics
  - tfidf_features  — order-invariant TF-IDF featurizer (PairTfidf)
  - tfidf_logreg / tfidf_random_forest / tfidf_xgboost — TF-IDF baselines
  - tfidf_main      — runner for the three TF-IDF baselines
  - gpt_5_4_nano    — zero-shot LLM baseline (no fine-tuning)
  - qwen            — Qwen3-8B QLoRA pair-wise ranker (fine-tuned)
"""
