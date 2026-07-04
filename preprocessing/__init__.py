"""Pair-wise argument dataset pipeline (delta vs. non-delta CMV replies).

Four stages run in order; each reads the previous stage's output and writes the
next Hugging Face Hub split, so they must run sequentially:

  1. data_creation.py       raw ArgumentPair rows from Webis-CMV-20 +
                            winning-args-corpus  ->  data/*.csv
  2. preprocess.py          clean text, dedup, one post per thread
                            ->  HF split "full"
  3. filter_by_similarity.py  drop off-topic / near-duplicate pairs by
                            embedding cosine similarity
                            "full"  ->  "filtered"
  4. filter_by_tokens.py    drop pairs outside the token-length / ratio bounds
                            "filtered"  ->  "filtered_v2"  (the split models train on)

text_utils.py holds the text-cleaning helpers used by stage 2.
"""
