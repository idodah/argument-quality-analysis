"""Shared, order-invariant TF-IDF features for the pair-wise baselines.

The dataset pairs a delta and a non-delta argument; `shuffle_pairs` assigns them
to slots A/B with a 50/50 coin flip and sets label=1 when A is the delta. Naively
concatenating tfidf(arg_a) and tfidf(arg_b) as separate feature blocks lets a
model key on *which slot* a token lands in, learning a positional shortcut that
has nothing to do with persuasiveness. The fine-tuned Qwen ranker avoids this by
construction (it scores each argument independently — see `models.qwen`); these
baselines should be held to the same bar so the comparison is apples-to-apples.

Design:
  - Context (topic, original_post) is symmetric under an A<->B swap, so it is
    vectorized as-is and gives the model the question context.
  - The two arguments share ONE vectorizer (fit on arg_a and arg_b together), and
    we feed the DIFFERENCE tfidf(arg_a) - tfidf(arg_b). Swapping A<->B negates
    this block and leaves context untouched, so the representation carries no
    "slot A is usually the delta" signal — only the contrast between the two
    arguments, which is what the label actually depends on.
"""

from __future__ import annotations

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

_CONTEXT_FIELDS = ["topic", "original_post"]
_ARG_FIELDS = ["arg_a", "arg_b"]


class PairTfidf:
    """Order-invariant TF-IDF featurizer for (context, arg_a, arg_b) rows.

    Fit on the training split, then `transform` any split. Context fields get one
    vectorizer each; both arguments share a single vectorizer and enter as a
    signed difference so an A<->B swap negates them (antisymmetric features).
    """

    def __init__(self, max_features: int):
        self._context_vecs: dict[str, TfidfVectorizer] = {}
        self._arg_vec = TfidfVectorizer(
            max_features=max_features, sublinear_tf=True, ngram_range=(1, 2)
        )
        self._max_features = max_features

    def fit(self, train_fields: dict) -> "PairTfidf":
        for field in _CONTEXT_FIELDS:
            vec = TfidfVectorizer(
                max_features=self._max_features, sublinear_tf=True, ngram_range=(1, 2)
            )
            vec.fit(train_fields[field])
            self._context_vecs[field] = vec
        # Shared argument vocabulary: fit on both slots so a token has the SAME
        # column whether it appears in arg_a or arg_b — a prerequisite for the
        # difference below to be meaningful.
        both_args = list(train_fields["arg_a"]) + list(train_fields["arg_b"])
        self._arg_vec.fit(both_args)
        return self

    def transform(self, fields: dict) -> csr_matrix:
        context = [self._context_vecs[f].transform(fields[f]) for f in _CONTEXT_FIELDS]
        arg_a = self._arg_vec.transform(fields["arg_a"])
        arg_b = self._arg_vec.transform(fields["arg_b"])
        arg_diff = arg_a - arg_b  # antisymmetric: swap A<->B -> negation
        return hstack(context + [arg_diff]).tocsr()
