"""TF-IDF + Random Forest pair-wise baseline.

Uses the shared order-invariant featurizer (`models.tfidf_features.PairTfidf`):
context fields as-is, the two arguments as a signed tfidf difference, so an
A<->B swap negates the argument features and the model can't learn a positional
shortcut.
"""

from sklearn.ensemble import RandomForestClassifier

from models.data import evaluate, RANDOM_SEED
from models.tfidf_features import PairTfidf

_MAX_FEATURES = 10_000


def run(train_fields, train_labels, val_fields, val_labels, test_fields, test_labels, **_) -> list[dict]:
    """Fit the featurizer + Random Forest on train, then return val/test metrics."""
    featurizer = PairTfidf(max_features=_MAX_FEATURES).fit(train_fields)
    X_train = featurizer.transform(train_fields)
    X_val = featurizer.transform(val_fields)
    X_test = featurizer.transform(test_fields)

    clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_train, train_labels)

    results = []
    for split_name, X, y in [("val", X_val, val_labels), ("test", X_test, test_labels)]:
        y_pred = clf.predict(X)
        y_prob = clf.predict_proba(X)[:, 1]
        results.append({**evaluate("tfidf_random_forest", y, y_pred, y_prob), "split": split_name, "n_train": len(train_labels)})
    return results
