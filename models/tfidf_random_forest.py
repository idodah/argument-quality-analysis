"""TF-IDF + Random Forest pair-wise baseline."""

from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from models.data import evaluate, RANDOM_SEED

_FIELDS = ["topic", "original_post", "arg_a", "arg_b"]
_MAX_FEATURES = 10_000


def _fit_vectorizers(train_fields: dict) -> list[TfidfVectorizer]:
    vectorizers = []
    for field in _FIELDS:
        vec = TfidfVectorizer(max_features=_MAX_FEATURES, sublinear_tf=True, ngram_range=(1, 2))
        vec.fit(train_fields[field])
        vectorizers.append(vec)
    return vectorizers


def _transform(fields: dict, vectorizers: list[TfidfVectorizer]):
    return hstack([vec.transform(fields[f]) for vec, f in zip(vectorizers, _FIELDS)])


def run(train_fields, train_labels, val_fields, val_labels, test_fields, test_labels, **_) -> list[dict]:
    vectorizers = _fit_vectorizers(train_fields)
    X_train = _transform(train_fields, vectorizers)
    X_val = _transform(val_fields, vectorizers)
    X_test = _transform(test_fields, vectorizers)

    clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_train, train_labels)

    results = []
    for split_name, X, y in [("val", X_val, val_labels), ("test", X_test, test_labels)]:
        y_pred = clf.predict(X)
        y_prob = clf.predict_proba(X)[:, 1]
        results.append({**evaluate("tfidf_random_forest", y, y_pred, y_prob), "split": split_name, "n_train": len(train_labels)})
    return results
