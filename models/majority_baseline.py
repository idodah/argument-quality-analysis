import numpy as np
from sklearn.dummy import DummyClassifier

from models.data import evaluate, RANDOM_SEED


def run(train_labels, val_labels, test_labels, **_) -> list[dict]:
    clf = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
    clf.fit(np.zeros((len(train_labels), 1)), train_labels)
    results = []
    for split_name, labels in [("val", val_labels), ("test", test_labels)]:
        y_pred = clf.predict(np.zeros((len(labels), 1)))
        results.append({**evaluate("majority_baseline", labels, y_pred), "split": split_name})
    return results
