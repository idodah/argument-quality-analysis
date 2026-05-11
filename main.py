import pandas as pd

from models import majority_baseline, tfidf_logreg, tfidf_random_forest, tfidf_xgboost
from models.data import load_data, split_by_date, shuffle_pairs, RANDOM_SEED

if __name__ == "__main__":
    print(f"Loading data...")
    df = load_data()
    print(f"  {len(df)} rows, date range: {df['date'].min().date()} → {df['date'].max().date()}")

    train_df, val_df, test_df = split_by_date(df)
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_fields, train_labels = shuffle_pairs(train_df, seed=RANDOM_SEED)
    val_fields, val_labels = shuffle_pairs(val_df, seed=RANDOM_SEED + 1)
    test_fields, test_labels = shuffle_pairs(test_df, seed=RANDOM_SEED + 2)

    kwargs = dict(
        train_fields=train_fields, train_labels=train_labels,
        val_fields=val_fields, val_labels=val_labels,
        test_fields=test_fields, test_labels=test_labels,
    )

    all_results = []
    for module in [majority_baseline, tfidf_logreg, tfidf_random_forest, tfidf_xgboost]:
        print(f"\nRunning {module.__name__.split('.')[-1]}...")
        all_results += module.run(**kwargs)

    results_df = pd.DataFrame(all_results)[["model", "split", "accuracy", "precision", "recall", "f1", "roc_auc"]]
    print(f"\n{results_df.to_string(index=False)}")
