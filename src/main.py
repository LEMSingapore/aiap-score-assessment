from pathlib import Path

import pandas as pd       # NEW (sprint 5): needed for DataFrame in main

from sklearn.model_selection import train_test_split

from src.preprocessing import prepare_dataset
from src.models import (
    train_and_evaluate_models,
    get_feature_importance,
    cross_validate_models,
    tune_models,                  # NEW (sprint 5)
    evaluate_regression,          # NEW (sprint 5) — used for tuned models
)
from src.settings import DB_PATH, RANDOM_STATE, TEST_SIZE

def main() -> None:
    """
    Run the end-to-end regression pipeline.

    Steps:
    1. Load and preprocess data from score.db.
    2. Train/test split.
    3. 5-fold CV on default-param models (broad comparison).
    4. GridSearchCV on tunable models (RF, GBR) to find best hyperparameters.
    5. Evaluate tuned models on held-out test set.
    6. Print results and feature importances.
    """
    X, y = prepare_dataset(DB_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Step 3: default-param CV — quick model comparison
    print("\n5-fold cross-validation results (default hyperparameters):")
    cv_df = cross_validate_models(X_train, y_train, cv=5)
    print(cv_df.to_string(index=False))

    # Step 4: tune RF and GBR via GridSearchCV
    print("\nTuning hyperparameters via GridSearchCV...")
    tuned_pipelines, tuning_df = tune_models(X_train, y_train, cv=5)
    print("\nTuning results (sorted by CV RMSE):")
    print(tuning_df.to_string(index=False))

    # Step 5: evaluate tuned models on held-out test set
    print("\nHeld-out test results (tuned models):")
    test_results = []
    for model_name, pipeline in tuned_pipelines.items():
        y_pred = pipeline.predict(X_test)
        metrics = evaluate_regression(y_test, y_pred)
        metrics["model"] = model_name
        test_results.append(metrics)

    test_df = (
        pd.DataFrame(test_results)
        .sort_values(by="rmse")
        .reset_index(drop=True)
    )
    print(test_df.to_string(index=False))

    best_model_name = test_df.iloc[0]["model"]
    print(f"\nBest tuned model: {best_model_name}")

    best_pipeline = tuned_pipelines[best_model_name]
    model = best_pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importance_df = get_feature_importance(best_pipeline)
        print(f"\nTop 10 feature importances ({best_model_name}):")
        print(importance_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()