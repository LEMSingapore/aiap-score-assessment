import pandas as pd  # NEW (sprint 5): needed for DataFrame in main
from sklearn.model_selection import train_test_split

from src.models import (
    cross_validate_models,
    evaluate_regression,  # NEW (sprint 5) — used for tuned models
    get_feature_importance,
    save_best_model,  # NEW (sprint 7)
    tune_models,  # NEW (sprint 5)
)
from src.preprocessing import prepare_dataset
from src.settings import (  # NEW: ARTIFACTS_DIR
    ARTIFACTS_DIR,
    DB_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)


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

    test_df = pd.DataFrame(test_results).sort_values(by="rmse").reset_index(drop=True)
    print(test_df.to_string(index=False))

    best_model_name = test_df.iloc[0]["model"]
    print(f"\nBest tuned model: {best_model_name}")

    best_pipeline = tuned_pipelines[best_model_name]

    # NEW (sprint 7): persist best model + metadata
    best_test_metrics = test_df.iloc[0].to_dict()
    best_cv_rmse = tuning_df[tuning_df["model"] == best_model_name][
        "best_cv_rmse"
    ].iloc[0]
    best_params = tuning_df[tuning_df["model"] == best_model_name]["best_params"].iloc[
        0
    ]

    model_path, metadata_path = save_best_model(
        pipeline=best_pipeline,
        model_name=best_model_name,
        cv_rmse=best_cv_rmse,
        test_metrics=best_test_metrics,
        best_params=best_params,
        feature_list=list(X.columns),
        artifacts_dir=ARTIFACTS_DIR,
    )
    print(f"\nSaved best model to: {model_path}")
    print(f"Saved metadata to:   {metadata_path}")

    # also save the comparison table for the README
    comparison_path = ARTIFACTS_DIR / "model_comparison.csv"
    test_df.to_csv(comparison_path, index=False)
    print(f"Saved test comparison to: {comparison_path}")

    model = best_pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importance_df = get_feature_importance(best_pipeline)
        print(f"\nTop 10 feature importances ({best_model_name}):")
        print(importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
