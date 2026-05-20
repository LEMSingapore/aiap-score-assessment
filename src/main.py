from pathlib import Path

from sklearn.model_selection import train_test_split

from src.preprocessing import prepare_dataset
from src.models import (
    train_and_evaluate_models,
    get_feature_importance,
    cross_validate_models,   # NEW (sprint 4)
)
from src.settings import DB_PATH, RANDOM_STATE, TEST_SIZE

def main() -> None:
    """
    Run the end-to-end regression pipeline.

    Steps:
    1. Load and preprocess the dataset from score.db.
    2. Split the data into train and test sets.
    3. Cross-validate candidate models on training set (model selection).
    4. Train final models on full training set and evaluate on held-out test.
    5. Print results and feature importances.
    """
    X, y = prepare_dataset(DB_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # NEW (sprint 4): 5-fold CV on training set for robust model comparison
    print("\n5-fold cross-validation results (sorted by mean RMSE):")
    cv_df = cross_validate_models(X_train, y_train, cv=5)
    print(cv_df.to_string(index=False))

    # Final training + held-out test evaluation (unchanged)
    trained_models, results_df = train_and_evaluate_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    print("\nHeld-out test results (sorted by RMSE):")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["model"]
    print(f"\nBest model: {best_model_name}")

    best_pipeline = trained_models[best_model_name]

    # FIX (sprint 4): use hasattr instead of name check — gradient_boosting
    # also exposes feature_importances_, the old gate missed it.
    model = best_pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importance_df = get_feature_importance(best_pipeline)
        print(f"\nTop 10 feature importances ({best_model_name}):")
        print(importance_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()