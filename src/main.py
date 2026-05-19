from pathlib import Path

from sklearn.model_selection import train_test_split

from src.preprocessing import prepare_dataset
from src.models import train_and_evaluate_models


RANDOM_STATE = 42
TEST_SIZE = 0.2
DB_PATH = Path("data/score.db")


def main() -> None:
    """
    Run the end-to-end regression pipeline.

    Steps:
    1. Load and preprocess the dataset from score.db.
    2. Split the data into train and test sets.
    3. Train candidate regression models.
    4. Evaluate models on the test set.
    5. Print a ranked results table.
    """
    X, y = prepare_dataset(DB_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    trained_models, results_df = train_and_evaluate_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    print("\nModel evaluation results (sorted by RMSE):")
    print(results_df)

    best_model_name = results_df.iloc[0]["model"]
    print(f"\nBest model: {best_model_name}")


if __name__ == "__main__":
    main()