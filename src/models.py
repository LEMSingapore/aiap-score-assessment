from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUM_FEATURES = [
    "number_of_siblings",
    "hours_per_week",
    "attendance_rate",
    "classsize",
]

CAT_FEATURES = [
    "direct_admission",
    "CCA",
    "learning_style",
    "tuition",
    "sleep_time",
]

INDICATOR_FEATURES = [
    "attendance_rate_was_nan",
]

def build_preprocessor() -> ColumnTransformer:
    """
    Build the sklearn preprocessing transformer.

    Returns:
        ColumnTransformer for numeric, categorical, and indicator features.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUM_FEATURES),
            ("cat", categorical_pipeline, CAT_FEATURES),
            ("ind", "passthrough", INDICATOR_FEATURES),
        ]
    )

    return preprocessor

def build_model_pipelines() -> Dict[str, Pipeline]:
    """
    Build candidate model pipelines.

    Returns:
        Dictionary mapping model name to sklearn Pipeline.
    """
    preprocessor = build_preprocessor()

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    pipelines = {
        name: Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        for name, model in models.items()
    }

    return pipelines

def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        Dictionary of MAE, RMSE, and R2.
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }

def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[Dict[str, Pipeline], pd.DataFrame]:
    """
    Train all candidate models and evaluate on the test set.

    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train: Training target.
        y_test: Test target.

    Returns:
        trained_models: Dict of fitted model pipelines.
        results_df: DataFrame with one row per model and regression metrics.
    """
    pipelines = build_model_pipelines()

    trained_models = {}
    results = []

    for model_name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = evaluate_regression(y_test, y_pred)
        metrics["model"] = model_name

        trained_models[model_name] = pipeline
        results.append(metrics)

    results_df = (
        pd.DataFrame(results)
        .sort_values(by="rmse", ascending=True)
        .reset_index(drop=True)
    )

    return trained_models, results_df