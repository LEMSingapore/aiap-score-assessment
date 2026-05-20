from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.settings import (
    NUM_FEATURES,
    CAT_FEATURES,
    INDICATOR_FEATURES,
    RANDOM_STATE,
)

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
        # OLD (sprint 3): no baseline
        # "linear_regression": LinearRegression(),
        #
        # NEW: DummyRegressor as floor — predicts mean of y_train for every row.
        # Establishes the trivial lower bound; any real model should beat this.
        "baseline_mean": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,   # bonus: also fixing the hardcoded 42 here
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,   # and here
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

def cross_validate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
) -> pd.DataFrame:
    """
    Run k-fold cross-validation on the training set for all candidate models.

    Reports mean and std of RMSE across folds. This gives a more robust
    estimate of model performance than a single train/test split, and the
    std column flags fragile models whose performance depends heavily on
    which rows happened to land in the test fold.

    Note: CV is run on X_train / y_train only — the held-out test set
    remains untouched for final evaluation in train_and_evaluate_models.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        cv: Number of CV folds (default 5).

    Returns:
        DataFrame with columns: model, rmse_mean, rmse_std, sorted by
        rmse_mean ascending.
    """
    from sklearn.model_selection import cross_val_score

    pipelines = build_model_pipelines()
    rows = []

    for model_name, pipeline in pipelines.items():
        # neg_root_mean_squared_error returns negative RMSE so that "higher is better"
        # convention holds for sklearn; we flip the sign for reporting.
        neg_rmse_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        rmse_scores = -neg_rmse_scores

        rows.append({
            "model": model_name,
            "rmse_mean": rmse_scores.mean(),
            "rmse_std": rmse_scores.std(),
        })

    cv_df = (
        pd.DataFrame(rows)
        .sort_values(by="rmse_mean", ascending=True)
        .reset_index(drop=True)
    )

    return cv_df

def tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
) -> Tuple[Dict[str, Pipeline], pd.DataFrame]:
    """
    Run GridSearchCV on tunable models (random_forest, gradient_boosting).

    Search grids are intentionally small so the assessment runs in a few
    minutes and the chosen values are easy to defend. The grids cover the
    most impactful hyperparameters for each model:

    - Random Forest: n_estimators (capacity), max_depth (overfit control),
      min_samples_leaf (smoothing).
    - Gradient Boosting: n_estimators, learning_rate, max_depth — the
      classic boosting tradeoff between learning rate and tree count.

    Linear Regression and DummyRegressor are not tuned (no meaningful
    hyperparameters for this task).

    Args:
        X_train: Training feature matrix.
        y_train: Training target.
        cv: Number of CV folds for inner grid search (default 5).

    Returns:
        tuned_pipelines: Dict mapping model name to refitted best pipeline.
        tuning_df: DataFrame with model, best_params, best_cv_rmse.
    """
    from sklearn.model_selection import GridSearchCV

    preprocessor = build_preprocessor()

    # ---- Random Forest grid ----
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])
    rf_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 2, 4],
    }

    # ---- Gradient Boosting grid ----
    gbr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(
            random_state=RANDOM_STATE,
        )),
    ])
    gbr_grid = {
        "model__n_estimators": [200, 400],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5],
    }

    search_configs = {
        "random_forest": (rf_pipeline, rf_grid),
        "gradient_boosting": (gbr_pipeline, gbr_grid),
    }

    tuned_pipelines = {}
    rows = []

    for model_name, (pipeline, grid) in search_configs.items():
        print(f"  Tuning {model_name}...")
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)

        tuned_pipelines[model_name] = search.best_estimator_
        rows.append({
            "model": model_name,
            "best_cv_rmse": -search.best_score_,
            "best_params": search.best_params_,
        })

    tuning_df = pd.DataFrame(rows).sort_values(by="best_cv_rmse").reset_index(drop=True)

    return tuned_pipelines, tuning_df

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

def get_feature_importance(best_pipeline: Pipeline) -> pd.DataFrame:
    """
    Extract feature importances from a fitted tree-based model pipeline.

    This function assumes:
    - the pipeline contains a fitted `preprocessor`
    - the categorical transformer uses OneHotEncoder
    - the model exposes `feature_importances_` (e.g. RandomForestRegressor)

    Args:
        best_pipeline: A fitted sklearn Pipeline containing:
            - "preprocessor"
            - "model"

    Returns:
        DataFrame with feature names and importance scores, sorted descending.
    """
    preprocessor = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not expose feature_importances_.")

    onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]

    cat_feature_names = list(onehot.get_feature_names_out(CAT_FEATURES))

    feature_names = NUM_FEATURES + cat_feature_names + INDICATOR_FEATURES
    importances = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return importance_df