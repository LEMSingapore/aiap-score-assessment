"""
Preprocessing utilities for the AIAP score.db technical assessment.

This module implements the data loading and cleaning decisions documented
in eda.ipynb, so the modelling pipeline can rely on a consistent, cleaned
feature set.
"""

from pathlib import Path
import sqlite3
from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Top-level constants (mirroring final feature set in eda.ipynb)
# ---------------------------------------------------------------------

from src.settings import (
    TARGET_COL,
    NUM_FEATURES,
    CAT_FEATURES,
    INDICATOR_FEATURES,
    ID_COLUMNS,
    LOW_SIGNAL_COLUMNS,
)

# ---------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------

def load_data(db_path: Path) -> pd.DataFrame:
    """
    Load the main table from the score.db SQLite database.

    This follows the same logic as the EDA notebook:
    - Connects to the SQLite file.
    - Lists tables and assumes the first table is the student-level data.
    - Reads the full table into a DataFrame.

    Args:
        db_path: Path to the SQLite database file (score.db).

    Returns:
        df: Raw DataFrame as loaded from the database.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    with sqlite3.connect(db_path) as conn:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;",
            conn,
        )["name"].tolist()

        if not tables:
            raise RuntimeError(f"No tables found in database at {db_path}")

        table_name = tables[0]
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    return df

# ---------------------------------------------------------------------
# 2. Categorical normalisation (CCA, tuition)
# ---------------------------------------------------------------------

def normalise_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise label variants for key categorical columns.

    Decisions (from EDA):
    - Map CCA to four canonical categories: Sports, Clubs, Arts, None.
      This collapses inconsistent capitalisation and singular/plural forms.
    - Map tuition from mixed Yes/No/Y/N into clean 'Yes' and 'No' labels.

    Args:
        df: Raw dataframe with original categorical columns.

    Returns:
        df: Dataframe with CCA and tuition normalised.
    """
    df = df.copy()

    # OLD (sprint 6): uppercase keys unreachable after .str.lower()
    # ccamap = {
    #     "sports": "Sports", "sport": "Sports", "SPORTS": "Sports",
    #     "clubs": "Clubs", "club": "Clubs", "CLUBS": "Clubs",
    #     "arts": "Arts", "art": "Arts", "ARTS": "Arts",
    #     "none": "None", "NONE": "None",
    # }

    ccamap = {
        "sports": "Sports",
        "sport": "Sports",
        "clubs": "Clubs",
        "club": "Clubs",
        "arts": "Arts",
        "art": "Arts",
        "none": "None",
    }

    tuitionmap = {
        "yes": "Yes",
        "y": "Yes",
        "no": "No",
        "n": "No",
    }

    # CCA: strip, lowercase, map, keep canonical labels
    df["CCA"] = df["CCA"].str.strip().str.lower().map(ccamap)

    # tuition: strip, lowercase, map to Yes/No
    df["tuition"] = df["tuition"].str.strip().str.lower().map(tuitionmap)
    return df

# ---------------------------------------------------------------------
# 3. Target handling
# ---------------------------------------------------------------------

def drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with missing regression target `final_test`.

    Rationale:
    - `final_test` is the regression target.
    - Imputing the target would leak information into the labels
      and distort evaluation, so rows are dropped instead.

    Args:
        df: Dataframe including `final_test`.

    Returns:
        df: Dataframe with rows where `final_test` is null removed.
    """
    df = df.copy()
    df = df.dropna(subset=[TARGET_COL])
    return df

# ---------------------------------------------------------------------
# 4. Age cleaning
# ---------------------------------------------------------------------

def clean_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix invalid ages and handle implausible values.

    Decisions from EDA:
    - Negative ages are treated as data errors and set to NaN.
    - Ages 5 and 6 are implausible for secondary students; they are
      corrected to 15 and 16 respectively (assumed leading digit dropped).
    - Statistical imputation is deferred to the sklearn pipeline so it
      can be fitted on the training split only.

    Args:
        df: Dataframe including an 'age' column.

    Returns:
        df: Dataframe with cleaned 'age' values (NaNs may remain).
    """
    df = df.copy()

    # Treat negative ages as missing
    df.loc[df["age"] < 0, "age"] = np.nan

    # OLD (sprint 1): leaked full-dataset median into training
    # median_age = df["age"].median()
    # df["age"] = df["age"].fillna(median_age)
    #
    # FIX: deferred to SimpleImputer in models.build_preprocessor().

    # Correct obviously mis-entered ages 5 and 6
    df.loc[df["age"] == 5, "age"] = 15
    df.loc[df["age"] == 6, "age"] = 16

    return df

# ---------------------------------------------------------------------
# 5. Attendance rate missingness
# ---------------------------------------------------------------------

def handle_attendance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Indicator of original missingness (deterministic, safe pre-split)
    df["attendance_rate_was_nan"] = df["attendance_rate"].isna().astype(int)

    # OLD (sprint 1): leaked test-set median into training
    # median_attendance = df["attendance_rate"].median()
    # df["attendance_rate"] = df["attendance_rate"].fillna(median_attendance)
    #
    # FIX: imputation deferred to SimpleImputer in models.build_preprocessor(),
    # which is fitted on the training split only.

    return df

# ---------------------------------------------------------------------
# 6. Identifier / low-signal drops and classsize
# ---------------------------------------------------------------------

def drop_id_and_low_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop identifier-like and low-signal categorical columns.

    Decisions:
    - Drop `index` and `student_id` as pure identifiers.
    - Drop `bag_color` as high-cardinality noise.
    - Drop `gender`, `wake_time`, `mode_of_transport` after weak relationships
      with `final_test` and low signal-to-noise in EDA.

    Args:
        df: Dataframe with original columns.

    Returns:
        df: Dataframe with ID and low-signal columns removed (if present).
    """
    df = df.copy()
    to_drop = ID_COLUMNS + LOW_SIGNAL_COLUMNS
    df = df.drop(columns=to_drop, errors="ignore")
    return df

def add_class_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer `classsize` from `n_male` and `n_female`, then drop the originals.

    Decisions from EDA:
    - `n_male` and `n_female` each have limited value on their own.
    - Their sum is a simple class size signal, which is easier to interpret.

    Args:
        df: Dataframe including `n_male` and `n_female`.

    Returns:
        df: Dataframe with `classsize` added and `n_male`/`n_female` removed.
    """
    df = df.copy()

    df["class_size"] = df["n_male"] + df["n_female"]
    df = df.drop(columns=["n_male", "n_female"], errors="ignore")

    return df

def maybe_drop_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optionally drop `age` from the feature set after cleaning.

    EDA conclusion:
    - After correction, age has very low variance and limited predictive value,
      so it is excluded from the final feature list.

    Args:
        df: Dataframe including 'age'.

    Returns:
        df: Dataframe without 'age' if present.
    """
    df = df.copy()
    df = df.drop(columns=["age"], errors="ignore")
    return df

# ---------------------------------------------------------------------
# 7. Final feature / target split
# ---------------------------------------------------------------------

def make_feature_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select the final feature set and separate X, y.

    Final set (from EDA):
    - Numeric: number_of_siblings, hours_per_week, attendance_rate, classsize
    - Categorical: direct_admission, CCA, learning_style, tuition, sleep_time
    - Indicator: attendance_rate_was_nan
    - Target: final_test

    Args:
        df: Cleaned dataframe.

    Returns:
        X: DataFrame with feature columns only.
        y: Series with the target `final_test`.
    """
    # Ensure we don't accidentally miss columns that should exist
    feature_cols = NUM_FEATURES + CAT_FEATURES + INDICATOR_FEATURES
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    return X, y


# ---------------------------------------------------------------------
# 8. Orchestration helper
# ---------------------------------------------------------------------

def prepare_dataset(db_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Full preprocessing pipeline used by main.py.

    Steps (mirrors the EDA notebook narrative):
    1. Load raw table from score.db.
    2. Normalise CCA and tuition labels.
    3. Drop rows with missing final_test.
    4. Clean age (negative values, 5/6 corrections, impute median).
    5. Handle attendancerate missingness (indicator + median).
    6. Engineer classsize and drop nmale / nfemale.
    7. Drop identifier and low-signal columns.
    8. Optionally drop age from features.
    9. Return X, y using the final feature list.

    Args:
        db_path: Path to the score.db file.

    Returns:
        X, y: Cleaned features and target ready for train/val/test splitting.
    """
    df = load_data(db_path)
    df = normalise_categoricals(df)
    df = drop_missing_target(df)
    df = clean_age(df)
    df = handle_attendance(df)
    df = add_class_size(df)
    df = drop_id_and_low_signal_columns(df)
    df = maybe_drop_age(df)
    X, y = make_feature_target(df)
    return X, y