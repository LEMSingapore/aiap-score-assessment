# AIAP Technical Assessment — score.db

**Name:** Chang Chee Young  
**Email:** cheeyoung.chang@gmail.com

## 1. Project Overview

This repository contains my solution for the AIAP technical assessment based on the `score.db` dataset.

The submission covers the two required tasks:

- An **EDA notebook** (`eda.ipynb`) for exploratory data analysis.
- A modular **machine learning pipeline** under `src/`.
- A single **entrypoint script** (`run.sh`) to reproduce the full workflow.

The objective of the pipeline is to predict students’ mathematics examination scores using the target column `final_test`, based on demographic, behavioural, and school-related features stored in the SQLite database.

### Data and preprocessing summary

The dataset is retrieved from `data/score.db` using a relative path, as required by the assessment brief.

The preprocessing pipeline in `src/preprocessing.py` implements the key EDA decisions:

- Normalise messy categorical labels in `CCA` and `tuition`.
- Drop rows with missing `final_test`, since the target should not be imputed for supervised learning.
- Treat negative ages as data errors, impute missing age values with the median, and correct implausible ages 5 and 6 to 15 and 16.
- Handle missing `attendance_rate` using both median imputation and a missingness flag (`attendance_rate_was_nan`).
- Engineer `classsize = n_male + n_female`.
- Drop identifier and low-signal columns such as `student_id`, `bag_color`, `gender`, `wake_time`, and `mode_of_transport`.

The final modelling feature set contains:

- Numeric features: `number_of_siblings`, `hours_per_week`, `attendance_rate`, `classsize`
- Categorical features: `direct_admission`, `CCA`, `learning_style`, `tuition`, `sleep_time`
- Indicator feature: `attendance_rate_was_nan`

## 2. Repository Structure

- `README.md` — overview, setup, execution, and results summary.
- `eda.ipynb` — exploratory data analysis notebook on `score.db`.
- `requirements.txt` — Python dependencies for reproducibility.
- `run.sh` — shell entrypoint to run the pipeline.
- `.gitignore` — excludes virtual environments, caches, and local-only files.

- `src/`
  - `main.py` — orchestrates the end-to-end pipeline.
  - `settings.py` — central configuration for paths, target, split settings, and feature groups.
  - `preprocessing.py` — data loading, cleaning, and feature engineering.
  - `models.py` — sklearn preprocessing, model training, evaluation, and feature importance extraction.

- `data/`
  - `score.db` — SQLite database used for the assessment; not committed to git as required.

- `artifacts/`
  - optional output directory for saved metrics, models, and plots.

## 3. Setup Instructions

1. Clone the repository and move into it:

```bash
git clone <your-github-url>.git
cd aiap-score-assessment
```

2. Create and activate a virtual environment:

# Requires Python 3.11 or higher (pandas 3.0 dependency)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Place the `score.db` file in the `data/` folder:

```bash
mv /path/to/score.db data/score.db
```

## 4. How to Run

### 4.1 EDA

Launch Jupyter from the repository root:

```bash
jupyter lab
```

Then open `eda.ipynb` and run the notebook from top to bottom. The notebook documents the EDA process, the conclusions from each step, and the decisions later implemented in the Python pipeline.

### 4.2 Pipeline

Run the end-to-end machine learning pipeline from the repository root:

```bash
./run.sh
```

or directly:

```bash
python -m src.main
```

The pipeline performs the following steps:

1. Load the data from `data/score.db`.
2. Apply the preprocessing and feature engineering logic.
3. Split the data into training and test sets.
4. Train and compare three regression models.
5. Evaluate the models using MAE, RMSE, and \(R^2\).
6. Print the best-performing model and the top feature importances when supported by that model.

## 5. Configuration

Key configuration is centralised in `src/settings.py`, including:

- `DB_PATH` — relative path to the SQLite database.
- `TARGET_COL` — target column name (`final_test`).
- `RANDOM_STATE` — random seed for reproducibility.
- `TEST_SIZE` — held-out test split proportion.
- `NUM_FEATURES`, `CAT_FEATURES`, `INDICATOR_FEATURES` — feature group definitions for modelling.
- `ID_COLUMNS`, `LOW_SIGNAL_COLUMNS` — columns excluded during preprocessing.

This design keeps the pipeline configurable and reduces hard-coded values across modules, which supports easier experimentation and maintenance.

## 6. Pipeline Design

The machine learning pipeline is structured in a modular way so that each file has a single responsibility.

### `preprocessing.py`

This module:

- loads the SQLite table into a pandas DataFrame,
- applies the cleaning decisions from EDA,
- engineers `classsize`,
- removes unused or low-signal columns,
- returns the final feature matrix `X` and target vector `y`.

### `models.py`

This module defines:

- a `ColumnTransformer` for numeric, categorical, and indicator features,
- candidate model pipelines for:
  - `LinearRegression`
  - `RandomForestRegressor`
  - `GradientBoostingRegressor`
- regression evaluation metrics:
  - MAE
  - RMSE
  - \(R^2\)
- a helper to extract feature importances from fitted tree-based models.

### `main.py`

This module orchestrates the full flow:

- calls `prepare_dataset()`,
- performs the train/test split,
- trains all candidate models,
- ranks them by RMSE,
- reports the best model,
- prints the top feature importances for the best tree-based model.

## 7. Feature Processing Summary

The table below summarises how every column in the raw `score.db` dataset is handled. Detailed reasoning is in `eda.ipynb`; this is a quick reference.

| Feature | Type | Processing Applied | Rationale |
|---|---|---|---|
| `final_test` | Target | Rows with missing values dropped (~3% of data) | Regression target; imputing it would distort the labels |
| `student_id` | Identifier | Dropped | Pure row identifier, no predictive signal |
| `index` | Identifier | Dropped | Row index artefact from data export |
| `number_of_siblings` | Numeric | Kept; scaled via StandardScaler | Moderate negative correlation with target |
| `n_male` / `n_female` | Numeric | Combined into engineered `class_size`, originals dropped | Class-composition counts; sum is more interpretable and removes redundancy |
| `class_size` | Numeric (engineered) | `n_male + n_female`; scaled | Strongest single predictor in the final model |
| `age` | Numeric | Negative values to NaN; ages 5/6 corrected to 15/16; then dropped from feature set | Near-zero variance after cleaning (all students 15–16); no predictive value |
| `hours_per_week` | Numeric | Kept; scaled | Weak but non-trivial relationship with target |
| `attendance_rate` | Numeric | Missingness indicator created; median imputation deferred to sklearn pipeline (post-split) | ~4.9% missing; missingness itself may carry signal |
| `attendance_rate_was_nan` | Indicator (engineered) | Binary flag, passed through unscaled | Lets the model use the fact a value was originally missing |
| `direct_admission` | Categorical | One-hot encoded | Visible group-mean difference in target |
| `CCA` | Categorical | Labels normalised (capitalisation/plural variants collapsed to 4 categories); one-hot encoded | Inconsistent raw labels; meaningful signal after cleaning |
| `tuition` | Categorical | Labels normalised (Y/N/Yes/No collapsed to Yes/No); one-hot encoded | Inconsistent raw labels |
| `learning_style` | Categorical | One-hot encoded | Visible group-mean difference in target |
| `sleep_time` | Categorical | Bucketed (Early/Normal/Late); one-hot encoded | Reduces cardinality of raw time strings; retains signal |
| `wake_time` | Categorical | Bucketed in EDA, then dropped | Near-identical target distribution across buckets; no signal |
| `gender` | Categorical | Dropped | Group means differ by <1 point; no usable signal |
| `mode_of_transport` | Categorical | Dropped | Overlapping distributions across categories; no signal |
| `bag_color` | Categorical | Dropped | High-cardinality nominal with no plausible link to exam scores |

**Processing applied inside the sklearn pipeline (post train/test split):**
- Numeric features: median imputation (`SimpleImputer`) → `StandardScaler`
- Categorical features: most-frequent imputation → `OneHotEncoder(handle_unknown="ignore")`
- Indicator feature: passed through unchanged

Imputation and scaling are fitted on the training split only, never on the full dataset, to avoid train-test leakage.

## 8. Model Choice and Evaluation

This is a regression task because the target `final_test` is a continuous numeric score.

Four models were evaluated:

- **DummyRegressor (mean strategy)** — trivial baseline that predicts the mean of `y_train` for every row. Establishes the lower bound; any real model should beat this comfortably.
- **Linear Regression** — captures linear signal between features and target. Acts as a simple-but-trained reference point.
- **Random Forest Regressor** — non-linear ensemble that handles mixed feature types and interactions without manual feature engineering.
- **Gradient Boosting Regressor** — boosting-based tree ensemble, typically requires more careful tuning than RF but can match or exceed it.

Model selection used 5-fold cross-validation on the training set, followed by GridSearchCV hyperparameter tuning for RF and GBR. The held-out test set was reserved for final evaluation and was not used during model selection or tuning.

### Evaluation metrics

- **MAE** — mean absolute error in score units. Easy to interpret for the client.
- **RMSE** — penalises large errors more strongly than MAE. Used as the primary ranking metric because for U.A Secondary School the cost of large prediction errors (misidentifying weak students who need intervention) outweighs many small errors.
- **R²** — proportion of target variance explained. Useful for comparing against the baseline (a model that explains 0% of variance is no better than predicting the mean).

## 9. Results Summary

### Cross-validation (5-fold, default hyperparameters)

| Model | Mean RMSE | Std RMSE |
|---|---:|---:|
| Random Forest | 5.47 | 0.14 |
| Gradient Boosting | 6.35 | 0.07 |
| Linear Regression | 9.10 | 0.06 |
| Baseline (mean) | 13.99 | 0.13 |

Small standard deviations (coefficient of variation under 3% for all models) indicate stable performance across folds.

### Hyperparameter tuning (GridSearchCV, 5-fold)

| Model | Best CV RMSE | Best Parameters |
|---|---:|---|
| Random Forest | 5.40 | `max_depth=10, min_samples_leaf=1, n_estimators=400` |
| Gradient Boosting | 5.47 | `learning_rate=0.1, max_depth=5, n_estimators=400` |

GBR improved by 14% in CV RMSE after tuning, RF by only 1%. This reflects their different hyperparameter sensitivities — RF is robust to most settings provided there are enough trees, whereas GBR's performance depends critically on the interaction between learning rate, tree depth, and tree count.

### Held-out test results (tuned models)

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 3.71 | 5.31 | 0.855 |
| Gradient Boosting | 3.81 | 5.36 | 0.852 |

Random Forest is selected as the final model, though after tuning RF and GBR are within 0.06 RMSE of each other — essentially tied. The convergence of two different model families to similar performance suggests the feature set is near its information ceiling; further improvements would likely require new features rather than better models.

### Feature importance (tuned Random Forest)

| Rank | Feature | Importance |
|---|---|---:|
| 1 | `classsize` | 0.386 |
| 2 | `number_of_siblings` | 0.283 |
| 3 | `hours_per_week` | 0.117 |
| 4 | `attendance_rate` | 0.058 |
| 5 | `learning_style_Visual` | 0.037 |
| 6 | `learning_style_Auditory` | 0.037 |
| 7 | `CCA_None` | 0.023 |
| 8 | `direct_admission_Yes` | 0.013 |
| 9 | `direct_admission_No` | 0.013 |
| 10 | `tuition_No` | 0.013 |

The top four features (`classsize`, `number_of_siblings`, `hours_per_week`, `attendance_rate`) together account for ~84% of total importance. `classsize` being the dominant predictor is consistent with educational research linking class size to learning outcomes; the strong negative correlation of `number_of_siblings` with the target may proxy for socio-economic context rather than a direct causal effect.

## 10. Key EDA Findings

Important findings from EDA that informed the pipeline include:

- The target `final_test` had missing values, so rows with missing target were removed rather than imputed.
- `CCA` and `tuition` contained inconsistent categorical labels and required normalisation.
- `attendance_rate` had missing values and benefited from both imputation and a missingness indicator.
- `n_male` and `n_female` were more useful after being combined into `classsize`.
- `age` had very low variance after correction and was dropped from the final feature set.
- Several columns were excluded because they were identifiers, high-noise attributes, or showed weak predictive value during EDA.

## 11. What I Would Do With More Time

The pipeline now includes cross-validation, hyperparameter tuning, and a baseline. Further improvements would explore:

- **More features.** The convergence of RF and GBR to similar RMSE suggests the current feature set is near its information ceiling. Likely candidates: time-derived features from `sleep_time` (sleep duration computed against `wake_time`), interaction terms between `attendance_rate` and `hours_per_week`, and student-cohort aggregates (e.g. peer mean score within class).
- **Broader GBR grid.** The tuning showed GBR is highly hyperparameter-sensitive; a wider grid or RandomizedSearchCV with more iterations might surface a configuration that overtakes RF.
- **Model persistence.** Save the best tuned model and its metadata (CV scores, best params, feature list, sklearn version) to `artifacts/` so the pipeline can be deployed for inference without retraining.
- **Out-of-time validation.** If the dataset spans multiple school years, splitting by time rather than randomly would give a more realistic estimate of forward-looking performance.
- **Calibration check.** Plot predicted vs. actual scores to see where the model systematically over- or under-predicts (e.g. is it accurate in the middle of the distribution but compresses extremes?).

## 12. Notes

- The pipeline uses the relative database path `data/score.db`, following the assessment instructions.
- The SQLite database file itself is not committed to the repository, also following the assessment requirements.