# AIAP Technical Assessment — score.db

**Name:** Chang Chee Young  
**Email:** cheeyoung.chang@gmail.com

## 1. Project Overview

This repository contains my solution for the AIAP technical assessment based on the `score.db` dataset.[file:2]

The submission covers the two required tasks:[file:2]

- An **EDA notebook** (`eda.ipynb`) for exploratory data analysis.
- A modular **machine learning pipeline** under `src/`.
- A single **entrypoint script** (`run.sh`) to reproduce the full workflow.

The objective of the pipeline is to predict students’ mathematics examination scores using the target column `final_test`, based on demographic, behavioural, and school-related features stored in the SQLite database.[file:2]

### Data and preprocessing summary

The dataset is retrieved from `data/score.db` using a relative path, as required by the assessment brief.[file:2]

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

- `README.md` — overview, setup, execution, and results summary.[file:10]
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
  - `score.db` — SQLite database used for the assessment; not committed to git as required.[file:2]

- `artifacts/`
  - optional output directory for saved metrics, models, and plots.

## 3. Setup Instructions

1. Clone the repository and move into it:

```bash
git clone <your-github-url>.git
cd aiap-score-assessment
```

2. Create and activate a virtual environment:

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

Then open `eda.ipynb` and run the notebook from top to bottom. The notebook documents the EDA process, the conclusions from each step, and the decisions later implemented in the Python pipeline.[file:2]

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

1. Load the data from `data/score.db`.[file:2]
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

This design keeps the pipeline configurable and reduces hard-coded values across modules, which supports easier experimentation and maintenance.[file:2]

## 6. Pipeline Design

The machine learning pipeline is structured in a modular way so that each file has a single responsibility.[file:2][file:1]

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

## 7. Model Choice and Evaluation

This is a regression task because the target `final_test` is a continuous numeric score.[file:2]

Three models were selected to provide a simple but meaningful comparison:[file:1]

- **Linear Regression** — baseline linear model for interpretability and benchmarking.
- **Random Forest Regressor** — non-linear ensemble model that can capture interactions and mixed feature types well.
- **Gradient Boosting Regressor** — boosting-based tree model for a second non-linear benchmark.

The chosen evaluation metrics are:

- **MAE** — average absolute prediction error in score units.
- **RMSE** — penalises larger errors more strongly and is the primary ranking metric.
- **R2** — proportion of target variance explained by the model.[file:1]

These metrics are appropriate for regression and were computed on a held-out test set rather than on training data.[file:1]

## 8. Results Summary

- **Target**
  - Regression: `final_test`
  - Classification: not applicable for this assessment

- **Best model**
  - `RandomForestRegressor`
  - Hyperparameters:
    - `n_estimators=300`
    - `min_samples_split=5`
    - `min_samples_leaf=2`
    - `random_state=42`

### Held-out test results

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Random Forest | 3.666 | 5.362 | 0.852 |
| Gradient Boosting | 4.829 | 6.469 | 0.784 |
| Linear Regression | 7.237 | 9.065 | 0.577 |

### Feature importance from best model

The top feature importances from the fitted random forest were:

| Rank | Feature | Importance |
|---|---|---:|
| 1 | `classsize` | 0.3645 |
| 2 | `number_of_siblings` | 0.2624 |
| 3 | `hours_per_week` | 0.1259 |
| 4 | `attendance_rate` | 0.0730 |
| 5 | `learning_style_Auditory` | 0.0346 |
| 6 | `learning_style_Visual` | 0.0337 |
| 7 | `CCA_None` | 0.0213 |
| 8 | `direct_admission_Yes` | 0.0139 |
| 9 | `direct_admission_No` | 0.0136 |
| 10 | `tuition_No` | 0.0133 |

These results suggest that class context and study-related variables, especially `classsize`, `number_of_siblings`, `hours_per_week`, and `attendance_rate`, contribute strongly to prediction performance in the current feature set.

## 9. Key EDA Findings

Important findings from EDA that informed the pipeline include:

- The target `final_test` had missing values, so rows with missing target were removed rather than imputed.
- `CCA` and `tuition` contained inconsistent categorical labels and required normalisation.
- `attendance_rate` had missing values and benefited from both imputation and a missingness indicator.
- `n_male` and `n_female` were more useful after being combined into `classsize`.
- `age` had very low variance after correction and was dropped from the final feature set.
- Several columns were excluded because they were identifiers, high-noise attributes, or showed weak predictive value during EDA.

## 10. What I Would Do With More Time

With more time, the following improvements would be explored:

- Cross-validation instead of a single train/test split for more robust model comparison.[file:1]
- Hyperparameter tuning for Random Forest and Gradient Boosting.
- Additional feature engineering, such as time-derived features from `sleep_time` and `wake_time`.
- Saving metrics and fitted models into `artifacts/` for reproducible experiment tracking.
- Adding automated tests and argument-based configurability for cleaner experimentation.

## 11. Notes

- The pipeline uses the relative database path `data/score.db`, following the assessment instructions.
- The SQLite database file itself is not committed to the repository, also following the assessment requirements.