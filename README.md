# AIAP Technical Assessment — score.db

**Name:** Chang Chee Young  
**Email:** cheeyoung.chang@gmail.com

## 1. Project Overview

This repository contains my solution for the AIAP technical assessment based on the `score.db` dataset.

It includes:

- An **EDA notebook** (`eda.ipynb`) for exploratory data analysis  
- A modular **ML pipeline** under `src/`  
- A single **entrypoint script** (`run.sh`) to reproduce the full workflow  

### Data & preprocessing

The `score.db` SQLite database contains 15,900 student-level rows and 18 columns, including:

- Target: `final_test` (numeric exam score)
- Demographics and context: `number_of_siblings`, `age`, `direct_admission`, `CCA`, `learning_style`, `tuition`, `n_male`, `n_female`
- Study behaviour: `hours_per_week`, `attendance_rate`, `sleep_time`, `wake_time`, `mode_of_transport`, `bag_color`, `gender`

The preprocessing pipeline in `src/preprocessing.py` applies the EDA decisions:

- Categorical normalisation:
  - Collapse messy `CCA` labels into four canonical categories: `Sports`, `Clubs`, `Arts`, `None`
  - Map `tuition` values from `{Yes, No, Y, N}` into clean `Yes`/`No`
- Target handling:
  - Drop rows with missing `final_test` (cannot be used for supervised training)
- Data quality fixes:
  - Treat negative ages as data errors; set them to missing and impute with the median age
  - Correct implausible ages 5 and 6 to 15 and 16 respectively (likely dropped leading digit)
- Attendance handling:
  - Create a binary indicator `attendance_rate_was_nan` to capture original missingness
  - Impute missing `attendance_rate` with the median
- Feature engineering and column selection:
  - Engineer `classsize = n_male + n_female` and drop `n_male` / `n_female`
  - Drop identifier and low-signal columns:
    - IDs: `index`, `student_id`
    - High-noise categoricals: `bag_color`, `gender`, `wake_time`, `mode_of_transport`
  - Drop `age` from the final feature set after cleaning due to very low variance

The final feature set used for modelling includes:

- Numeric: `number_of_siblings`, `hours_per_week`, `attendance_rate`, `classsize`
- Categorical: `direct_admission`, `CCA`, `learning_style`, `tuition`, `sleep_time`
- Indicator: `attendance_rate_was_nan`

## 2. Repository Structure

- `README.md` — this file; overview, instructions, and findings  
- `eda.ipynb` — exploratory data analysis on `score.db`  
- `run.sh` — bash script to execute the end‑to‑end pipeline  
- `requirements.txt` — Python dependencies for reproducibility  
- `.gitignore` — ignores virtualenvs, caches, and large local files  

- `src/`  
  - `main.py` — orchestrates the full pipeline  
  - `settings.py` — central configuration (paths, targets, flags)  
  - `preprocessing.py` — data loading and cleaning from `score.db`  
  - `feature_engineering.py` — feature creation / selection  
  - `models.py` — model training, evaluation, and persistence  

- `data/`  
  - `score.db` — assessment SQLite database (not tracked in git)  
  - other intermediate CSVs/parquets as needed  

- `artifacts/`  
  - trained models, metrics, and plots saved by the pipeline  

## 3. Setup Instructions

1. Clone this repository and move into it:

   \`\`\`bash
   git clone <your-github-url>.git
   cd aiap-score-assessment
   \`\`\`

2. Create and activate a virtual environment (Python 3.10+ recommended):

   \`\`\`bash
   python3 -m venv .venv
   source .venv/bin/activate
   \`\`\`

3. Install dependencies:

   \`\`\`bash
   pip install --upgrade pip
   pip install -r requirements.txt
   \`\`\`

4. Place the `score.db` file into the `data/` folder:

   \`\`\`bash
   mv /path/to/score.db data/score.db
   \`\`\`

## 4. How to Run

### 4.1 EDA (Part 1)

From the repo root (with the virtualenv activated):

\`\`\`bash
jupyter lab
\`\`\`

Open `eda.ipynb` and run through the notebook.  
This notebook:

- Loads tables from `score.db`  
- Explores distributions, missingness, and correlations  
- Documents assumptions and modelling decisions  

### 4.2 Pipeline (Part 2)

After configuring `src/settings.py`:

\`\`\`bash
./run.sh
\`\`\`

The script will:

1. Load and clean the data from `score.db`  
2. Perform feature engineering  
3. Train the model(s)  
4. Save metrics and artifacts under `artifacts/`  

See `src/main.py` for the detailed execution flow.

## 5. Configuration

Key configuration lives in `src/settings.py`, including:

- Input paths (e.g. `DATA_DIR`, `DB_PATH`)  
- Target column(s)  
- Train/validation split parameters  
- Model hyperparameters  

Change these values instead of hard‑coding paths or constants inside other modules.

## 6. Results Summary

- **Target:** Regression on `final_test` (student exam score, range 32–100)

- **Models evaluated:**

  | Model | MAE | RMSE | R² |
  |---|---|---|---|
  | RandomForestRegressor | 3.67 | 5.36 | 0.85 |
  | GradientBoostingRegressor | 4.83 | 6.47 | 0.78 |
  | LinearRegression (baseline) | 7.24 | 9.06 | 0.58 |

- **Best model:** `RandomForestRegressor` (default hyperparameters, `random_state=42`)

- **Key findings:**
  - Tree-based models significantly outperform Linear Regression, suggesting
    non-linear relationships between student habits, demographics, and exam scores.
  - Random Forest (RMSE 5.36, R² 0.85) explains 85% of variance in `final_test`,
    with an average prediction error of ~5 score points.
  - The gap between Linear Regression and Random Forest (RMSE 9.06 vs 5.36)
    confirms the value of ensemble methods on this dataset.

- **Known limitations:**
  - No hyperparameter tuning applied; default sklearn settings used throughout.
  - Median imputation for `attendance_rate` was computed on the full dataset
    rather than the training split only; a production pipeline would fit
    imputers on train data only to avoid leakage.
  - Results are from a single train/val/test split (no cross-validation).