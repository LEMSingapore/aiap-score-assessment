"""Central configuration for the AIAP score assessment pipeline.

Single source of truth for filesystem paths, the target column, the
train/test split settings, and the feature-group definitions used
across preprocessing and modelling.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DB_PATH = DATA_DIR / "score.db"

TARGET_COL = "final_test"

RANDOM_STATE = 42
TEST_SIZE = 0.2

NUM_FEATURES = [
    "number_of_siblings",
    "hours_per_week",
    "attendance_rate",
    "class_size",  # was: classsize
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

ID_COLUMNS = ["index", "student_id", "bag_color"]

LOW_SIGNAL_COLUMNS = ["gender", "wake_time", "mode_of_transport"]
