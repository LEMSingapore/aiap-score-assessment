from pathlib import Path

# Project paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

DB_PATH = DATA_DIR / "score.db"

# Targets (update once you inspect score.db)
TARGET_REGRESSION = None
TARGET_CLASSIFICATION = None
