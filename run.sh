#!/usr/bin/env bash
set -euo pipefail

# Activate virtual environment if present
if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "[run.sh] Starting AIAP score.db pipeline..."
python -m src.main "$@"
echo "[run.sh] Pipeline completed."
