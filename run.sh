#!/usr/bin/env bash

set -euo pipefail
export PYTHONUNBUFFERED=1


# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Log everything to bubbly_run.log and console
LOG_FILE="bubbly_run.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Allow overriding python command, default to venv or system python
if [[ -x "venv/bin/python" ]]; then
    PYTHON_CMD="venv/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
    PYTHON_CMD=".venv/bin/python"
else
    PYTHON_CMD="python3"
fi

echo "========================================================"
echo "Starting Bubbly Pipeline"
echo "Root Directory: $SCRIPT_DIR"
echo "Python Command: $PYTHON_CMD"
echo "========================================================"

# --- Execution ---
echo "[1/3] Running bubble_watch.py..."
$PYTHON_CMD bubble_watch.py

# --- Verification ---
echo "[2/3] Verifying artifacts..."

REQUIRED_ARTIFACTS=(
    "docs/index.html"
    "docs/bubbly_history.csv"
    "docs/bubbly_report.html"
)

MISSING=0
for artifact in "${REQUIRED_ARTIFACTS[@]}"; do
    if [[ ! -f "$artifact" ]]; then
        echo "❌ Missing: $artifact"
        MISSING=1
    else
        echo "✅ Found: $artifact"
    fi
done

if [[ "$MISSING" -eq 1 ]]; then
    echo "========================================================"
    echo "❌ Pipeline completed with MISSING artifacts."
    echo "========================================================"
    exit 1
fi

echo "========================================================"
echo "✅ Bubbly Pipeline Completed Successfully."
echo "========================================================"
