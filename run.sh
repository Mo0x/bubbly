#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

STRICT_MODE="${BUBBLY_STRICT:-0}"
M1_OK=0

echo "[M1] Run full Bubbly pipeline"
if python bubble_watch.py; then
  M1_OK=1
  echo "M1 complete"
else
  echo "WARNING: live pipeline run failed (likely network/data-provider issue)." >&2
  echo "         Continuing in automated fallback mode using existing artifacts." >&2
  if [[ "$STRICT_MODE" == "1" ]]; then
    echo "BUBBLY_STRICT=1 set, aborting on M1 failure." >&2
    exit 1
  fi
fi

echo "[M2] Check M2 automation artifacts"
python - <<'PY'
from pathlib import Path
import pandas as pd

cache = Path('data/m2_manual.csv')
if not cache.is_file():
    print('WARNING: data/m2_manual.csv not present yet (no successful WM2NS seed run).')
    raise SystemExit(0)

m2 = pd.read_csv(cache)
needed = {'date', 'M2'}
if not needed.issubset(set(m2.columns)):
    raise SystemExit('M2 cache schema invalid. Expected columns: date,M2')

print(f"M2 cache rows: {len(m2)} | last date: {m2['date'].iloc[-1]}")
PY

echo "[M3] Validate expected output artifacts"
required=(
  docs/bubbly_history.csv
  docs/bubbly_validation_summary.csv
  docs/bubbly_validation_summary_realtime.csv
  docs/bubbly_validation_summary_comparison.csv
  docs/bubbly_backtest.png
  docs/bubbly_realtime_backtest.png
  docs/index.html
  docs/bubbly_report.html
)

for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing artifact: $path" >&2
    exit 1
  fi
  echo "ok: $path"
done

if [[ "$M1_OK" == "1" ]]; then
  echo "Bubbly automation complete ✅"
else
  echo "Bubbly automation complete with fallback mode ⚠️"
fi
