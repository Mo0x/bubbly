# Bubbly — Bubble Phase Monitor

Bubbly is a macro/market composite model that tracks bubble regimes using valuation pressure + liquidity/credit triggers. It generates:
- a historical dataset (`output/bubbly_history.csv`),
- static backtest charts (`output/bubbly_backtest.png`, `output/bubbly_realtime_backtest.png`),
- an interactive dashboard (`output/bubbly_dashboard.html`),
- a styled HTML summary report (`output/bubbly_report.html`),
- and validation tables for forward drawdown hit-rates.

## Where we left off (project status)

### ✅ What is already done
- Multi-source data ingestion is wired (FRED, Yahoo Finance, local Shiller fallback for CAPE).
- Composite model is implemented with `Pressure` + `Trigger` blocks and z-score normalization.
- Historical and pseudo real-time composite backtesting are implemented.
- Forward drawdown validation (6M/12M horizons) is implemented for both full-sample and pseudo real-time signals.
- Outputs are already generated under `output/` and include dashboard/report artifacts.

### 📌 Latest generated snapshot in repo
From the existing checked-in artifacts:
- Latest data coverage appears to be **2025-09-30** (`output/bubbly_report.html`).
- Latest composite level is about **1.03**, classified as **Euphoria**.
- Latest validation comparison shows pseudo real-time hit-rate higher than full-sample for both 6M and 12M windows (`output/bubbly_validation_summary_comparison.csv`).

## Milestones

### M1 (core model + outputs)
**Status: done**
- Composite regime engine working.
- Backtest + report/dashboard output generation working.

### M2 (money/liquidity data reliability)
**Status: automated**
- M2 signal is sourced from FRED weekly money stock (`WM2NS`) and converted to YoY in code.
- The pipeline auto-refreshes a local cache at `data/m2_manual.csv` after successful FRED pulls.
- If FRED is temporarily unavailable, the run automatically falls back to `data/m2_manual.csv` (no manual editing required for normal outages).

### M3 (automation and validation)
**Status: automated run wrapper added**
- `./run.sh` now executes the full pipeline end-to-end.
- It verifies M2 cache integrity and checks all expected output artifacts.
- Future optional enhancement: replace C&I proxy with higher-fidelity margin debt feed when available.

## How to run (fully automated)

1. (Optional) Create `apikeys.env` in repo root:
   - `FRED_API_KEY=...`
   - `QUANDL_API_KEY=...`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run everything (M1 + M2 + M3 checks):
   ```bash
   ./run.sh
   ```
   Optional strict mode (fail immediately if live data pull fails):
   ```bash
   BUBBLY_STRICT=1 ./run.sh
   ```
4. Open artifacts in `output/`.

## M2 resilience behavior
- Primary source: FRED `WM2NS`.
- Automatic fallback: `data/m2_manual.csv`.
- First-time setup note: if FRED is down and no local cache exists yet, run once when FRED is reachable to seed the cache.
