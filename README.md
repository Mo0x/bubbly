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
**Status: done with operational workaround**
- M2 signal is currently sourced from FRED weekly money stock (`WM2NS`) and converted to YoY in code.
- Operational note remembered from previous work: when FRED had intermittent/API issues, **M2 had to be updated manually** to keep runs unblocked.
- Action kept in mind: if FRED fails for M2 again, update M2 input manually (or cache local series) before rerunning the model.

### M3 (future hardening candidates)
**Status: open**
- Improve data-source robustness/caching (especially for FRED outages).
- Consider replacing C&I proxy with higher-fidelity margin debt feed when available.

## How to rerun

1. (Optional) Create `apikeys.env` in repo root:
   - `FRED_API_KEY=...`
   - `QUANDL_API_KEY=...`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run:
   ```bash
   python bubble_watch.py
   ```
4. Open artifacts in `output/`.

## Practical note about M2
If you observe missing/buggy FRED responses for `WM2NS`, treat M2 as a manual maintenance point for the run (inject/patch the latest M2 values locally, then rerun). This matches the M2 workaround used previously.
