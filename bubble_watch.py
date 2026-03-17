#!/usr/bin/env python3
"""
Bubbly v1.1 – multi-source Bubble Phase Monitor
Refactored Version
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

# Import from new library modules
from lib.data_loaders import (
    load_api_keys,
    fetch_fred,
    fetch_yahoo,
    fetch_cape_series,
    ensure_series,
    compute_weekly_m2_yoy,
)
from lib.indicators import (
    monthly_gdp_with_proxy,
    expanding_zscore,
    extend_gdp_with_nowcast,
    extend_monthly_denominator_with_proxy,
)
from lib.composite import (
    compute_realtime_composite,
    build_validation_tables,
)
from lib.visualization import (
    build_plotly_dashboard,
    save_html_report,
    generate_static_plots,
)

# === Main Pipeline ===

def main():
    load_api_keys()

    # === Fetch data ===
    print("Fetching data ...")

    # GDP & GDPNow
    gdp = extend_gdp_with_nowcast(fetch_fred("GDP", freq="q"), fetch_fred("GDPNOW", freq="q"))

    # M2
    m2_weekly, m2_yoy_weekly = compute_weekly_m2_yoy()

    # Other FRED series
    ci_loans_weekly = fetch_fred("TOTCI")
    hy_spread = fetch_fred("BAMLH0A0HYM2")
    walcl = fetch_fred("WALCL")
    rrp = fetch_fred("RRPONTSYAWARD")
    ig_spread = fetch_fred("BAA10Y")
    indpro = fetch_fred("INDPRO")
    personal_income = fetch_fred("PI")

    # Yahoo series
    wilshire = fetch_yahoo("^W5000")
    sp500 = fetch_yahoo("^GSPC")
    vix = fetch_yahoo("^VIX")
    vxv = fetch_yahoo("^VIX3M")

    # CAPE
    cape = fetch_cape_series("data/ie_data.xls")

    # === Process & align data ===
    
    # 1. Buffett ratio
    if isinstance(sp500, pd.DataFrame):
        sp500 = sp500["Adj Close"] if "Adj Close" in sp500.columns else sp500.iloc[:, 0]
    
    wilshire = ensure_series(wilshire, "Wilshire5000")
    sp500 = ensure_series(sp500, "SP500")
    gdp = ensure_series(gdp, "GDP")

    gdp_m = monthly_gdp_with_proxy(gdp, indpro)
    buffett_denominator = extend_monthly_denominator_with_proxy(
        gdp,
        personal_income,
        proxy_name="PersonalIncome",
    )
    if buffett_denominator.index.max() <= gdp_m.index.max():
        buffett_denominator = gdp_m
    wilshire_m = wilshire.resample("ME").last()
    
    # Align for Buffett
    buffett_df = pd.concat([wilshire_m, buffett_denominator], axis=1, join="inner")
    buffett_df.columns = ["Wilshire", "Denominator"]
    buffett = (buffett_df["Wilshire"] / buffett_df["Denominator"]) * 100.0
    buffett = buffett.dropna()
    buffett.name = "Buffett-style valuation ratio"

    # 2. Process weekly/daily into monthly history
    ci_loans_weekly = ensure_series(ci_loans_weekly, "CI_Loans").asfreq("W-WED").ffill()
    weekly_index = ci_loans_weekly.index
    ci_loans_yoy_weekly = ci_loans_weekly.pct_change(52) * 100
    
    leverage_yoy_hist = ensure_series(ci_loans_yoy_weekly, "CI_Loans_YoY").resample("ME").last()

    walcl_weekly = ensure_series(walcl, "FedBalanceSheet").asfreq("W-WED").ffill()
    walcl_weekly = walcl_weekly.reindex(weekly_index).ffill()
    walcl_yoy_weekly = walcl_weekly.pct_change(52) * 100
    walcl_hist = ensure_series(walcl_yoy_weekly, "FedBalanceSheet_YoY").resample("ME").last()

    rrp_daily = ensure_series(rrp, "RRP").sort_index()
    rrp_weekly = rrp_daily.resample("W-FRI").last() # Fallback to Friday
    rrp_weekly = rrp_weekly.reindex(weekly_index, fill_value=0.0)
    rrp_yoy_weekly = rrp_weekly - rrp_weekly.shift(52)
    rrp_hist = ensure_series(rrp_yoy_weekly, "RRP_YoY").resample("ME").last()

    hy_series = ensure_series(hy_spread, "HY_Spread")
    hy_series = hy_series.reindex(wilshire.index, method="ffill").bfill()
    hy_hist = hy_series.resample("ME").mean()

    ig_series = ensure_series(ig_spread, "IG_Spread")
    ig_series = ig_series.reindex(wilshire.index, method="ffill").bfill()
    ig_hist = ig_series.resample("ME").mean()

    vix_series = ensure_series(vix, "VIX")
    try:
        vxv_series = ensure_series(vxv, "VXV")
        vol_term_daily = pd.concat([vix_series, vxv_series], axis=1, join="inner")
        # Rename columns to handle potential duplicate names if series handle differs
        vol_term_daily.columns = ["VIX", "VXV"] 
        vol_term_daily = (vol_term_daily["VIX"] / vol_term_daily["VXV"]).rename("VolTerm")
        vol_term_daily = vol_term_daily.replace([np.inf, -np.inf], np.nan)
    except Exception:
        vol_term_daily = pd.Series(dtype=float, name="VolTerm")

    if vol_term_daily.empty:
        vol_term_hist = vix_series.resample("ME").last().apply(lambda _: 1.0)
        vol_term_hist.name = "VolTerm"
    else:
        vol_term_hist = ensure_series(vol_term_daily, "VolTerm").resample("ME").mean()

    # Track latest source dates for report
    source_dates = {
        "Buffett_ratio": buffett.index.max(),
        "CAPE": cape.index.max(),
        "CI_Loans_YoY": leverage_yoy_hist.dropna().index.max(),
        "M2_YoY": m2_yoy_weekly.dropna().index.max(),
        "FedBalanceSheet_YoY": walcl_yoy_weekly.dropna().index.max(),
        "RRP_YoY": rrp_yoy_weekly.dropna().index.max(),
        "HY_Spread": hy_series.index.max(),
        "IG_Spread": ig_series.index.max(),
        "VIX": vix_series.index.max(),
        "VolTerm": vol_term_daily.index.max() if not vol_term_daily.empty else pd.NaT,
    }

    # Weights configuration
    raw_weights = pd.Series(
        {
            "Buffett_ratio": 0.30,
            "CAPE": 0.30,
            "CI_Loans_YoY": 0.22,
            "M2_YoY": -0.25,
            "FedBalanceSheet_YoY": -0.20,
            "RRP_YoY": 0.15,
            "HY_Spread": 0.24,
            "IG_Spread": 0.12,
            "VIX": -0.18,
            "VolTerm": 0.24,
        }
    )
    weight_scale = 2.0
    weights = (raw_weights / raw_weights.abs().sum() * weight_scale).to_dict()

    # === Build Composite History ===
    
    # Align all to monthly end
    buffett_hist = ensure_series(buffett, "Buffett")
    common_index = buffett_hist.index

    m_cape = ensure_series(cape, "CAPE").resample("ME").last()
    cape_hist = m_cape.reindex(common_index).ffill().bfill()
    
    m2_hist = ensure_series(m2_weekly, "M2").resample("ME").last().reindex(common_index).ffill().bfill()
    
    m2_yoy_hist = (
        ensure_series(m2_yoy_weekly, "M2_YoY")
        .resample("ME")
        .last()
        .reindex(common_index)
        .ffill()
        .bfill()
        .fillna(0.0)
    )
    
    vix_hist = ensure_series(vix, "VIX").resample("ME").last().reindex(common_index).ffill().bfill()
    sp_hist = ensure_series(sp500, "SP500").resample("ME").last().reindex(common_index).ffill().bfill()
    
    hy_hist = hy_hist.reindex(common_index).ffill().bfill()
    leverage_yoy_hist = leverage_yoy_hist.reindex(common_index).ffill().bfill().fillna(0.0)
    walcl_hist = walcl_hist.reindex(common_index).ffill().bfill().fillna(0.0)
    rrp_hist = rrp_hist.reindex(common_index).ffill().bfill().fillna(0.0)
    ig_hist = ig_hist.reindex(common_index).ffill().bfill()
    vol_term_hist = vol_term_hist.reindex(common_index).ffill().bfill().fillna(1.0)
    
    hist_df = pd.DataFrame(
        {
            "Buffett": buffett_hist,
            "CAPE": cape_hist,
            "M2_YoY": m2_yoy_hist,
            "VIX": vix_hist,
            "CI_Loans_YoY": leverage_yoy_hist,
            "HY_Spread": hy_hist,
            "FedBalanceSheet_YoY": walcl_hist,
            "RRP_YoY": rrp_hist,
            "IG_Spread": ig_hist,
            "VolTerm": vol_term_hist,
            "SP500": sp_hist,
        }
    ).dropna()
    
    # Compute Z-scores
    hist_df["Buffett_z"] = expanding_zscore(hist_df["Buffett"])
    hist_df["CAPE_z"] = expanding_zscore(hist_df["CAPE"])
    hist_df["M2_YoY_z"] = expanding_zscore(hist_df["M2_YoY"])
    hist_df["VIX_z"] = expanding_zscore(hist_df["VIX"])
    hist_df["CI_Loans_z"] = expanding_zscore(hist_df["CI_Loans_YoY"])
    hist_df["HY_Spread_z"] = expanding_zscore(hist_df["HY_Spread"])
    hist_df["FedBalanceSheet_z"] = expanding_zscore(hist_df["FedBalanceSheet_YoY"])
    hist_df["RRP_z"] = expanding_zscore(hist_df["RRP_YoY"])
    hist_df["IG_Spread_z"] = expanding_zscore(hist_df["IG_Spread"])
    hist_df["VolTerm_z"] = expanding_zscore(hist_df["VolTerm"])
    
    hist_df = hist_df.dropna(subset=[c for c in hist_df.columns if c.endswith("_z")])

    # Diagnostic components
    pressure_series = hist_df[["Buffett_z", "CAPE_z", "CI_Loans_z"]].mean(axis=1)
    
    trigger_components = pd.DataFrame(
        {
            "M2": -hist_df["M2_YoY_z"],
            "FedBalanceSheet": -hist_df["FedBalanceSheet_z"],
            "RRP": hist_df["RRP_z"],
            "HY": hist_df["HY_Spread_z"],
            "IG": hist_df["IG_Spread_z"],
            "VolTerm": hist_df["VolTerm_z"],
            "VIX": -hist_df["VIX_z"],
        }
    )
    trigger_series = trigger_components.mean(axis=1)

    pressure_raw = hist_df[["Buffett", "CAPE", "CI_Loans_YoY"]].mean(axis=1)
    trigger_raw = (
        -hist_df["M2_YoY"]
        - hist_df["FedBalanceSheet_YoY"]
        + hist_df["RRP_YoY"]
        + hist_df["HY_Spread"]
        + hist_df["IG_Spread"]
        + hist_df["VolTerm"]
        - hist_df["VIX"]
    ) / 7.0

    hist_df["Pressure_z"] = pressure_series
    hist_df["Trigger_z"] = trigger_series
    hist_df["Pressure_raw"] = pressure_raw
    hist_df["Trigger_raw"] = trigger_raw
    
    hist_df["Composite"] = (
        weights["Buffett_ratio"] * hist_df["Buffett_z"]
        + weights["CAPE"] * hist_df["CAPE_z"]
        + weights["CI_Loans_YoY"] * hist_df["CI_Loans_z"]
        + weights["M2_YoY"] * hist_df["M2_YoY_z"]
        + weights["FedBalanceSheet_YoY"] * hist_df["FedBalanceSheet_z"]
        + weights["RRP_YoY"] * hist_df["RRP_z"]
        + weights["VIX"] * hist_df["VIX_z"]
        + weights["VolTerm"] * hist_df["VolTerm_z"]
        + weights["HY_Spread"] * hist_df["HY_Spread_z"]
        + weights["IG_Spread"] * hist_df["IG_Spread_z"]
    )
    hist_df["SPX_Drawdown"] = hist_df["SP500"] / hist_df["SP500"].cummax() - 1.0

    # Validation
    signals_full, summary_full = build_validation_tables(hist_df)
    
    lag_map = {
        "Buffett_ratio": 1,
        "CAPE": 3,
        "CI_Loans_YoY": 1,
        "M2_YoY": 1,
        "FedBalanceSheet_YoY": 1,
        "RRP_YoY": 1,
        "HY_Spread": 0,
        "IG_Spread": 0,
        "VIX": 0,
        "VolTerm": 0,
    }
    realtime_series = compute_realtime_composite(hist_df, weights, lag_map=lag_map)
    hist_df["Composite_real_time"] = realtime_series
    signals_rt, summary_rt = build_validation_tables(
        hist_df.dropna(subset=["Composite_real_time"]), composite_column="Composite_real_time"
    )
    
    summary_comparison = summary_full.merge(
        summary_rt,
        on="horizon_months",
        how="outer",
        suffixes=("_full", "_rt"),
    )
    summary_comparison["hit_rate_delta"] = (
        summary_comparison["hit_rate_rt"] - summary_comparison["hit_rate_full"]
    )
    
    # === Snapshot & Report Data ===
    
    # (Snapshot extraction logic omitted for brevity in thought but needed in code)
    # I'll implement the snapshot logic concisely using iloc[-1]
    
    latest = hist_df.iloc[-1]
    
    df_report = pd.DataFrame(
        {
            "indicator": list(weights.keys()),
            "Z-score": [
                latest["Buffett_z"], latest["CAPE_z"], latest["CI_Loans_z"],
                latest["M2_YoY_z"], latest["FedBalanceSheet_z"], latest["RRP_z"],
                latest["HY_Spread_z"], latest["IG_Spread_z"], latest["VIX_z"], latest["VolTerm_z"]
            ],  # Order matters! Map keys to columns carefully.
        }
    ).set_index("indicator")
    
    # Fix order to match weights keys
    # weights keys: Buffett_ratio, CAPE, CI_Loans_YoY, M2_YoY, FedBalanceSheet_YoY, RRP_YoY, HY_Spread, IG_Spread, VIX, VolTerm
    # df entries must match
    vals_z = [
        latest["Buffett_z"], latest["CAPE_z"], latest["CI_Loans_z"],
        latest["M2_YoY_z"], latest["FedBalanceSheet_z"], latest["RRP_z"],
        latest["HY_Spread_z"], latest["IG_Spread_z"], latest["VIX_z"], latest["VolTerm_z"]
    ]
    vals_raw = [
        latest["Buffett"], latest["CAPE"], latest["CI_Loans_YoY"],
        latest["M2_YoY"], latest["FedBalanceSheet_YoY"], latest["RRP_YoY"],
        latest["HY_Spread"], latest["IG_Spread"], latest["VIX"], latest["VolTerm"]
    ]
    
    df_report = pd.DataFrame(
        {
            "Z-score": vals_z,
            "Latest_value": vals_raw,
        },
        index=weights.keys() 
    )
    
    df_report["Weight"] = df_report.index.map(weights.get)
    df_report["Contribution"] = df_report["Z-score"] * df_report["Weight"]
    df_report["Last_date"] = df_report.index.map(
        lambda name: source_dates.get(name).date().isoformat()
        if pd.notna(source_dates.get(name))
        else "N/A"
    )

    comp = latest["Composite"]
    if comp < 1.0:
        phase = "Expansion"
    elif comp < 2.0:
        phase = "Euphoria"
    else:
        phase = "Instability"

    pressure = latest["Pressure_z"]
    trigger = latest["Trigger_z"]
    composite_last_date = hist_df.index[-1]
    
    # Print summary to stdout (keeping original behavior)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 100)
    print("\n--- BUBBLY REPORT ---")
    print(df_report.round(2))
    print(f"\nComposite phase index: {comp:.2f} → {phase}")
    print(f"Valuation pressure: {pressure:.2f} | Liquidity/vol trigger: {trigger:.2f}")
    
    # === Save Artifacts ===
    
    os.makedirs("docs", exist_ok=True)
    
    history_cols = [
        "Buffett", "CAPE", "M2_YoY", "VIX", "CI_Loans_YoY", 
        "FedBalanceSheet_YoY", "RRP_YoY", "HY_Spread", "IG_Spread", "VolTerm",
        "Pressure_raw", "Trigger_raw",
        "Buffett_z", "CAPE_z", "M2_YoY_z", "VIX_z", "CI_Loans_z", 
        "FedBalanceSheet_z", "RRP_z", "HY_Spread_z", "IG_Spread_z", "VolTerm_z",
        "Pressure_z", "Trigger_z", "Composite", "Composite_real_time", 
        "SP500", "SPX_Drawdown"
    ]
    hist_df[history_cols].to_csv("docs/bubbly_history.csv", float_format="%.4f")
    signals_full.to_csv("docs/bubbly_validation_signals_full.csv", index=False)
    summary_full.to_csv("docs/bubbly_validation_summary.csv", index=False)
    signals_rt.to_csv("docs/bubbly_validation_signals_realtime.csv", index=False)
    summary_rt.to_csv("docs/bubbly_validation_summary_realtime.csv", index=False)
    summary_comparison.to_csv("docs/bubbly_validation_summary_comparison.csv", index=False)
    
    event_dates = list(
        pd.to_datetime(
            signals_full.loc[signals_full["hit_threshold"], "signal_date"]
        ).dropna().unique()
    )
    event_dates = [d for d in event_dates if d in hist_df.index]
    
    # Static Plots
    generate_static_plots(hist_df, event_dates, "docs")
    
    # Interactive Dashboard & Report
    dashboard_path = build_plotly_dashboard(
        hist_df,
        event_dates=event_dates,
        output_path=Path("docs") / "index.html",
    )
    print(f"Interactive dashboard saved to {dashboard_path}")
    
    report_path = save_html_report(
        overview_df=df_report,
        validation_summary=summary_full,
        source_dates=source_dates,
        composite_value=comp,
        phase=phase,
        pressure=pressure,
        trigger=trigger,
        data_coverage=composite_last_date,
        output_dir="docs",
        history_chart_filename="bubbly_backtest.png",
    )
    print(f"Report saved to {report_path}")

    # Cleanup
    try:
        import multitasking
        multitasking.killall()
    except ImportError:
        pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
