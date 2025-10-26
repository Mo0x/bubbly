#!/usr/bin/env python3
"""
Bubbly v1.1 – multi-source Bubble Phase Monitor
"""

import os
from pathlib import Path
import urllib.request, urllib.parse, json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

# === Keys ===

def load_api_keys(env_path: str = "apikeys.env") -> None:
    """Populate missing env vars from simple KEY=VALUE file."""
    path = Path(env_path)
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key and value and key not in os.environ:
            os.environ[key] = value


load_api_keys()

FRED_KEY = os.environ.get("FRED_API_KEY", "")
QUANDL_KEY = os.environ.get("QUANDL_API_KEY", "")

# === Utilities ===

def fetch_fred(series_id: str, freq=None) -> pd.Series:
    """Fetch FRED series and return as pandas Series"""
    params = {
        "series_id": series_id,
        "file_type": "json",
    }
    if FRED_KEY:
        params["api_key"] = FRED_KEY
    if freq:
        params["frequency"] = freq
    url = "https://api.stlouisfed.org/fred/series/observations?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode())

    vals, dates = [], []
    for o in data.get("observations", []):
        try:
            vals.append(float(o["value"]))
            dates.append(pd.to_datetime(o["date"]))
        except ValueError:
            continue
    return pd.Series(vals, index=dates).dropna()


def fetch_yahoo(symbol, period="max") -> pd.Series:
    """Fetch a single Yahoo Finance series and return a clean 1D Series."""
    df = yf.download(
        symbol, period=period, progress=False, auto_adjust=False, group_by="column"
    )
    if isinstance(df, pd.DataFrame):
        # prefer 'Adj Close' if present
        if "Adj Close" in df.columns:
            s = df["Adj Close"]
        elif "Close" in df.columns:
            s = df["Close"]
        else:
            # fallback to first numeric column
            s = df.select_dtypes(include="number").iloc[:, 0]
    else:
        s = df  # already a Series

    s = s.dropna()
    s.name = symbol
    return s


def fetch_quandl(code: str) -> pd.Series:
    """Fetch from Nasdaq Data Link (ex-Quandl)"""
    url = f"https://data.nasdaq.com/api/v3/datasets/{code}.json?api_key={QUANDL_KEY}"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode())
    df = pd.DataFrame(data["dataset"]["data"], columns=data["dataset"]["column_names"])
    s = pd.Series(df["Value"].astype(float).values, index=pd.to_datetime(df["Date"]))
    s.name = code
    return s.sort_index()

def zscore(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 3:
        return float("nan")
    z = (s.iloc[-1] - s.mean()) / s.std()
    if hasattr(z, "item"):
        return float(z.item())
    return float(z)


def expanding_zscore(series: pd.Series, min_periods: int = 24) -> pd.Series:
    """Compute expanding z-score to avoid look-ahead bias in history."""
    exp = series.expanding(min_periods=min_periods)
    mean = exp.mean()
    std = exp.std().replace(0, np.nan)
    return (series - mean) / std


def ensure_series(obj, name: str) -> pd.Series:
    """Ensure we are working with a single pandas Series."""
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise ValueError(f"{name} expected 1 column, got {obj.shape[1]}")
        obj = obj.iloc[:, 0]
    obj = obj.copy()
    obj.name = name
    return obj


def extend_gdp_with_nowcast(gdp: pd.Series, gdpnow: pd.Series) -> pd.Series:
    """Extend quarterly GDP levels with Atlanta Fed GDPNow projections."""
    gdp = ensure_series(gdp, "GDP").sort_index()
    gdpnow = ensure_series(gdpnow, "GDPNow").sort_index()

    if gdp.empty or gdpnow.empty:
        return gdp

    last_actual = gdp.index.max()
    future_nowcasts = gdpnow[gdpnow.index > last_actual]
    if future_nowcasts.empty:
        return gdp

    prior_level = gdp.iloc[-1]
    projected_levels = []
    projected_idx = []
    for ts, ann_growth in future_nowcasts.items():
        # GDPNow is an annualized quarterly growth rate (SAAR %)
        quarterly_growth = (1 + ann_growth / 100.0) ** 0.25 - 1
        prior_level = prior_level * (1 + quarterly_growth)
        projected_idx.append(ts)
        projected_levels.append(prior_level)

    extension = pd.Series(projected_levels, index=projected_idx, name="GDP")
    return pd.concat([gdp, extension])


def compute_weekly_m2_yoy() -> tuple[pd.Series, pd.Series]:
    """Return weekly M2 levels and YoY % derived from WM2NS."""
    m2_weekly = ensure_series(fetch_fred("WM2NS"), "M2")
    m2_weekly = m2_weekly.asfreq("W-MON")
    m2_weekly = m2_weekly.ffill()
    m2_yoy_weekly = m2_weekly.pct_change(52) * 100
    return m2_weekly, m2_yoy_weekly


def fetch_shiller_local_xls(path="data/ie_data.xls") -> pd.Series:
    """
    Read Shiller CAPE (cyclically adjusted P/E, column 'CAPE') from local Excel file
    downloaded from shillerdata.com
    """
    # Read the sheet and normalize column names
    df = pd.read_excel(path, sheet_name="Data", skiprows=7)
    df.columns = [str(c).strip() for c in df.columns]

    # Basic column sanity
    if "Date" not in df.columns:
        raise RuntimeError("No 'Date' column found in Shiller file")
    if "CAPE" not in df.columns:
        raise RuntimeError("No 'CAPE' column found in Shiller file")

    # Drop fully empty rows first (this avoids NaN crashing conversion)
    df = df.dropna(subset=["Date", "CAPE"], how="any")

    # Convert Shiller fractional date (e.g. 2025.10) -> pandas Timestamp(2025-10-01)
    def ym_to_timestamp(x):
        # x is numeric like 2025.10 or 1999.03
        # logic: integer part = year, decimal part*100 = month
        try:
            x = float(x)
        except Exception:
            return pd.NaT

        year = int(x)
        month_frac = x - year
        month = int(round(month_frac * 100))

        # guard bad rows
        if month < 1 or month > 12:
            return pd.NaT

        return pd.Timestamp(year=year, month=month, day=1)

    df["ts"] = df["Date"].apply(ym_to_timestamp)
    df = df.dropna(subset=["ts", "CAPE"])

    # Build time series
    s = pd.Series(df["CAPE"].astype(float).values, index=df["ts"])
    s.name = "CAPE"
    s = s.sort_index()

    # Optional: keep last ~40 years (stabilizes z-scores)
    cutoff = s.index.max() - pd.DateOffset(years=40)
    s = s[s.index >= cutoff]

    return s

def latest_float(series: pd.Series) -> float:
    s = series.dropna()
    val = s.iloc[-1]
    if hasattr(val, "item"):
        val = val.item()
    return float(val)


# === Fetch data ===
print("Fetching data ...")

gdp = extend_gdp_with_nowcast(fetch_fred("GDP", freq="q"), fetch_fred("GDPNOW", freq="q"))
m2_weekly, m2_yoy_weekly = compute_weekly_m2_yoy()
ci_loans = fetch_fred("BUSLOANS")
hy_spread = fetch_fred("BAMLH0A0HYM2")
wilshire = fetch_yahoo("^W5000")
sp500 = fetch_yahoo("^GSPC")
vix = fetch_yahoo("^VIX")
cape = fetch_shiller_local_xls("data/ie_data.xls")

# === Compute Buffett ratio proxy ===
# --- Buffett ratio (SP500 / GDP * 100) ---

# Make sure both are Series (not DataFrames)
if isinstance(sp500, pd.DataFrame):
    if "Adj Close" in sp500.columns:
        sp500 = sp500["Adj Close"]
    else:
        sp500 = sp500.iloc[:, 0]

wilshire = ensure_series(wilshire, "Wilshire5000")
sp500 = ensure_series(sp500, "SP500")
gdp = ensure_series(gdp, "GDP")

# Resample both to month-end frequency
gdp_q = gdp.copy()
gdp_q.index = gdp_q.index + pd.offsets.QuarterEnd(0)
gdp_m = gdp_q.resample("ME").ffill()
wilshire_m = wilshire.resample("ME").last()
sp_m = sp500.resample("ME").last()

# Align and compute
buffett_df = pd.concat([wilshire_m, gdp_m], axis=1, join="inner")
buffett_df.columns = ["Wilshire", "GDP"]

buffett = (buffett_df["Wilshire"] / buffett_df["GDP"]) * 100.0
buffett = buffett.dropna()
buffett.name = "Buffett % GDP"

# Track latest source dates
leverage_hist = ensure_series(ci_loans, "CI_Loans").resample("ME").last()
leverage_yoy_hist = leverage_hist.pct_change(12) * 100
hy_hist = ensure_series(hy_spread, "HY_Spread").resample("ME").mean()

source_dates = {
    "Buffett_ratio": buffett.index.max(),
    "CAPE": cape.index.max(),
    "CI_Loans_YoY": leverage_yoy_hist.dropna().index.max(),
    "M2_YoY": m2_yoy_weekly.dropna().index.max(),
    "VIX": vix.index.max(),
    "HY_Spread": ensure_series(hy_spread, "HY_Spread").index.max(),
}

weights = {
    "Buffett_ratio": 0.3,
    "CAPE": 0.3,
    "CI_Loans_YoY": 0.2,
    "M2_YoY": -0.1,
    "VIX": -0.05,
    "HY_Spread": -0.05,
}

# === Historical composites ===
# Align raw indicator series at month-end frequency
buffett_hist = ensure_series(buffett, "Buffett")
cape_hist = ensure_series(cape, "CAPE").resample("ME").last()
m2_hist = ensure_series(m2_weekly, "M2").resample("ME").last()
m2_yoy_hist = ensure_series(m2_yoy_weekly, "M2_YoY").resample("ME").last()
vix_hist = ensure_series(vix, "VIX").resample("ME").last()
sp_hist = ensure_series(sp500, "SP500").resample("ME").last()
hy_hist = hy_hist
leverage_yoy_hist = leverage_yoy_hist

hist_df = pd.DataFrame(
    {
        "Buffett": buffett_hist,
        "CAPE": cape_hist,
        "M2_YoY": m2_yoy_hist,
        "VIX": vix_hist,
        "CI_Loans_YoY": leverage_yoy_hist,
        "HY_Spread": hy_hist,
        "SP500": sp_hist,
    }
).dropna()

hist_df["Buffett_z"] = expanding_zscore(hist_df["Buffett"])
hist_df["CAPE_z"] = expanding_zscore(hist_df["CAPE"])
hist_df["M2_YoY_z"] = expanding_zscore(hist_df["M2_YoY"])
hist_df["VIX_z"] = expanding_zscore(hist_df["VIX"])
hist_df["CI_Loans_z"] = expanding_zscore(hist_df["CI_Loans_YoY"])
hist_df["HY_Spread_z"] = expanding_zscore(hist_df["HY_Spread"])
hist_df = hist_df.dropna(
    subset=[
        "Buffett_z",
        "CAPE_z",
        "M2_YoY_z",
        "VIX_z",
        "CI_Loans_z",
        "HY_Spread_z",
    ]
)

hist_df["Composite"] = (
    weights["Buffett_ratio"] * hist_df["Buffett_z"]
    + weights["CAPE"] * hist_df["CAPE_z"]
    + weights["CI_Loans_YoY"] * hist_df["CI_Loans_z"]
    + weights["M2_YoY"] * hist_df["M2_YoY_z"]
    + weights["VIX"] * hist_df["VIX_z"]
    + weights["HY_Spread"] * hist_df["HY_Spread_z"]
)
hist_df["SPX_Drawdown"] = hist_df["SP500"] / hist_df["SP500"].cummax() - 1.0

# === Latest snapshot ===
buffett_z = float(hist_df["Buffett_z"].iloc[-1])
cape_z = float(hist_df["CAPE_z"].iloc[-1])
m2_z = float(hist_df["M2_YoY_z"].iloc[-1])
vix_z = float(hist_df["VIX_z"].iloc[-1])
leverage_z = float(hist_df["CI_Loans_z"].iloc[-1])
hy_z = float(hist_df["HY_Spread_z"].iloc[-1])

buffett_latest = float(hist_df["Buffett"].iloc[-1])
cape_latest = float(hist_df["CAPE"].iloc[-1])
m2_latest = float(hist_df["M2_YoY"].iloc[-1])
vix_latest = float(hist_df["VIX"].iloc[-1])
leverage_latest = float(hist_df["CI_Loans_YoY"].iloc[-1])
hy_latest = float(hist_df["HY_Spread"].iloc[-1])

composite_last_date = hist_df.index[-1]

# Build dataframe row by row
df = pd.DataFrame(
    {
        "indicator": [
            "Buffett_ratio",
            "CAPE",
            "CI_Loans_YoY",
            "M2_YoY",
            "VIX",
            "HY_Spread",
        ],
        "Z-score": [
            buffett_z,
            cape_z,
            leverage_z,
            m2_z,
            vix_z,
            hy_z,
        ],
        "Latest_value": [
            buffett_latest,
            cape_latest,
            leverage_latest,
            m2_latest,
            vix_latest,
            hy_latest,
        ],
    }
).set_index("indicator")

df["Weight"] = df.index.map(weights.get)
df["Contribution"] = df["Z-score"] * df["Weight"]
df["Last_date"] = df.index.map(
    lambda name: source_dates.get(name).date().isoformat()
    if pd.notna(source_dates.get(name))
    else "N/A"
)

comp = float(hist_df["Composite"].iloc[-1])
if comp < 1.0:
    phase = "Expansion"
elif comp < 2.0:
    phase = "Euphoria"
else:
    phase = "Instability"

# Diagnostic split between valuation "pressure" and liquidity/volatility "triggers"
pressure = float(
    np.mean(
        [
            hist_df["Buffett_z"].iloc[-1],
            hist_df["CAPE_z"].iloc[-1],
            hist_df["CI_Loans_z"].iloc[-1],
        ]
    )
)
trigger = float(
    np.mean(
        [
            -hist_df["M2_YoY_z"].iloc[-1],
            -hist_df["VIX_z"].iloc[-1],
            -hist_df["HY_Spread_z"].iloc[-1],
        ]
    )
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 100)
print("\n--- BUBBLY REPORT ---")
print(df.round(2))
print(f"\nComposite phase index: {comp:.2f} → {phase}")
print(
    f"Valuation pressure: {pressure:.2f} | Liquidity/vol trigger: {trigger:.2f}"
)
print(f"\nData coverage through: {composite_last_date.date().isoformat()}")
print("Latest source timestamps:")
for name, ts in sorted(source_dates.items(), key=lambda kv: kv[1]):
    stamp = ts.date().isoformat() if pd.notna(ts) else "N/A"
    print(f"  {name}: {stamp}")
print("\nDebug check:")
for name in ["Buffett_ratio", "CAPE", "CI_Loans_YoY", "M2_YoY", "VIX", "HY_Spread"]:
    print(
        f"{name} z={df.loc[name, 'Z-score']:.2f} weight={df.loc[name, 'Weight']:.2f}"
        f" contribution={df.loc[name, 'Contribution']:.2f}"
    )

os.makedirs("output", exist_ok=True)
df.to_html("output/bubbly_report.html")
print("\nReport saved to output/bubbly_report.html")

# Persist historical backtest
history_cols = [
    "Buffett",
    "CAPE",
    "M2_YoY",
    "VIX",
    "CI_Loans_YoY",
    "HY_Spread",
    "Buffett_z",
    "CAPE_z",
    "M2_YoY_z",
    "VIX_z",
    "CI_Loans_z",
    "HY_Spread_z",
    "Composite",
    "SP500",
    "SPX_Drawdown",
]
hist_df[history_cols].to_csv("output/bubbly_history.csv", float_format="%.4f")

# Plot composite vs SPX drawdown
fig, ax1 = plt.subplots(figsize=(11, 6))
ax1.plot(hist_df.index, hist_df["Composite"], color="tab:blue", label="Composite Index")
ax1.axhline(1.0, color="tab:blue", linestyle="--", linewidth=0.8, alpha=0.6)
ax1.axhline(2.0, color="tab:blue", linestyle=":", linewidth=0.8, alpha=0.6)
ax1.set_ylabel("Composite Z-score")
ax1.set_xlabel("Date")

ax2 = ax1.twinx()
ax2.fill_between(
    hist_df.index,
    hist_df["SPX_Drawdown"],
    0,
    color="tab:red",
    alpha=0.3,
    label="SPX Drawdown",
)
ax2.set_ylabel("SPX Drawdown")
ax2.set_ylim(-0.8, 0.05)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

ax1.set_title("Bubbly Composite vs. S&P 500 Drawdowns")
fig.tight_layout()
plt.savefig("output/bubbly_backtest.png", dpi=200)
plt.close(fig)

print("Historical series saved to output/bubbly_history.csv")
print("Backtest chart saved to output/bubbly_backtest.png")
