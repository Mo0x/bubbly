#!/usr/bin/env python3
"""
Bubbly v1.1 – multi-source Bubble Phase Monitor
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
import urllib.request, urllib.parse, json

# === Keys ===
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
    df = yf.download(symbol, period=period, progress=False)
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

gdp = fetch_fred("GDP", freq="q")
m2 = fetch_fred("M2SL")
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

if isinstance(gdp, pd.DataFrame):
    gdp = gdp.iloc[:, 0]

# Resample both to month-end frequency
gdp_m = gdp.resample("ME").ffill()
sp_m = sp500.resample("ME").last()

# Align and compute
buffett_df = pd.concat([sp_m, gdp_m], axis=1, join="inner")
buffett_df.columns = ["SP500", "GDP"]

buffett = (buffett_df["SP500"] / buffett_df["GDP"]) * 100.0
buffett = buffett.dropna()
buffett.name = "Buffett % GDP"

# === Compute z-scores ===
# Compute individual z-scores
buffett_z = zscore(buffett)
cape_z = zscore(cape)
m2_z = zscore(m2.pct_change(12) * 100)
vix_z = zscore(vix)

# Compute latest raw values
buffett_latest = latest_float(buffett)
cape_latest = latest_float(cape)
m2_latest = latest_float(m2.pct_change(12) * 100)
vix_latest = latest_float(vix)

# Build dataframe row by row
df = pd.DataFrame(
    {
        "indicator": [
            "Buffett_ratio",
            "CAPE",
            "M2_YoY",
            "VIX",
        ],
        "Z-score": [
            buffett_z,
            cape_z,
            m2_z,
            vix_z,
        ],
        "Latest_value": [
            buffett_latest,
            cape_latest,
            m2_latest,
            vix_latest,
        ],
    }
).set_index("indicator")

# === Composite phase ===
weights = {
    "Buffett_ratio": 0.4,
    "CAPE": 0.4,
    "M2_YoY": -0.1,
    "VIX": -0.1,
}

df["Weight"] = df.index.map(weights.get)
df["Contribution"] = df["Z-score"] * df["Weight"]

comp = df["Contribution"].sum()
if comp < 1.0:
    phase = "Expansion"
elif comp < 2.0:
    phase = "Euphoria"
else:
    phase = "Instability"

# Diagnostic split between valuation "pressure" and liquidity/volatility "triggers"
pressure = 0.5 * (
    df.loc["Buffett_ratio", "Z-score"] + df.loc["CAPE", "Z-score"]
)
trigger = -0.5 * (
    df.loc["M2_YoY", "Z-score"] + df.loc["VIX", "Z-score"]
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 100)
print("\n--- BUBBLY REPORT ---")
print(df.round(2))
print(f"\nComposite phase index: {comp:.2f} → {phase}")
print(
    f"Valuation pressure: {pressure:.2f} | Liquidity/vol trigger: {trigger:.2f}"
)
print("\nDebug check:")
for name in ["Buffett_ratio", "CAPE", "M2_YoY", "VIX"]:
    print(
        f"{name} z={df.loc[name, 'Z-score']:.2f} weight={df.loc[name, 'Weight']:.2f}"
        f" contribution={df.loc[name, 'Contribution']:.2f}"
    )

os.makedirs("output", exist_ok=True)
df.to_html("output/bubbly_report.html")
print("\nReport saved to output/bubbly_report.html")
