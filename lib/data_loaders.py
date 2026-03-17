import os
import urllib.request
import urllib.parse
import json
import socket
from pathlib import Path
from io import StringIO
import pandas as pd
import yfinance as yf
import numpy as np

# Set global timeout for all network operations
socket.setdefaulttimeout(30)

FRED_KEY = os.environ.get("FRED_API_KEY", "")
QUANDL_KEY = os.environ.get("QUANDL_API_KEY", "")

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

# Load keys immediately on import or let main call it? 
# Better to let main call it, but defaults above rely on env vars. 
# We'll stick to the original logic where it was called at module level or init.
# But for a library, we shouldn't side-effect too much. 
# I will expose load_api_keys and call it in main.
# However, FRED_KEY is module-level constant. 
# I will update FRED_KEY usage to query os.environ inside functions or update these constants.

def get_fred_key():
    return os.environ.get("FRED_API_KEY", "")

def get_quandl_key():
    return os.environ.get("QUANDL_API_KEY", "")

def fetch_fred(series_id: str, freq=None) -> pd.Series:
    """Fetch FRED series and return as pandas Series"""
    params = {
        "series_id": series_id,
        "file_type": "json",
    }
    fred_key = get_fred_key()
    if fred_key:
        params["api_key"] = fred_key
    if freq:
        params["frequency"] = freq
    url = "https://api.stlouisfed.org/fred/series/observations?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"Error fetching FRED series {series_id} via API: {e}")
        csv_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?" + urllib.parse.urlencode({"id": series_id})
        try:
            with urllib.request.urlopen(csv_url, timeout=30) as resp:
                csv_text = resp.read().decode("utf-8")
            df = pd.read_csv(StringIO(csv_text))
            date_col = next((c for c in ("DATE", "observation_date") if c in df.columns), None)
            if date_col is None or series_id not in df.columns:
                raise RuntimeError(f"Unexpected FRED CSV shape for {series_id}")
            s = pd.Series(df[series_id].values, index=pd.to_datetime(df[date_col]), name=series_id)
            s = pd.to_numeric(s, errors="coerce").dropna()
            return s.sort_index()
        except Exception as csv_exc:
            print(f"Error fetching FRED series {series_id} via CSV fallback: {csv_exc}")
            return pd.Series(dtype=float)

    vals, dates = [], []
    for o in data.get("observations", []):
        try:
            vals.append(float(o["value"]))
            dates.append(pd.to_datetime(o["date"]))
        except ValueError:
            continue
    return pd.Series(vals, index=dates).dropna()


def fetch_yahoo(symbol, start="1980-01-01") -> pd.Series:
    """Fetch a single Yahoo Finance series and return a clean 1D Series."""
    try:
        df = yf.download(
            symbol,
            start=start,
            progress=False,
            auto_adjust=False,
            group_by="column",
            threads=False,
        )
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        raise RuntimeError(f"Failed to fetch data for {symbol}")

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
    quandl_key = get_quandl_key()
    url = f"https://data.nasdaq.com/api/v3/datasets/{code}.json?api_key={quandl_key}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"Error fetching Quandl series {code}: {e}")
        return pd.Series(dtype=float)
    df = pd.DataFrame(data["dataset"]["data"], columns=data["dataset"]["column_names"])
    s = pd.Series(df["Value"].astype(float).values, index=pd.to_datetime(df["Date"]))
    s.name = code
    return s.sort_index()

def fetch_shiller_local_xls(path="data/ie_data.xls") -> pd.Series:
    """
    Read Shiller CAPE (cyclically adjusted P/E, column 'CAPE') from local Excel file
    downloaded from shillerdata.com
    """
    # Read the sheet and normalize column names
    try:
        df = pd.read_excel(path, sheet_name="Data", skiprows=7)
    except Exception as e:
         print(f"Error reading local Shiller file: {e}")
         return pd.Series(dtype=float, name="CAPE")

    df.columns = [str(c).strip() for c in df.columns]

    # Basic column sanity
    if "Date" not in df.columns or "CAPE" not in df.columns:
         # fail silently or log? Original raised RuntimeError.
         # We will raise to match original behavior if vital.
         raise RuntimeError("No 'Date' or 'CAPE' column found in Shiller file")

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
    # Original code had this cut off.
    cutoff = s.index.max() - pd.DateOffset(years=40)
    s = s[s.index >= cutoff]

    return s

def ensure_series(obj, name: str) -> pd.Series:
    """Ensure we are working with a single pandas Series."""
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise ValueError(f"{name} expected 1 column, got {obj.shape[1]}")
        obj = obj.iloc[:, 0]
    obj = obj.copy()
    obj.name = name
    return obj

def fetch_cape_series(local_path="data/ie_data.xls") -> pd.Series:
    """Load CAPE from FRED when possible, otherwise fall back to local Shiller file."""
    try:
        cape = fetch_fred("CAPE")
        if cape.empty:
            raise ValueError("FRED CAPE series is empty")
        cape = ensure_series(cape, "CAPE")
        cape.index = pd.to_datetime(cape.index)
        cape = cape.sort_index()
        cape = cape[cape.index >= cape.index.max() - pd.DateOffset(years=60)]
        return cape
    except Exception:
        return fetch_shiller_local_xls(local_path)

def load_m2_cache(cache_path: str | Path = "data/m2_manual.csv") -> pd.Series | None:
    """Load cached/manual M2 weekly levels when FRED is unavailable."""
    path = Path(cache_path)
    if not path.is_file():
        return None

    df = pd.read_csv(path)
    required = {"date", "M2"}
    if not required.issubset(set(df.columns)):
        raise RuntimeError(
            f"M2 cache at {path} must contain columns {sorted(required)}"
        )

    s = pd.Series(df["M2"].astype(float).values, index=pd.to_datetime(df["date"]), name="M2")
    return s.sort_index().dropna()

def save_m2_cache(m2_weekly: pd.Series, cache_path: str | Path = "data/m2_manual.csv") -> Path:
    """Persist M2 weekly levels so runs can continue through temporary FRED outages."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = ensure_series(m2_weekly, "M2").dropna().sort_index().to_frame(name="M2")
    out.index.name = "date"
    out.reset_index().to_csv(path, index=False)
    return path

def compute_weekly_m2_yoy(cache_path: str | Path = "data/m2_manual.csv") -> tuple[pd.Series, pd.Series]:
    """Return weekly M2 levels and YoY % derived from WM2NS with automatic local fallback."""
    try:
        m2_weekly = ensure_series(fetch_fred("WM2NS"), "M2")
        save_m2_cache(m2_weekly, cache_path)
        print(f"M2 source: FRED WM2NS (cache refreshed at {cache_path})")
    except Exception as exc:
        cached = load_m2_cache(cache_path)
        if cached is None:
            raise RuntimeError(
                "Unable to fetch WM2NS from FRED and no local M2 cache found at "
                f"{cache_path}."
            ) from exc
        m2_weekly = cached
        print(f"M2 source: local cache fallback at {cache_path} ({type(exc).__name__})")

    m2_weekly = m2_weekly.asfreq("W-MON")
    m2_weekly = m2_weekly.ffill()
    m2_yoy_weekly = m2_weekly.pct_change(52) * 100
    return m2_weekly, m2_yoy_weekly
