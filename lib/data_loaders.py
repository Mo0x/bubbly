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
SHILLER_XLS_URL = (
    "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/"
    "downloads/3228b83a-7bad-4e69-b405-71e3a1ca6351/ie_data.xls?ver=1772737019751"
)

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

def fetch_fred(series_id: str, freq=None, quiet: bool = False) -> pd.Series:
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
        if not quiet:
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
            if not quiet:
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
    df = fetch_shiller_table(path)
    s = pd.Series(df["CAPE"].astype(float).values, index=df.index)
    s.name = "CAPE"
    return s


def fetch_shiller_table(path="data/ie_data.xls") -> pd.DataFrame:
    """
    Read the Shiller workbook into a normalized monthly table.

    The workbook publishes monthly average price (`P`), CPI, and CAPE.
    We preserve those columns so CAPE can be extended beyond the last
    workbook month using live S&P 500 data.
    """
    # Read the sheet and normalize column names
    try:
        df = pd.read_excel(path, sheet_name="Data", skiprows=7)
    except Exception as e:
         print(f"Error reading local Shiller file: {e}")
         return pd.DataFrame(columns=["P", "CPI", "CAPE"])

    df.columns = [str(c).strip() for c in df.columns]

    # Basic column sanity
    required_cols = {"Date", "P", "CPI", "CAPE"}
    if not required_cols.issubset(set(df.columns)):
         # fail silently or log? Original raised RuntimeError.
         # We will raise to match original behavior if vital.
         raise RuntimeError(f"Missing required columns in Shiller file: {sorted(required_cols)}")

    # Drop fully empty rows first (this avoids NaN crashing conversion)
    df = df.dropna(subset=["Date", "P", "CPI", "CAPE"], how="any")

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
    df = df.dropna(subset=["ts"])
    table = (
        df[["ts", "P", "CPI", "CAPE"]]
        .assign(
            P=lambda x: pd.to_numeric(x["P"], errors="coerce"),
            CPI=lambda x: pd.to_numeric(x["CPI"], errors="coerce"),
            CAPE=lambda x: pd.to_numeric(x["CAPE"], errors="coerce"),
        )
        .dropna(subset=["P", "CPI", "CAPE"])
        .set_index("ts")
        .sort_index()
    )

    cutoff = table.index.max() - pd.DateOffset(years=40)
    return table[table.index >= cutoff]


def refresh_shiller_workbook(
    local_path: str | Path = "data/ie_data.xls",
    source_url: str = SHILLER_XLS_URL,
) -> Path:
    """Refresh the local Shiller workbook from the official public download URL."""
    path = Path(local_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(source_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = resp.read()
    path.write_bytes(payload)
    return path

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
    """Refresh and load CAPE from the official Shiller workbook when possible."""
    path = Path(local_path)
    try:
        refreshed = refresh_shiller_workbook(local_path)
        print(f"CAPE source: official Shiller workbook refreshed at {refreshed}")
        return fetch_shiller_local_xls(refreshed)
    except Exception as exc:
        if path.is_file():
            print(f"CAPE source: local workbook fallback at {path} ({type(exc).__name__})")
            return fetch_shiller_local_xls(path)

    try:
        cape = fetch_fred("CAPE", quiet=True)
        if cape.empty:
            raise ValueError("FRED CAPE series is empty")
        cape = ensure_series(cape, "CAPE")
        cape.index = pd.to_datetime(cape.index)
        cape = cape.sort_index()
        cape = cape[cape.index >= cape.index.max() - pd.DateOffset(years=60)]
        return cape
    except Exception:
        return fetch_shiller_local_xls(local_path)


def extend_cape_with_market(
    local_path: str | Path,
    sp500_daily: pd.Series,
    cpi_monthly: pd.Series | None = None,
) -> tuple[pd.Series, float, pd.Timestamp]:
    """
    Extend CAPE beyond the last Shiller workbook month.

    For completed months we use the monthly average S&P 500 level to match
    the workbook's `P` column. For the current incomplete month we use the
    latest daily close as a live nowcast. CPI is used when available so the
    approximation stays in real terms.
    """
    shiller = fetch_shiller_table(local_path)
    if shiller.empty:
        cape = fetch_cape_series(local_path)
        latest_date = pd.Timestamp(sp500_daily.dropna().index.max())
        latest_cape = float(cape.dropna().iloc[-1])
        return cape, latest_cape, latest_date

    sp500_daily = ensure_series(sp500_daily, "SP500").dropna().sort_index()
    if sp500_daily.empty:
        cape = pd.Series(shiller["CAPE"].astype(float).values, index=shiller.index, name="CAPE")
        latest_date = pd.Timestamp(shiller.index.max())
        latest_cape = float(cape.dropna().iloc[-1])
        return cape, latest_cape, latest_date

    anchor = shiller.iloc[-1]
    anchor_date = pd.Timestamp(shiller.index.max())
    anchor_price = float(anchor["P"])
    anchor_cpi = float(anchor["CPI"])
    anchor_cape = float(anchor["CAPE"])

    if cpi_monthly is None:
        cpi_lookup = pd.Series(dtype=float)
    else:
        cpi_lookup = ensure_series(cpi_monthly, "CPI").dropna().sort_index()
        cpi_lookup.index = cpi_lookup.index.to_period("M").to_timestamp(how="start")

    def cape_from_price(price: float, month_start: pd.Timestamp) -> float:
        cpi_target = anchor_cpi
        if not cpi_lookup.empty:
            cpi_target = float(cpi_lookup.reindex([month_start], method="ffill").iloc[0])
        return anchor_cape * (price / anchor_price) * (anchor_cpi / cpi_target)

    cape = pd.Series(shiller["CAPE"].astype(float).values, index=shiller.index, name="CAPE")

    latest_market_date = pd.Timestamp(sp500_daily.index.max())
    current_month_start = latest_market_date.to_period("M").to_timestamp(how="start")
    next_month_start = anchor_date + pd.offsets.MonthBegin(1)

    if next_month_start <= current_month_start:
        for month_start in pd.date_range(next_month_start, current_month_start, freq="MS"):
            month_end = month_start + pd.offsets.MonthEnd(0)
            month_prices = sp500_daily[(sp500_daily.index >= month_start) & (sp500_daily.index <= month_end)]
            if month_prices.empty:
                continue
            if month_start < current_month_start:
                price = float(month_prices.mean())
            else:
                price = float(month_prices.iloc[-1])
            cape.loc[month_start] = cape_from_price(price, month_start)

    current_cape = float(cape_from_price(float(sp500_daily.iloc[-1]), current_month_start))
    return cape.sort_index(), current_cape, latest_market_date

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
    s = s.sort_index().dropna()
    if s.empty:
        return None
    return s

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
        if m2_weekly.dropna().empty:
            raise RuntimeError("FRED WM2NS returned no observations")
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
