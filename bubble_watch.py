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
# from pandas_datareader import data as pdr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot as plotly_plot

from html_report import save_html_report

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
        # try:
        #     df = pdr.get_data_yahoo(symbol, start=start)
        # except Exception as exc:
        raise RuntimeError(f"Failed to fetch data for {symbol}") # from exc

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
    std_raw = exp.std()

    count = exp.count()
    valid = count >= min_periods

    std = std_raw.copy()
    std = std.where(std > 0.5, 0.5)

    z = (series - mean) / std
    z = z.where(valid)
    z = z.clip(-4.0, 4.0)
    return z


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


def monthly_gdp_with_proxy(gdp: pd.Series, proxy: pd.Series) -> pd.Series:
    """Blend quarterly GDP into monthly using a high-frequency activity proxy."""
    gdp = ensure_series(gdp, "GDP").sort_index()
    gdp_q = gdp.copy()
    gdp_q.index = gdp_q.index + pd.offsets.QuarterEnd(0)
    gdp_m = gdp_q.resample("ME").ffill()

    proxy = ensure_series(proxy, "ActivityProxy").sort_index()
    proxy_m = proxy.resample("ME").last().ffill()
    proxy_quarter_avg = proxy_m.groupby(proxy_m.index.to_period("Q")).transform("mean")
    adj = (proxy_m / proxy_quarter_avg).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    adj = adj.reindex(gdp_m.index).ffill().fillna(1.0)

    blended = gdp_m * adj
    blended.name = "GDP"
    return blended


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


def build_validation_tables(
    hist_df: pd.DataFrame,
    composite_column: str = "Composite",
    horizons: tuple[int, ...] = (6, 12),
    threshold: float = -0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate forward drawdowns after composite breaches."""
    records = []
    composite = hist_df[composite_column]
    prices = hist_df["SP500"]

    for idx in range(len(hist_df)):
        if composite.iloc[idx] <= 1.0:
            continue
        start_date = hist_df.index[idx]
        start_price = prices.iloc[idx]
        for horizon in horizons:
            future_prices = prices.iloc[idx : idx + horizon + 1]
            if len(future_prices) <= 1:
                continue
            rel_returns = future_prices / start_price - 1.0
            max_dd = float(rel_returns.min())
            hit = bool(max_dd <= threshold)
            lead_months = float("nan")
            if hit:
                for step, val in enumerate(rel_returns.iloc[1:], start=1):
                    if val <= threshold:
                        lead_months = float(step)
                        break
            records.append(
                {
                    "signal_date": start_date,
                    "horizon_months": horizon,
                    "composite_at_signal": float(composite.iloc[idx]),
                    "max_forward_drawdown": max_dd,
                    "hit_threshold": hit,
                    "threshold_lead_months": lead_months,
                }
            )

    columns = [
        "signal_date",
        "horizon_months",
        "composite_at_signal",
        "max_forward_drawdown",
        "hit_threshold",
        "threshold_lead_months",
    ]
    signals_df = pd.DataFrame(records, columns=columns)

    summary_rows = []
    for horizon in horizons:
        subset = signals_df[signals_df["horizon_months"] == horizon]
        total = len(subset)
        hits = subset["hit_threshold"].sum() if total else 0
        hit_rate = hits / total if total else float("nan")
        avg_hit_lead = (
            subset.loc[subset["hit_threshold"], "threshold_lead_months"].mean()
            if hits
            else float("nan")
        )
        median_dd = subset["max_forward_drawdown"].median() if total else float("nan")
        worst_dd = subset["max_forward_drawdown"].min() if total else float("nan")
        summary_rows.append(
            {
                "horizon_months": horizon,
                "signals": total,
                "hits": int(hits),
                "hit_rate": hit_rate,
                "avg_hit_lead_months": avg_hit_lead,
                "median_drawdown": median_dd,
                "worst_drawdown": worst_dd,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return signals_df, summary_df


def compute_realtime_composite(
    hist_df: pd.DataFrame,
    weights: dict[str, float],
    lag_map: dict[str, int] | None = None,
    min_history: int = 12,
) -> pd.Series:
    """Approximate real-time composite using only data available at each month."""
    if lag_map is None:
        lag_map = {}

    series_map = {
        "Buffett_ratio": "Buffett",
        "CAPE": "CAPE",
        "CI_Loans_YoY": "CI_Loans_YoY",
        "M2_YoY": "M2_YoY",
        "FedBalanceSheet_YoY": "FedBalanceSheet_YoY",
        "RRP_YoY": "RRP_YoY",
        "HY_Spread": "HY_Spread",
        "IG_Spread": "IG_Spread",
        "VIX": "VIX",
        "VolTerm": "VolTerm",
    }

    realtime_values = []
    for idx, date in enumerate(hist_df.index):
        total = 0.0
        valid = True
        for key, col in series_map.items():
            lag = lag_map.get(key, 0)
            series = hist_df[col].shift(lag).iloc[: idx + 1].dropna()
            if len(series) < min_history:
                valid = False
                break
            z = expanding_zscore(series).iloc[-1]
            total += weights[key] * z
        realtime_values.append(total if valid else float("nan"))
    if len(realtime_values) != len(hist_df.index):
        raise RuntimeError(
            f"Real-time composite length mismatch: values={len(realtime_values)} index={len(hist_df.index)}"
        )
    return pd.Series(realtime_values, index=hist_df.index, name="Composite_real_time")


def build_plotly_dashboard(
    hist_df: pd.DataFrame,
    event_dates: list[pd.Timestamp],
    output_path: str | Path,
) -> Path:
    """Create an interactive Plotly dashboard and write it to disk."""
    composite = hist_df["Composite"]
    realtime = hist_df["Composite_real_time"]
    pressure_z = hist_df["Pressure_z"]
    trigger_z = hist_df["Trigger_z"]
    pressure_raw = hist_df["Pressure_raw"]
    trigger_raw = hist_df["Trigger_raw"]
    drawdown = hist_df["SPX_Drawdown"]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.45, 0.30, 0.25],
    )

    # Regime shading on composite panel
    x_start = composite.index.min()
    x_end = composite.index.max()
    regime_shapes = [
        {
            "type": "rect",
            "xref": "x1",
            "yref": "y1",
            "x0": x_start,
            "x1": x_end,
            "y0": -4,
            "y1": 1,
            "fillcolor": "rgba(46, 125, 50, 0.18)",
            "line": {"width": 0},
            "layer": "below",
        },
        {
            "type": "rect",
            "xref": "x1",
            "yref": "y1",
            "x0": x_start,
            "x1": x_end,
            "y0": 1,
            "y1": 2,
            "fillcolor": "rgba(249, 199, 79, 0.18)",
            "line": {"width": 0},
            "layer": "below",
        },
        {
            "type": "rect",
            "xref": "x1",
            "yref": "y1",
            "x0": x_start,
            "x1": x_end,
            "y0": 2,
            "y1": 4,
            "fillcolor": "rgba(239, 35, 60, 0.18)",
            "line": {"width": 0},
            "layer": "below",
        },
    ]

    fig.add_trace(
        go.Scatter(
            x=composite.index,
            y=composite,
            name="Composite (z)",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="Composite (z): %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if realtime.notna().any():
        fig.add_trace(
            go.Scatter(
                x=realtime.index,
                y=realtime,
                name="Composite pseudo real-time",
                line=dict(color="#ff7f0e", width=1.5, dash="dash"),
                hovertemplate="Composite RT (z): %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(go.Scatter(x=[], y=[], showlegend=False), row=1, col=1)

    if event_dates:
        event_points = hist_df.loc[hist_df.index.isin(event_dates)]
        fig.add_trace(
            go.Scatter(
                x=event_points.index,
                y=event_points["Composite"],
                mode="markers",
                name="Signal → drawdown hit",
                marker=dict(color="#ef233c", size=8, symbol="x"),
                hovertemplate="Event: %{x|%Y-%m}<br>Composite: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(go.Scatter(x=[], y=[], showlegend=False), row=1, col=1)

    # Pressure/Trigger traces (z scores)
    fig.add_trace(
        go.Scatter(
            x=pressure_z.index,
            y=pressure_z,
            name="Pressure (z)",
            line=dict(color="#264653", width=2),
            hovertemplate="Pressure (z): %{y:.2f}<br>Raw: %{customdata[0]:.2f}<extra></extra>",
            customdata=np.column_stack([pressure_raw]),
            visible=True,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=trigger_z.index,
            y=trigger_z,
            name="Trigger (z)",
            line=dict(color="#e76f51", width=2),
            hovertemplate="Trigger (z): %{y:.2f}<br>Raw: %{customdata[0]:.2f}<extra></extra>",
            customdata=np.column_stack([trigger_raw]),
            visible=True,
        ),
        row=2,
        col=1,
    )

    # Raw traces (initially hidden)
    fig.add_trace(
        go.Scatter(
            x=pressure_raw.index,
            y=pressure_raw,
            name="Pressure (raw)",
            line=dict(color="#264653", width=2, dash="dot"),
            hovertemplate="Pressure (raw): %{y:.2f}<br>Z-score: %{customdata[0]:.2f}<extra></extra>",
            customdata=np.column_stack([pressure_z]),
            visible=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=trigger_raw.index,
            y=trigger_raw,
            name="Trigger (raw)",
            line=dict(color="#e76f51", width=2, dash="dot"),
            hovertemplate="Trigger (raw): %{y:.2f}<br>Z-score: %{customdata[0]:.2f}<extra></extra>",
            customdata=np.column_stack([trigger_z]),
            visible=False,
        ),
        row=2,
        col=1,
    )

    # Drawdown area
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            name="S&P Drawdown",
            fill="tozeroy",
            line=dict(color="rgba(214, 39, 40, 0.7)", width=1.5),
            fillcolor="rgba(214, 39, 40, 0.3)",
            hovertemplate="Drawdown: %{y:.2%}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    vis_z = [True, True, bool(event_dates), True, True, False, False, True]
    vis_raw = [True, True, bool(event_dates), False, False, True, True, True]

    updatemenus = [
        dict(
            type="buttons",
            direction="right",
            x=0,
            y=1.18,
            xanchor="left",
            buttons=[
                dict(
                    label="Z-score",
                    method="update",
                    args=[
                        {"visible": vis_z},
                        {"yaxis2": {"title": "Pressure / Trigger (z-score)"}},
                    ],
                ),
                dict(
                    label="Raw",
                    method="update",
                    args=[
                        {"visible": vis_raw},
                        {"yaxis2": {"title": "Pressure / Trigger (raw units)"}},
                    ],
                ),
            ],
        )
    ]

    fig.update_layout(
        template="plotly_dark",
        height=900,
        margin=dict(l=70, r=40, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        updatemenus=updatemenus,
        shapes=regime_shapes,
    )

    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Composite (z)", row=1, col=1)
    fig.update_yaxes(title_text="Pressure / Trigger (z-score)", row=2, col=1)
    fig.update_yaxes(
        title_text="S&P 500 Drawdown",
        row=3,
        col=1,
        tickformat=".0%",
        range=[min(drawdown.min() * 1.1, -1.0), 0.05],
    )

    # Threshold lines
    for thresh in [1, 2]:
        fig.add_hline(
            y=thresh,
            line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot"),
            row=1,
            col=1,
        )
    fig.add_hline(
        y=-0.15,
        line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"),
        row=3,
        col=1,
    )

    output_path = Path(output_path)
    plotly_plot(
        fig,
        filename=str(output_path),
        auto_open=False,
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {"format": "png", "filename": "bubbly_dashboard", "scale": 2},
        },
    )
    return output_path


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


def fetch_cape_series(local_path="data/ie_data.xls") -> pd.Series:
    """Load CAPE from FRED when possible, otherwise fall back to local Shiller file."""
    try:
        cape = fetch_fred("CAPE")
        cape = ensure_series(cape, "CAPE")
        cape.index = pd.to_datetime(cape.index)
        cape = cape.sort_index()
        cape = cape[cape.index >= cape.index.max() - pd.DateOffset(years=60)]
        return cape
    except Exception:
        return fetch_shiller_local_xls(local_path)


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
ci_loans_weekly = fetch_fred("TOTCI")
hy_spread = fetch_fred("BAMLH0A0HYM2")
walcl = fetch_fred("WALCL")
rrp = fetch_fred("RRPONTSYAWARD")
ig_spread = fetch_fred("BAA10Y")
indpro = fetch_fred("INDPRO")
wilshire = fetch_yahoo("^W5000")
sp500 = fetch_yahoo("^GSPC")
vix = fetch_yahoo("^VIX")
vxv = fetch_yahoo("^VIX3M")
cape = fetch_cape_series("data/ie_data.xls")

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
gdp_m = monthly_gdp_with_proxy(gdp, indpro)
wilshire_m = wilshire.resample("ME").last()
sp_m = sp500.resample("ME").last()

# Align and compute
buffett_df = pd.concat([wilshire_m, gdp_m], axis=1, join="inner")
buffett_df.columns = ["Wilshire", "GDP"]

buffett = (buffett_df["Wilshire"] / buffett_df["GDP"]) * 100.0
buffett = buffett.dropna()
buffett.name = "Buffett % GDP"

# Track latest source dates
ci_loans_weekly = ensure_series(ci_loans_weekly, "CI_Loans").asfreq("W-WED").ffill()
weekly_index = ci_loans_weekly.index
ci_loans_yoy_weekly = ci_loans_weekly.pct_change(52) * 100
leverage_hist = ci_loans_weekly.resample("ME").last()
leverage_yoy_hist = ensure_series(ci_loans_yoy_weekly, "CI_Loans_YoY").resample("ME").last()

walcl_weekly = ensure_series(walcl, "FedBalanceSheet").asfreq("W-WED").ffill()
walcl_weekly = walcl_weekly.reindex(weekly_index).ffill()
walcl_yoy_weekly = walcl_weekly.pct_change(52) * 100
walcl_hist = ensure_series(walcl_yoy_weekly, "FedBalanceSheet_YoY").resample("ME").last()

rrp_daily = ensure_series(rrp, "RRP").sort_index()
rrp_weekly = rrp_daily.resample("W-FRI").last()
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
    vol_term_daily = pd.concat([vix_series, vxv_series], axis=1, join="inner").rename(
        columns={"VIX": "VIX", "VXV": "VXV"}
    )
    vol_term_daily = (vol_term_daily["VIX"] / vol_term_daily["VXV"]).rename("VolTerm")
    vol_term_daily = vol_term_daily.replace([np.inf, -np.inf], np.nan)
except Exception:
    vol_term_daily = pd.Series(dtype=float, name="VolTerm")

if vol_term_daily.empty:
    vol_term_hist = vix_series.resample("ME").last().apply(lambda _: 1.0)
    vol_term_hist.name = "VolTerm"
else:
    vol_term_hist = ensure_series(vol_term_daily, "VolTerm").resample("ME").mean()

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


# === Historical composites ===
# Align raw indicator series at month-end frequency
buffett_hist = ensure_series(buffett, "Buffett")
common_index = buffett_hist.index

m_cape = ensure_series(cape, "CAPE").resample("ME").last()
cape_hist = m_cape.reindex(common_index).ffill().bfill()
m2_hist = (
    ensure_series(m2_weekly, "M2").resample("ME").last().reindex(common_index).ffill().bfill()
)
m2_yoy_hist = (
    ensure_series(m2_yoy_weekly, "M2_YoY")
    .resample("ME")
    .last()
    .reindex(common_index)
    .ffill()
    .bfill()
    .fillna(0.0)
)
vix_hist = (
    ensure_series(vix, "VIX").resample("ME").last().reindex(common_index).ffill().bfill()
)
sp_hist = (
    ensure_series(sp500, "SP500").resample("ME").last().reindex(common_index).ffill().bfill()
)
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
hist_df = hist_df.dropna(
    subset=[
        "Buffett_z",
        "CAPE_z",
        "M2_YoY_z",
        "VIX_z",
        "CI_Loans_z",
        "HY_Spread_z",
        "FedBalanceSheet_z",
        "RRP_z",
        "IG_Spread_z",
        "VolTerm_z",
    ]
)

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

signals_full, summary_full = build_validation_tables(hist_df)

lag_map = {
    "Buffett_ratio": 2,
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
summary_comparison["median_drawdown_delta"] = (
    summary_comparison["median_drawdown_rt"] - summary_comparison["median_drawdown_full"]
)

# === Latest snapshot ===
buffett_z = float(hist_df["Buffett_z"].iloc[-1])
cape_z = float(hist_df["CAPE_z"].iloc[-1])
m2_z = float(hist_df["M2_YoY_z"].iloc[-1])
vix_z = float(hist_df["VIX_z"].iloc[-1])
leverage_z = float(hist_df["CI_Loans_z"].iloc[-1])
hy_z = float(hist_df["HY_Spread_z"].iloc[-1])
fed_z = float(hist_df["FedBalanceSheet_z"].iloc[-1])
rrp_z = float(hist_df["RRP_z"].iloc[-1])
ig_z = float(hist_df["IG_Spread_z"].iloc[-1])
volterm_z = float(hist_df["VolTerm_z"].iloc[-1])

buffett_latest = float(hist_df["Buffett"].iloc[-1])
cape_latest = float(hist_df["CAPE"].iloc[-1])
m2_latest = float(hist_df["M2_YoY"].iloc[-1])
vix_latest = float(hist_df["VIX"].iloc[-1])
leverage_latest = float(hist_df["CI_Loans_YoY"].iloc[-1])
hy_latest = float(hist_df["HY_Spread"].iloc[-1])
fed_latest = float(hist_df["FedBalanceSheet_YoY"].iloc[-1])
rrp_latest = float(hist_df["RRP_YoY"].iloc[-1])
ig_latest = float(hist_df["IG_Spread"].iloc[-1])
volterm_latest = float(hist_df["VolTerm"].iloc[-1])

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
            "FedBalanceSheet_YoY",
            "RRP_YoY",
            "HY_Spread",
            "IG_Spread",
            "VolTerm",
        ],
        "Z-score": [
            buffett_z,
            cape_z,
            leverage_z,
            m2_z,
            vix_z,
            fed_z,
            rrp_z,
            hy_z,
            ig_z,
            volterm_z,
        ],
        "Latest_value": [
            buffett_latest,
            cape_latest,
            leverage_latest,
            m2_latest,
            vix_latest,
            fed_latest,
            rrp_latest,
            hy_latest,
            ig_latest,
            volterm_latest,
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
            -hist_df["FedBalanceSheet_z"].iloc[-1],
            hist_df["RRP_z"].iloc[-1],
            hist_df["HY_Spread_z"].iloc[-1],
            hist_df["IG_Spread_z"].iloc[-1],
            hist_df["VolTerm_z"].iloc[-1],
            -hist_df["VIX_z"].iloc[-1],
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
for name in [
    "Buffett_ratio",
    "CAPE",
    "CI_Loans_YoY",
    "M2_YoY",
    "FedBalanceSheet_YoY",
    "RRP_YoY",
    "HY_Spread",
    "IG_Spread",
    "VIX",
    "VolTerm",
]:
    print(
        f"{name} z={df.loc[name, 'Z-score']:.2f} weight={df.loc[name, 'Weight']:.2f}"
        f" contribution={df.loc[name, 'Contribution']:.2f}"
    )

if summary_full.empty:
    print("\nNo composite signals above threshold available for validation yet.")
else:
    print("\nForward drawdown validation (Composite > 1, drawdown ≤ -15%):")
    print(summary_full.round(3).to_string(index=False))

if not summary_rt.empty:
    print("\nPseudo real-time validation (lagged inputs, same threshold):")
    print(summary_rt.round(3).to_string(index=False))
    print("\nValidation deltas (real-time minus full-information):")
    print(summary_comparison[["horizon_months", "hit_rate_full", "hit_rate_rt", "hit_rate_delta"]].round(3).to_string(index=False))

alert_thresholds = {
    "Observation": 0.6,
    "Caution": 0.9,
    "Euphoria": 1.2,
    "Instability": 1.5,
}
print("\nAlert counts (full composite):")
for label, threshold in alert_thresholds.items():
    count = int((hist_df["Composite"] >= threshold).sum())
    current = hist_df["Composite"].iloc[-1] >= threshold
    print(f"  {label} (≥{threshold:.2f}): {count} hits | current={'YES' if current else 'no'}")

if not hist_df["Composite_real_time"].dropna().empty:
    print("\nAlert counts (pseudo real-time composite):")
    for label, threshold in alert_thresholds.items():
        series = hist_df["Composite_real_time"].dropna()
        count = int((series >= threshold).sum())
        current = series.iloc[-1] >= threshold if not series.empty else False
        print(f"  {label} (≥{threshold:.2f}): {count} hits | current={'YES' if current else 'no'}")

# Persist historical backtest
history_cols = [
    "Buffett",
    "CAPE",
    "M2_YoY",
    "VIX",
    "CI_Loans_YoY",
    "FedBalanceSheet_YoY",
    "RRP_YoY",
    "HY_Spread",
    "IG_Spread",
    "VolTerm",
    "Pressure_raw",
    "Trigger_raw",
    "Buffett_z",
    "CAPE_z",
    "M2_YoY_z",
    "VIX_z",
    "CI_Loans_z",
    "FedBalanceSheet_z",
    "RRP_z",
    "HY_Spread_z",
    "IG_Spread_z",
    "VolTerm_z",
    "Pressure_z",
    "Trigger_z",
    "Composite",
    "Composite_real_time",
    "SP500",
    "SPX_Drawdown",
]
os.makedirs("output", exist_ok=True)
hist_df[history_cols].to_csv("output/bubbly_history.csv", float_format="%.4f")
signals_full.to_csv("output/bubbly_validation_signals_full.csv", index=False)
summary_full.to_csv("output/bubbly_validation_summary.csv", index=False)
signals_rt.to_csv("output/bubbly_validation_signals_realtime.csv", index=False)
summary_rt.to_csv("output/bubbly_validation_summary_realtime.csv", index=False)
summary_comparison.to_csv("output/bubbly_validation_summary_comparison.csv", index=False)

event_dates = list(
    pd.to_datetime(
        signals_full.loc[signals_full["hit_threshold"], "signal_date"]
    ).dropna().unique()
)
event_dates = [d for d in event_dates if d in hist_df.index]
event_points = hist_df.loc[event_dates] if event_dates else pd.DataFrame()

plt.rcParams.update(
    {
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "font.family": "DejaVu Sans",
    }
)

fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(3, 1, height_ratios=[3, 2, 1.8], hspace=0.05)
ax_top = fig.add_subplot(gs[0])
ax_mid = fig.add_subplot(gs[1], sharex=ax_top)
ax_bot = fig.add_subplot(gs[2], sharex=ax_top)

regime_bands = [
    (-4, 1, "#2d6a4f"),
    (1, 2, "#f9c74f"),
    (2, 4, "#ef233c"),
]
for y0, y1, color in regime_bands:
    ax_top.axhspan(y0, y1, color=color, alpha=0.12, lw=0)

ax_top.plot(
    hist_df.index,
    hist_df["Composite"],
    label="Composite (z)",
    color="#1f77b4",
    linewidth=1.8,
)
if hist_df["Composite_real_time"].notna().any():
    ax_top.plot(
        hist_df.index,
        hist_df["Composite_real_time"],
        label="Composite pseudo real-time",
        color="#ff7f0e",
        linewidth=1.4,
        linestyle="--",
    )

if not event_points.empty:
    ax_top.scatter(
        event_points.index,
        event_points["Composite"],
        color="#ef233c",
        marker="o",
        edgecolors="white",
        linewidths=0.4,
        s=36,
        label="Signal → drawdown hit",
        zorder=5,
    )

ax_top.axhline(0, color="white", alpha=0.3, linewidth=0.8)
for thresh, style in [(1, "--"), (2, ":")]:
    ax_top.axhline(thresh, color="white", alpha=0.4, linewidth=0.8, linestyle=style)

ax_top.set_ylabel("Composite (z)")
ax_top.set_title("Bubbly Composite Index")
ax_top.legend(loc="upper left", frameon=False)
ax_top.grid(alpha=0.15, linestyle="--")

# Middle panel: pressure vs trigger
ax_mid.plot(
    hist_df.index,
    hist_df["Pressure_z"],
    label="Valuation pressure (z)",
    color="#264653",
    linewidth=1.6,
)
ax_mid.plot(
    hist_df.index,
    hist_df["Trigger_z"],
    label="Liquidity / Vol trigger (z)",
    color="#e76f51",
    linewidth=1.6,
)
ax_mid.axhline(0, color="white", alpha=0.3, linewidth=0.8)
ax_mid.set_ylabel("Pressure & Trigger (z)")
ax_mid.legend(loc="upper left", frameon=False)
ax_mid.grid(alpha=0.15, linestyle="--")

# Bottom panel: drawdown
ax_bot.fill_between(
    hist_df.index,
    hist_df["SPX_Drawdown"],
    0,
    color="#d62828",
    alpha=0.3,
    label="S&P 500 drawdown",
)
ax_bot.axhline(-0.15, color="white", alpha=0.4, linewidth=0.8, linestyle="--", label="-15% threshold")
ax_bot.set_ylabel("Drawdown")
ax_bot.set_xlabel("Date")
ax_bot.set_ylim(min(hist_df["SPX_Drawdown"].min() * 1.1, -1.0), 0.05)
ax_bot.legend(loc="lower left", frameon=False)
ax_bot.grid(alpha=0.15, linestyle="--")

for ax in (ax_top, ax_mid):
    ax.tick_params(labelbottom=False)

fig.autofmt_xdate()
fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.08, hspace=0.05)
plt.savefig("output/bubbly_backtest.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Real-time comparison quick chart
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(hist_df.index, hist_df["Composite"], label="Composite (z)", color="#1f77b4")
ax.plot(
    hist_df.index,
    hist_df["Composite_real_time"],
    label="Composite pseudo real-time",
    color="#ff7f0e",
    linestyle="--",
)
ax.axhline(1, color="white", linestyle="--", alpha=0.3, linewidth=0.8)
ax.axhline(2, color="white", linestyle=":", alpha=0.3, linewidth=0.8)
ax.set_ylabel("Composite (z)")
ax.set_xlabel("Date")
ax.set_title("Composite (full vs. pseudo real-time)")
ax.legend(loc="upper left", frameon=False)
ax.grid(alpha=0.15, linestyle="--")

ax2 = ax.twinx()
ax2.fill_between(
    hist_df.index,
    hist_df["SPX_Drawdown"],
    0,
    color="#d62828",
    alpha=0.2,
)
ax2.set_ylabel("S&P 500 drawdown")
ax2.set_ylim(min(hist_df["SPX_Drawdown"].min() * 1.1, -1.0), 0.05)
fig.tight_layout()
plt.savefig("output/bubbly_realtime_backtest.png", dpi=300, bbox_inches="tight")
plt.close(fig)

dashboard_path = build_plotly_dashboard(
    hist_df,
    event_dates=event_dates,
    output_path=Path("output") / "bubbly_dashboard.html",
)

print("Historical series saved to output/bubbly_history.csv")
print("Backtest chart saved to output/bubbly_backtest.png")
print("Real-time comparison chart saved to output/bubbly_realtime_backtest.png")
print(f"Interactive dashboard saved to {dashboard_path}")

report_path = save_html_report(
    overview_df=df,
    validation_summary=summary_full,
    source_dates=source_dates,
    composite_value=comp,
    phase=phase,
    pressure=pressure,
    trigger=trigger,
    data_coverage=composite_last_date,
    output_dir="output",
    history_chart_filename="bubbly_backtest.png",
)

print(f"\nReport saved to {report_path}")
