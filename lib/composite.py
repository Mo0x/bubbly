import pandas as pd
import numpy as np
from .indicators import expanding_zscore

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
