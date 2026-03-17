import pandas as pd
import numpy as np

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


def extend_monthly_denominator_with_proxy(
    base_quarterly: pd.Series,
    monthly_proxy: pd.Series,
    proxy_name: str = "FastProxy",
) -> pd.Series:
    """Extend a quarterly denominator with a monthly proxy rebased to the same level.

    The proxy is only used after the last available quarter, which preserves the
    original historical construction while allowing fresher monthly updates.
    """
    base_q = ensure_series(base_quarterly, "BaseDenominator").sort_index().dropna()
    proxy_m = ensure_series(monthly_proxy, proxy_name).sort_index().dropna()

    if base_q.empty:
        return proxy_m.resample("ME").last().ffill()
    if proxy_m.empty:
        base_m = base_q.copy()
        base_m.index = base_m.index + pd.offsets.QuarterEnd(0)
        return base_m.resample("ME").ffill()

    base_q.index = base_q.index + pd.offsets.QuarterEnd(0)
    base_m = base_q.resample("ME").ffill()
    proxy_m = proxy_m.resample("ME").last().ffill()

    proxy_q = proxy_m.resample("QE").last()
    overlap = pd.concat([base_q.rename("base"), proxy_q.rename("proxy")], axis=1, join="inner").dropna()

    scale = 1.0
    if not overlap.empty:
        ratios = overlap["base"] / overlap["proxy"]
        ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()
        if not ratios.empty:
            scale = float(ratios.median())

    proxy_scaled = (proxy_m * scale).rename(base_q.name)
    extension = proxy_scaled[proxy_scaled.index > base_m.index.max()]
    extended = pd.concat([base_m, extension]).sort_index()
    extended.name = base_q.name
    return extended
