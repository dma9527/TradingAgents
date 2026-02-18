"""
Advanced Technical Indicators — Python translations of Pine Script algorithms.

ADD1: FSVZO (Fourier-Smoothed Volume Zone Oscillator)
  - Volume Zone Oscillator with Fourier smoothing + ADF trend filter
  - Flow momentum, divergence detection, extreme zone signals

ADD2: IDEAL BB with Hull Trend + Kahlman Filter
  - 3rd Generation Moving Average (NMA)
  - Hull Moving Average with Kahlman filter
  - Bollinger Bands, VWAP

All functions operate on pandas DataFrames with columns:
  open, high, low, close, volume (lowercase)
"""

import numpy as np
import pandas as pd
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average matching Pine Script's ta.ema."""
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=window, min_periods=1).mean()


def _wma(series: pd.Series, window: int) -> pd.Series:
    """Weighted moving average matching Pine Script's wma."""
    weights = np.arange(1, window + 1, dtype=float)
    return series.rolling(window=window, min_periods=1).apply(
        lambda x: np.dot(x[-len(weights):], weights[-len(x):]) / weights[-len(x):].sum(),
        raw=True,
    )


def _stdev(series: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation."""
    return series.rolling(window=window, min_periods=1).std()


def _vwma(series: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """Volume-weighted moving average."""
    return (series * volume).rolling(window).sum() / volume.rolling(window).sum()


# ──────────────────────────────────────────────────────────────────────────────
# ADD1: FSVZO — Fourier-Smoothed Volume Zone Oscillator
# ──────────────────────────────────────────────────────────────────────────────

def _fourier_smooth(series: pd.Series, length: int) -> pd.Series:
    """Fourier-inspired exponential decay smoothing (Pine: fourier_smooth)."""
    result = pd.Series(np.nan, index=series.index, dtype=float)
    values = series.values
    for i in range(len(values)):
        total = 0.0
        weight_sum = 0.0
        for j in range(min(length, i + 1)):
            w = np.exp(-j / (length * 0.3))
            val = values[i - j]
            if np.isnan(val):
                continue
            total += val * w
            weight_sum += w
        result.iloc[i] = total / weight_sum if weight_sum > 0 else np.nan
    return result


def _adf_trend_filter(close: pd.Series, window: int) -> pd.Series:
    """Simplified ADF trend filter (Pine: adf_trend_filter).
    Returns a multiplier around 1.0 that adjusts for trend strength."""
    if window <= 0:
        return pd.Series(1.0, index=close.index)
    short_w = max(1, window // 3)
    sma_short = _sma(close, short_w)
    sma_long = _sma(close, window)
    volatility = _stdev(close, window)
    trend_strength = np.where(
        volatility > 0, (sma_short - sma_long) / volatility, 0.0
    )
    adjustment = np.clip(trend_strength * 0.2, -0.1, 0.1)
    return pd.Series(1.0 + adjustment, index=close.index)


def fsvzo(
    df: pd.DataFrame,
    length: int = 9,
    signal_length: int = 2,
    smoothing_length: int = 2,
    fourier_length: int = 31,
    adf_window: int = 50,
) -> pd.DataFrame:
    """Calculate the Fourier-Smoothed Volume Zone Oscillator.

    Returns DataFrame with columns:
      fsvzo, fsvzo_signal, flow_momentum
    """
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    # ADF trend filter
    trend = _adf_trend_filter(close, adf_window) if adf_window > 10 else pd.Series(1.0, index=df.index)

    # Relative volume
    rel_volume = volume / _sma(volume, length)
    smoothed_vol = _ema(rel_volume, smoothing_length)

    # Price change
    price_change = close.diff()
    smoothed_change = _ema(price_change, smoothing_length)

    # Momentum with trend weighting (70% base + 30% trend-adjusted)
    base_momentum = _ema(smoothed_change * smoothed_vol, smoothing_length)
    trend_momentum = _ema(smoothed_change * smoothed_vol * trend, smoothing_length)
    momentum = base_momentum * 0.7 + trend_momentum * 0.3

    # Positive / negative momentum
    pos_mom = _ema(momentum.clip(lower=0), length)
    neg_mom = _ema(momentum.clip(upper=0).abs(), length)

    # Ratio → VZO raw
    ratio = np.where(
        neg_mom > 1e-5,
        pos_mom / neg_mom,
        np.where(pos_mom > 1e-5, 100.0, 1.0),
    )
    vzo_raw = pd.Series(100.0 * (ratio - 1.0) / (ratio + 1.0), index=df.index)

    # Final VZO: blend Fourier smooth + EMA
    if fourier_length >= 5:
        fourier_component = _fourier_smooth(vzo_raw, fourier_length)
        ema_component = _ema(vzo_raw, smoothing_length)
        final_vzo = ema_component * 0.6 + fourier_component * 0.4
    else:
        final_vzo = _ema(vzo_raw, smoothing_length)

    final_vzo = final_vzo.clip(-100, 100)
    signal = _sma(final_vzo, signal_length)

    # Flow momentum
    flow_mom = (final_vzo - _ema(final_vzo, 14)) * 0.5

    result = pd.DataFrame(index=df.index)
    result["fsvzo"] = final_vzo
    result["fsvzo_signal"] = signal
    result["flow_momentum"] = flow_mom
    return result


def fsvzo_signals(df: pd.DataFrame, vzo_df: pd.DataFrame) -> pd.DataFrame:
    """Generate buy/sell signals and divergence from FSVZO data.

    Returns DataFrame with columns:
      bull_signal, bear_signal, extreme_bull, extreme_bear,
      bull_divergence, bear_divergence
    """
    vzo = vzo_df["fsvzo"]
    signal = vzo_df["fsvzo_signal"]
    close = df["close"].astype(float)
    low = df["low"].astype(float)
    high = df["high"].astype(float)

    result = pd.DataFrame(index=df.index)

    # Crossover signals
    result["bull_signal"] = (vzo > vzo.shift(1)) & (vzo.shift(1) <= vzo.shift(2))
    result["bear_signal"] = (vzo < vzo.shift(1)) & (vzo.shift(1) >= vzo.shift(2))

    # Extreme zone signals
    result["extreme_bull"] = result["bull_signal"] & (vzo < -90)
    result["extreme_bear"] = result["bear_signal"] & (vzo > 90)

    # Overbought / oversold
    result["overbought"] = vzo > 80
    result["oversold"] = vzo < -80

    # Simple divergence detection (price makes new low but VZO doesn't)
    lookback = 20
    price_new_low = low == low.rolling(lookback).min()
    vzo_not_new_low = vzo > vzo.rolling(lookback).min()
    result["bull_divergence"] = price_new_low & vzo_not_new_low & (vzo < 0)

    price_new_high = high == high.rolling(lookback).max()
    vzo_not_new_high = vzo < vzo.rolling(lookback).max()
    result["bear_divergence"] = price_new_high & vzo_not_new_high & (vzo > 0)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# ADD2: Hull Trend + Kahlman Filter, NMA, Bollinger Bands, VWAP
# ──────────────────────────────────────────────────────────────────────────────

def _hma(series: pd.Series, length: int) -> pd.Series:
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    half = max(1, length // 2)
    sqrt_len = max(1, int(np.sqrt(length)))
    return _wma(2 * _wma(series, half) - _wma(series, length), sqrt_len)


def _hma3(series: pd.Series, length: int) -> pd.Series:
    """Triple HMA variant: WMA(3*WMA(p/3) - WMA(p/2) - WMA(p), p) where p=length/2."""
    p = max(2, length // 2)
    p3 = max(1, p // 3)
    p2 = max(1, p // 2)
    return _wma(3 * _wma(series, p3) - _wma(series, p2) - _wma(series, p), p)


def _kahlman_filter(series: pd.Series, gain: int = 10000) -> pd.Series:
    """Kahlman filter as implemented in Pine Script."""
    g = gain / 10000
    kf = np.zeros(len(series))
    velo = np.zeros(len(series))
    values = series.values

    for i in range(len(values)):
        if np.isnan(values[i]):
            kf[i] = kf[i - 1] if i > 0 else 0
            velo[i] = velo[i - 1] if i > 0 else 0
            continue
        prev_kf = kf[i - 1] if i > 0 else values[i]
        prev_velo = velo[i - 1] if i > 0 else 0
        dk = values[i] - prev_kf
        smooth = prev_kf + dk * np.sqrt(g * 2)
        velo[i] = prev_velo + g * dk
        kf[i] = smooth + velo[i]

    return pd.Series(kf, index=series.index)


def hull_kahlman_trend(
    df: pd.DataFrame,
    length: int = 24,
    gain: int = 10000,
    use_kahlman: bool = True,
    src_col: str = "hl2",
) -> pd.DataFrame:
    """Hull Moving Average with Kahlman filter trend system.

    Returns DataFrame with columns:
      hull_a (long HMA), hull_b (short HMA3),
      hull_trend (1=bullish, -1=bearish),
      hull_buy, hull_sell (crossover signals)
    """
    if src_col == "hl2":
        src = (df["high"] + df["low"]) / 2
    else:
        src = df["close"]

    src = src.astype(float)

    a = _kahlman_filter(_hma(src, length), gain) if use_kahlman else _hma(src, length)
    b = _kahlman_filter(_hma3(src, length), gain) if use_kahlman else _hma3(src, length)

    result = pd.DataFrame(index=df.index)
    result["hull_a"] = a
    result["hull_b"] = b
    result["hull_trend"] = np.where(b > a, 1, -1)
    result["hull_buy"] = (b > a) & (b.shift(1) <= a.shift(1))
    result["hull_sell"] = (a > b) & (a.shift(1) <= b.shift(1))
    return result


def nma_3gen(
    df: pd.DataFrame,
    length1: int = 120,
    length2: int = 12,
    ma_type: str = "EMA",
    src_col: str = "hl2",
) -> pd.Series:
    """3rd Generation Moving Average (NMA).
    Reduces lag by combining two MA passes with alpha correction."""
    if src_col == "hl2":
        src = (df["high"] + df["low"]) / 2
    else:
        src = df["close"]
    src = src.astype(float)

    def _get_ma(s: pd.Series, length: int) -> pd.Series:
        if ma_type == "EMA":
            return _ema(s, length)
        elif ma_type == "SMA":
            return _sma(s, length)
        elif ma_type == "WMA":
            return _wma(s, length)
        elif ma_type == "VWMA":
            return _vwma(s, df["volume"].astype(float), length)
        return _ema(s, length)

    lam = length1 / length2
    alpha = lam * (length1 - 1) / (length1 - lam)
    ma1 = _get_ma(src, length1)
    ma2 = _get_ma(ma1, length2)
    return (1 + alpha) * ma1 - alpha * ma2


def bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    dev_multiple: float = 2.0,
    src_col: str = "close",
) -> pd.DataFrame:
    """Standard Bollinger Bands.

    Returns DataFrame with columns:
      bb_middle, bb_upper, bb_lower, bb_width, bb_pct_b
    """
    src = df[src_col].astype(float)
    middle = _sma(src, period)
    std = _stdev(src, period)
    upper = middle + dev_multiple * std
    lower = middle - dev_multiple * std

    result = pd.DataFrame(index=df.index)
    result["bb_middle"] = middle
    result["bb_upper"] = upper
    result["bb_lower"] = lower
    result["bb_width"] = (upper - lower) / middle
    result["bb_pct_b"] = (src - lower) / (upper - lower)
    return result


def vwap_line(df: pd.DataFrame) -> pd.Series:
    """Session VWAP using HLC3 as typical price."""
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_vp = (hlc3 * df["volume"]).cumsum()
    return cum_vp / cum_vol


# ──────────────────────────────────────────────────────────────────────────────
# UNIFIED ANALYSIS: Combine all indicators into a single report
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all advanced indicators on an OHLCV DataFrame.

    Returns the original DataFrame augmented with all indicator columns.
    """
    out = df.copy()

    # FSVZO
    vzo = fsvzo(df)
    for col in vzo.columns:
        out[col] = vzo[col]

    # FSVZO signals
    sigs = fsvzo_signals(df, vzo)
    for col in sigs.columns:
        out[col] = sigs[col]

    # Hull + Kahlman trend
    hull = hull_kahlman_trend(df)
    for col in hull.columns:
        out[col] = hull[col]

    # NMA 3rd gen
    out["nma"] = nma_3gen(df)

    # Bollinger Bands
    bb = bollinger_bands(df)
    for col in bb.columns:
        out[col] = bb[col]

    # VWAP
    out["vwap"] = vwap_line(df)

    return out


def generate_advanced_report(df: pd.DataFrame, ticker: str) -> str:
    """Generate a text report of advanced indicator readings for the latest bar.

    This is the primary output consumed by the Market Analyst agent.
    """
    data = compute_all_advanced(df)
    latest = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else latest

    lines = [
        f"## Advanced Technical Analysis: {ticker}",
        f"Date: {df.index[-1] if isinstance(df.index[-1], str) else str(df.index[-1])[:10]}",
        f"Close: {latest['close']:.2f}",
        "",
        "### FSVZO (Fourier-Smoothed Volume Zone Oscillator)",
        f"  VZO Value: {latest['fsvzo']:.2f}  (prev: {prev['fsvzo']:.2f})",
        f"  Signal Line: {latest['fsvzo_signal']:.2f}",
        f"  Flow Momentum: {latest['flow_momentum']:.2f}",
        f"  Zone: {'OVERBOUGHT (>80)' if latest['fsvzo'] > 80 else 'OVERSOLD (<-80)' if latest['fsvzo'] < -80 else 'NEUTRAL'}",
    ]

    if latest.get("bull_divergence", False):
        lines.append("  ⚠️ BULLISH DIVERGENCE detected (price new low, VZO not)")
    if latest.get("bear_divergence", False):
        lines.append("  ⚠️ BEARISH DIVERGENCE detected (price new high, VZO not)")
    if latest.get("extreme_bull", False):
        lines.append("  🟢 EXTREME BULLISH signal (reversal from oversold)")
    if latest.get("extreme_bear", False):
        lines.append("  🔴 EXTREME BEARISH signal (reversal from overbought)")

    lines += [
        "",
        "### Hull Trend + Kahlman Filter",
        f"  Hull A (long): {latest['hull_a']:.2f}",
        f"  Hull B (short): {latest['hull_b']:.2f}",
        f"  Trend: {'BULLISH ▲' if latest['hull_trend'] > 0 else 'BEARISH ▼'}",
    ]
    if latest.get("hull_buy", False):
        lines.append("  🟢 HULL BUY SIGNAL (crossover)")
    if latest.get("hull_sell", False):
        lines.append("  🔴 HULL SELL SIGNAL (crossunder)")

    lines += [
        "",
        "### 3rd Generation Moving Average (NMA)",
        f"  NMA Value: {latest['nma']:.2f}",
        f"  Price vs NMA: {'ABOVE ▲' if latest['close'] > latest['nma'] else 'BELOW ▼'}",
        "",
        "### Bollinger Bands",
        f"  Upper: {latest['bb_upper']:.2f}  |  Middle: {latest['bb_middle']:.2f}  |  Lower: {latest['bb_lower']:.2f}",
        f"  Width: {latest['bb_width']:.4f}",
        f"  %B: {latest['bb_pct_b']:.2f}  ({'overbought' if latest['bb_pct_b'] > 1 else 'oversold' if latest['bb_pct_b'] < 0 else 'within bands'})",
        "",
        "### VWAP",
        f"  VWAP: {latest['vwap']:.2f}",
        f"  Price vs VWAP: {'ABOVE ▲' if latest['close'] > latest['vwap'] else 'BELOW ▼'}",
    ]

    # Summary table
    lines += [
        "",
        "### Summary Table",
        "| Indicator | Value | Signal |",
        "|-----------|-------|--------|",
        f"| FSVZO | {latest['fsvzo']:.1f} | {'Bullish' if latest['fsvzo'] > 0 else 'Bearish'} |",
        f"| Hull Trend | {'Bull' if latest['hull_trend'] > 0 else 'Bear'} | {'Buy' if latest.get('hull_buy', False) else 'Sell' if latest.get('hull_sell', False) else 'Hold'} |",
        f"| NMA | {latest['nma']:.2f} | {'Above' if latest['close'] > latest['nma'] else 'Below'} |",
        f"| BB %B | {latest['bb_pct_b']:.2f} | {'OB' if latest['bb_pct_b'] > 1 else 'OS' if latest['bb_pct_b'] < 0 else 'Neutral'} |",
        f"| VWAP | {latest['vwap']:.2f} | {'Above' if latest['close'] > latest['vwap'] else 'Below'} |",
    ]

    return "\n".join(lines)
