"""
LangChain tools exposing the advanced indicators (FSVZO, Hull+Kahlman, NMA, BB, VWAP)
to the Market Analyst agent as the PRIMARY technical analysis toolkit.
"""

from langchain_core.tools import tool
from typing import Annotated
import pandas as pd
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.advanced_indicators import (
    compute_all_advanced,
    generate_advanced_report,
    fsvzo,
    fsvzo_signals,
    hull_kahlman_trend,
    nma_3gen,
    bollinger_bands,
    vwap_line,
)


def _fetch_ohlcv(symbol: str, curr_date: str, lookback_days: int = 200) -> pd.DataFrame:
    """Fetch OHLCV data and return as a clean DataFrame."""
    import yfinance as yf
    from datetime import datetime, timedelta

    end = datetime.strptime(curr_date, "%Y-%m-%d") + timedelta(days=1)
    start = end - timedelta(days=lookback_days)

    data = yf.download(
        symbol,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        multi_level_index=False,
        progress=False,
        auto_adjust=True,
    )
    if data.empty:
        raise ValueError(f"No data available for {symbol}")

    data.columns = [c.lower() for c in data.columns]
    return data


@tool
def get_advanced_analysis(
    symbol: Annotated[str, "Ticker symbol (e.g. AAPL, TSLA)"],
    curr_date: Annotated[str, "Current trading date, YYYY-mm-dd"],
    lookback_days: Annotated[int, "Days of history to analyze"] = 200,
) -> str:
    """
    PRIMARY technical analysis tool using advanced algorithms:
    - FSVZO (Fourier-Smoothed Volume Zone Oscillator) with divergence detection
    - Hull Moving Average + Kahlman filter trend system
    - 3rd Generation Moving Average (NMA) for lag-free trend
    - Bollinger Bands with %B and width
    - VWAP

    Returns a detailed report with all indicator readings, signals, and a summary table.
    This should be called FIRST before any legacy indicators.
    """
    df = _fetch_ohlcv(symbol, curr_date, lookback_days)
    return generate_advanced_report(df, symbol)


@tool
def get_fsvzo(
    symbol: Annotated[str, "Ticker symbol"],
    curr_date: Annotated[str, "Current trading date, YYYY-mm-dd"],
    length: Annotated[int, "VZO calculation length"] = 9,
    lookback_days: Annotated[int, "Days of history"] = 60,
) -> str:
    """
    Get FSVZO (Fourier-Smoothed Volume Zone Oscillator) readings.
    Measures buying vs selling volume pressure with Fourier smoothing.
    Range: -100 to +100. Above 80 = overbought, below -80 = oversold.
    """
    df = _fetch_ohlcv(symbol, curr_date, lookback_days)
    vzo_df = fsvzo(df, length=length)
    sigs = fsvzo_signals(df, vzo_df)

    recent = pd.concat([vzo_df, sigs], axis=1).tail(20)
    lines = [f"## FSVZO for {symbol} (last 20 bars)\n"]
    lines.append("| Date | VZO | Signal | Flow Mom | Bull Sig | Bear Sig | Divergence |")
    lines.append("|------|-----|--------|----------|----------|----------|------------|")

    for idx, row in recent.iterrows():
        date_str = str(idx)[:10]
        div = ""
        if row.get("bull_divergence", False):
            div = "BULL DIV"
        elif row.get("bear_divergence", False):
            div = "BEAR DIV"
        lines.append(
            f"| {date_str} | {row['fsvzo']:.1f} | {row['fsvzo_signal']:.1f} | "
            f"{row['flow_momentum']:.1f} | {'✓' if row.get('bull_signal', False) else ''} | "
            f"{'✓' if row.get('bear_signal', False) else ''} | {div} |"
        )
    return "\n".join(lines)


@tool
def get_hull_trend(
    symbol: Annotated[str, "Ticker symbol"],
    curr_date: Annotated[str, "Current trading date, YYYY-mm-dd"],
    length: Annotated[int, "Hull MA lookback period"] = 24,
    lookback_days: Annotated[int, "Days of history"] = 60,
) -> str:
    """
    Get Hull Moving Average + Kahlman filter trend readings.
    Ultra-low lag trend detection with buy/sell crossover signals.
    """
    df = _fetch_ohlcv(symbol, curr_date, lookback_days)
    hull = hull_kahlman_trend(df, length=length)

    recent = hull.tail(20)
    lines = [f"## Hull+Kahlman Trend for {symbol} (last 20 bars)\n"]
    lines.append("| Date | Hull A | Hull B | Trend | Buy | Sell |")
    lines.append("|------|--------|--------|-------|-----|------|")

    for idx, row in recent.iterrows():
        date_str = str(idx)[:10]
        trend = "BULL ▲" if row["hull_trend"] > 0 else "BEAR ▼"
        lines.append(
            f"| {date_str} | {row['hull_a']:.2f} | {row['hull_b']:.2f} | "
            f"{trend} | {'✓' if row['hull_buy'] else ''} | {'✓' if row['hull_sell'] else ''} |"
        )
    return "\n".join(lines)
