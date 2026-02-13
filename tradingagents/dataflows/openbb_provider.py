"""OpenBB data provider adapter for TradingAgents.

Provides unified access to 30+ financial data sources through OpenBB Platform.
Phase 1: Free providers only (yfinance via OpenBB, SEC, FRED).

All functions match the existing vendor interface signatures so they can be
used as drop-in replacements in interface.py VENDOR_METHODS.
"""

import logging
from typing import Annotated, Optional
from datetime import datetime
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)

# Lazy-init singleton — avoids import cost until first use
_obb = None


def _get_obb():
    """Lazy-initialize the OpenBB SDK singleton."""
    global _obb
    if _obb is None:
        try:
            from openbb import obb

            # Configure credentials from environment variables
            import os
            fred_key = os.getenv("FRED_API_KEY") or os.getenv("OPENBB_FRED_API_KEY")
            if fred_key:
                obb.user.credentials.fred_api_key = fred_key

            _obb = obb
            logger.info("OpenBB SDK initialized successfully")
        except ImportError:
            raise ImportError(
                "openbb is not installed. "
                "Install with: pip install 'openbb[yfinance,sec,fred]'"
            )
    return _obb


# ---------------------------------------------------------------------------
# Core Stock APIs — matches get_YFin_data_online signature
# ---------------------------------------------------------------------------

def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Fetch OHLCV historical price data via OpenBB."""
    obb = _get_obb()
    try:
        result = obb.equity.price.historical(
            symbol=symbol.upper(),
            start_date=start_date,
            end_date=end_date,
            provider="yfinance",
        )
        df = result.to_dataframe()

        if df.empty:
            return (
                f"No data found for symbol '{symbol}' "
                f"between {start_date} and {end_date}"
            )

        # Normalize column names to match yfinance output
        col_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        df.rename(columns={k: v for k, v in col_map.items() if k in df.columns},
                  inplace=True)

        numeric_cols = ["Open", "High", "Low", "Close"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].round(2)

        csv_string = df.to_csv()
        header = (
            f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
            f"# Total records: {len(df)}\n"
            f"# Source: OpenBB (yfinance provider)\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        return header + csv_string

    except Exception as e:
        logger.warning(f"OpenBB get_stock_data failed for {symbol}: {e}")
        return f"Error fetching stock data for {symbol}: {str(e)}"


# ---------------------------------------------------------------------------
# Technical Indicators — matches get_stock_stats_indicators_window signature
# ---------------------------------------------------------------------------

def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get"],
    curr_date: Annotated[str, "current trading date, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    """Fetch technical indicators via OpenBB.

    Falls back to the existing stockstats-based implementation since OpenBB's
    technical indicator coverage uses the same yfinance data underneath.
    The real value of OpenBB for indicators will come when premium providers
    (Intrinio, FMP) are added in Phase 2.
    """
    # For Phase 1, delegate to the existing stockstats implementation
    # which is already well-tested and handles caching.
    from .y_finance import get_stock_stats_indicators_window
    return get_stock_stats_indicators_window(symbol, indicator, curr_date, look_back_days)


# ---------------------------------------------------------------------------
# Fundamental Data — matches get_fundamentals / get_balance_sheet / etc.
# ---------------------------------------------------------------------------

def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date"] = None,
) -> str:
    """Get company fundamentals overview via OpenBB."""
    obb = _get_obb()
    try:
        # Use OpenBB equity profile for overview
        profile = obb.equity.profile(
            symbol=ticker.upper(), provider="yfinance"
        )
        profile_df = profile.to_dataframe()

        if profile_df.empty:
            return f"No fundamentals data found for symbol '{ticker}'"

        row = profile_df.iloc[0]
        lines = []
        field_map = {
            "name": "Name",
            "sector": "Sector",
            "industry": "Industry",
            "market_cap": "Market Cap",
            "beta": "Beta",
        }
        for src, label in field_map.items():
            val = row.get(src)
            if val is not None and str(val) != "nan":
                lines.append(f"{label}: {val}")

        # Supplement with key metrics if available
        try:
            metrics_result = obb.equity.fundamental.metrics(
                symbol=ticker.upper(), provider="yfinance"
            )
            metrics_df = metrics_result.to_dataframe()
            if not metrics_df.empty:
                m = metrics_df.iloc[0]
                metric_fields = {
                    "pe_ratio": "PE Ratio (TTM)",
                    "forward_pe": "Forward PE",
                    "peg_ratio": "PEG Ratio",
                    "eps_ttm": "EPS (TTM)",
                    "dividend_yield": "Dividend Yield",
                    "return_on_equity": "Return on Equity",
                    "debt_to_equity": "Debt to Equity",
                    "current_ratio": "Current Ratio",
                    "revenue_per_share_ttm": "Revenue Per Share (TTM)",
                    "price_to_book": "Price to Book",
                }
                for src, label in metric_fields.items():
                    val = m.get(src)
                    if val is not None and str(val) != "nan":
                        lines.append(f"{label}: {val}")
        except Exception:
            pass  # Metrics endpoint may not be available for all providers

        header = (
            f"# Company Fundamentals for {ticker.upper()}\n"
            f"# Source: OpenBB\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        return header + "\n".join(lines)

    except Exception as e:
        logger.warning(f"OpenBB get_fundamentals failed for {ticker}: {e}")
        return f"Error retrieving fundamentals for {ticker}: {str(e)}"


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date"] = None,
) -> str:
    """Get balance sheet data via OpenBB."""
    obb = _get_obb()
    try:
        period = "quarter" if freq.lower() == "quarterly" else "annual"
        result = obb.equity.fundamental.balance(
            symbol=ticker.upper(),
            period=period,
            provider="yfinance",
            limit=8,
        )
        df = result.to_dataframe()

        if df.empty:
            return f"No balance sheet data found for symbol '{ticker}'"

        csv_string = df.to_csv()
        header = (
            f"# Balance Sheet data for {ticker.upper()} ({freq})\n"
            f"# Source: OpenBB\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        return header + csv_string

    except Exception as e:
        logger.warning(f"OpenBB get_balance_sheet failed for {ticker}: {e}")
        return f"Error retrieving balance sheet for {ticker}: {str(e)}"


def get_cashflow(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date"] = None,
) -> str:
    """Get cash flow data via OpenBB."""
    obb = _get_obb()
    try:
        period = "quarter" if freq.lower() == "quarterly" else "annual"
        result = obb.equity.fundamental.cash(
            symbol=ticker.upper(),
            period=period,
            provider="yfinance",
            limit=8,
        )
        df = result.to_dataframe()

        if df.empty:
            return f"No cash flow data found for symbol '{ticker}'"

        csv_string = df.to_csv()
        header = (
            f"# Cash Flow data for {ticker.upper()} ({freq})\n"
            f"# Source: OpenBB\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        return header + csv_string

    except Exception as e:
        logger.warning(f"OpenBB get_cashflow failed for {ticker}: {e}")
        return f"Error retrieving cash flow for {ticker}: {str(e)}"


def get_income_statement(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date"] = None,
) -> str:
    """Get income statement data via OpenBB."""
    obb = _get_obb()
    try:
        period = "quarter" if freq.lower() == "quarterly" else "annual"
        result = obb.equity.fundamental.income(
            symbol=ticker.upper(),
            period=period,
            provider="yfinance",
            limit=8,
        )
        df = result.to_dataframe()

        if df.empty:
            return f"No income statement data found for symbol '{ticker}'"

        csv_string = df.to_csv()
        header = (
            f"# Income Statement data for {ticker.upper()} ({freq})\n"
            f"# Source: OpenBB\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        return header + csv_string

    except Exception as e:
        logger.warning(f"OpenBB get_income_statement failed for {ticker}: {e}")
        return f"Error retrieving income statement for {ticker}: {str(e)}"


def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol of the company"],
) -> str:
    """Get insider transactions via OpenBB (SEC provider — free)."""
    obb = _get_obb()
    try:
        result = obb.equity.ownership.insider_trading(
            symbol=ticker.upper(),
            provider="sec",
            limit=50,
        )
        df = result.to_dataframe()

        if df.empty:
            return f"No insider transactions data found for symbol '{ticker}'"

        csv_string = df.to_csv()
        header = (
            f"# Insider Transactions data for {ticker.upper()}\n"
            f"# Source: OpenBB (SEC)\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        return header + csv_string

    except Exception as e:
        logger.warning(f"OpenBB get_insider_transactions failed for {ticker}: {e}")
        return f"Error retrieving insider transactions for {ticker}: {str(e)}"


# ---------------------------------------------------------------------------
# News Data — matches get_news_yfinance / get_global_news_yfinance signatures
# ---------------------------------------------------------------------------

def get_news(
    ticker: Annotated[str, "ticker symbol"],
    start_date: Annotated[str, "start date yyyy-mm-dd"],
    end_date: Annotated[str, "end date yyyy-mm-dd"],
) -> str:
    """Get stock-specific news via OpenBB."""
    obb = _get_obb()
    try:
        result = obb.news.company(
            symbol=ticker.upper(),
            start_date=start_date,
            end_date=end_date,
            provider="yfinance",
            limit=20,
        )
        df = result.to_dataframe()

        if df.empty:
            return f"No news found for {ticker} between {start_date} and {end_date}"

        news_str = ""
        for _, row in df.iterrows():
            title = row.get("title", "No title")
            source = row.get("source", row.get("publisher", "Unknown"))
            url = row.get("url", row.get("link", ""))
            summary = row.get("text", row.get("summary", ""))

            news_str += f"### {title} (source: {source})\n"
            if summary:
                # Truncate long summaries
                news_str += f"{str(summary)[:500]}\n"
            if url:
                news_str += f"Link: {url}\n"
            news_str += "\n"

        return (
            f"## {ticker} News, from {start_date} to {end_date}:\n\n"
            f"{news_str}"
        )

    except Exception as e:
        logger.warning(f"OpenBB get_news failed for {ticker}: {e}")
        return f"Error fetching news for {ticker}: {str(e)}"


def get_global_news(
    curr_date: Annotated[str, "current date yyyy-mm-dd"],
    look_back_days: int = 7,
    limit: int = 10,
) -> str:
    """Get global/macro market news via OpenBB."""
    obb = _get_obb()
    try:
        curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        start_dt = curr_dt - relativedelta(days=look_back_days)
        start_date = start_dt.strftime("%Y-%m-%d")

        result = obb.news.world(
            start_date=start_date,
            end_date=curr_date,
            provider="yfinance",
            limit=limit,
        )
        df = result.to_dataframe()

        if df.empty:
            return f"No global news found for {curr_date}"

        news_str = ""
        for _, row in df.iterrows():
            title = row.get("title", "No title")
            source = row.get("source", row.get("publisher", "Unknown"))
            url = row.get("url", row.get("link", ""))

            news_str += f"### {title} (source: {source})\n"
            if url:
                news_str += f"Link: {url}\n"
            news_str += "\n"

        return (
            f"## Global Market News, from {start_date} to {curr_date}:\n\n"
            f"{news_str}"
        )

    except Exception as e:
        logger.warning(f"OpenBB get_global_news failed: {e}")
        return f"Error fetching global news: {str(e)}"


# ===========================================================================
# NEW CAPABILITIES — not available in yfinance/alpha_vantage vendors
# ===========================================================================

def get_sec_filings(
    ticker: Annotated[str, "ticker symbol of the company"],
    filing_type: Annotated[str, "SEC filing type: 10-K, 10-Q, 8-K, etc."] = "10-K",
    limit: int = 5,
) -> str:
    """Get SEC filings for a company (FREE — SEC provider).

    This is a new capability not available in yfinance or alpha_vantage.
    Useful for the Fundamentals Analyst agent.
    """
    obb = _get_obb()
    try:
        result = obb.equity.fundamental.filings(
            symbol=ticker.upper(),
            type=filing_type,
            provider="sec",
            limit=limit,
        )
        df = result.to_dataframe()

        if df.empty:
            return f"No {filing_type} filings found for {ticker}"

        lines = []
        for _, row in df.iterrows():
            date = row.get("filing_date", row.get("date", "Unknown"))
            url = row.get("link", row.get("url", ""))
            desc = row.get("description", row.get("title", filing_type))
            lines.append(f"- [{date}] {desc}")
            if url:
                lines.append(f"  URL: {url}")

        header = (
            f"# SEC {filing_type} Filings for {ticker.upper()}\n"
            f"# Source: OpenBB (SEC)\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        return header + "\n".join(lines)

    except Exception as e:
        logger.warning(f"OpenBB get_sec_filings failed for {ticker}: {e}")
        return f"Error retrieving SEC filings for {ticker}: {str(e)}"


def get_economic_indicators(
    indicator: Annotated[str, "FRED series ID, e.g. 'GDP', 'UNRATE', 'CPIAUCSL'"],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """Get macroeconomic data from FRED (FREE).

    This is a new capability not available in yfinance or alpha_vantage.
    Useful for the Market Analyst agent to understand macro context.

    Common series IDs:
        GDP       — Gross Domestic Product
        UNRATE    — Unemployment Rate
        CPIAUCSL  — Consumer Price Index
        FEDFUNDS  — Federal Funds Rate
        DGS10     — 10-Year Treasury Rate
        VIXCLS    — VIX Volatility Index
        SP500     — S&P 500 Index
    """
    obb = _get_obb()
    try:
        kwargs = {"symbol": indicator, "provider": "fred"}
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date

        result = obb.economy.fred_series(**kwargs)
        df = result.to_dataframe()

        if df.empty:
            return f"No data found for FRED series '{indicator}'"

        # Show last 20 data points for context
        recent = df.tail(20)
        csv_string = recent.to_csv()

        header = (
            f"# FRED Economic Data: {indicator}\n"
            f"# Source: OpenBB (FRED)\n"
            f"# Showing last {len(recent)} data points\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        return header + csv_string

    except Exception as e:
        logger.warning(f"OpenBB get_economic_indicators failed for {indicator}: {e}")
        return f"Error retrieving FRED data for {indicator}: {str(e)}"


def get_market_overview() -> str:
    """Get a snapshot of major market indices and economic indicators.

    Combines multiple data points for the Market Analyst agent.
    """
    obb = _get_obb()
    sections = []

    # Major indices
    indices = ["^GSPC", "^DJI", "^IXIC", "^VIX"]
    index_names = {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones",
        "^IXIC": "NASDAQ",
        "^VIX": "VIX",
    }

    for idx in indices:
        try:
            result = obb.equity.price.historical(
                symbol=idx, provider="yfinance", limit=5
            )
            df = result.to_dataframe()
            if not df.empty:
                latest = df.iloc[-1]
                name = index_names.get(idx, idx)
                close = latest.get("close", "N/A")
                sections.append(f"{name}: {close}")
        except Exception:
            pass

    # Key FRED indicators (latest values)
    fred_series = {
        "FEDFUNDS": "Fed Funds Rate",
        "DGS10": "10Y Treasury",
        "UNRATE": "Unemployment Rate",
    }

    for series_id, label in fred_series.items():
        try:
            result = obb.economy.fred_series(symbol=series_id, provider="fred")
            df = result.to_dataframe()
            if not df.empty:
                # Get the last value from the rightmost data column
                data_cols = [c for c in df.columns if c != "date"]
                if data_cols:
                    val = df[data_cols[0]].iloc[-1]
                    sections.append(f"{label}: {val}")
        except Exception:
            pass

    if not sections:
        return "Unable to retrieve market overview data"

    header = (
        f"# Market Overview\n"
        f"# Source: OpenBB\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + "\n".join(sections)
