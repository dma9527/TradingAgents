"""Macro economic data tools powered by OpenBB (FRED, market indices).

These tools provide capabilities not available in yfinance or alpha_vantage.
They are used by the Market Analyst and News Analyst agents.
"""

from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_economic_indicators(
    indicator: Annotated[str, "FRED series ID, e.g. GDP, UNRATE, CPIAUCSL, FEDFUNDS, DGS10"],
    start_date: Annotated[str, "Start date yyyy-mm-dd (optional)"] = None,
    end_date: Annotated[str, "End date yyyy-mm-dd (optional)"] = None,
) -> str:
    """
    Retrieve macroeconomic indicator data from FRED via OpenBB.

    Common series IDs:
        GDP       - Gross Domestic Product
        UNRATE    - Unemployment Rate
        CPIAUCSL  - Consumer Price Index
        FEDFUNDS  - Federal Funds Rate
        DGS10     - 10-Year Treasury Rate
        VIXCLS    - VIX Volatility Index

    Args:
        indicator: FRED series ID
        start_date: Optional start date
        end_date: Optional end date
    Returns:
        Formatted string with recent economic data points
    """
    return route_to_vendor("get_economic_indicators", indicator, start_date, end_date)


@tool
def get_market_overview() -> str:
    """
    Get a snapshot of major market indices and key economic indicators.

    Returns current values for S&P 500, Dow Jones, NASDAQ, VIX,
    Fed Funds Rate, 10Y Treasury, and Unemployment Rate.

    Returns:
        Formatted string with market overview data
    """
    return route_to_vendor("get_market_overview")


@tool
def get_sec_filings(
    ticker: Annotated[str, "ticker symbol of the company"],
    filing_type: Annotated[str, "SEC filing type: 10-K, 10-Q, 8-K"] = "10-K",
    limit: Annotated[int, "max number of filings to return"] = 5,
) -> str:
    """
    Retrieve SEC filings for a company via OpenBB.

    Useful for accessing original 10-K (annual), 10-Q (quarterly),
    and 8-K (material events) filings directly from SEC.

    Args:
        ticker: Stock ticker symbol
        filing_type: Type of SEC filing (default 10-K)
        limit: Maximum filings to return (default 5)
    Returns:
        Formatted string with filing dates and links
    """
    return route_to_vendor("get_sec_filings", ticker, filing_type, limit)
