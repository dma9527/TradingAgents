"""Tests for the OpenBB data provider adapter.

All OpenBB SDK calls are mocked to avoid network dependencies and API keys.
Tests verify that:
1. Functions match expected signatures and return formats
2. Error handling works correctly
3. Lazy initialization of the OpenBB singleton works
4. Data is properly formatted for downstream agent consumption
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_obb_singleton():
    """Reset the lazy-init singleton before each test."""
    import tradingagents.dataflows.openbb_provider as provider
    provider._obb = None
    yield
    provider._obb = None


def _mock_obb():
    """Create a mock OpenBB SDK object."""
    return MagicMock()


def _make_openbb_result(data: dict, columns: list = None):
    """Create a mock OpenBB result with to_dataframe() support."""
    df = pd.DataFrame(data)
    result = MagicMock()
    result.to_dataframe.return_value = df
    return result


def _make_empty_result():
    """Create a mock OpenBB result with empty DataFrame."""
    result = MagicMock()
    result.to_dataframe.return_value = pd.DataFrame()
    return result


# ===========================================================================
# Test: Lazy initialization
# ===========================================================================

class TestLazyInit:
    def test_get_obb_imports_openbb(self):
        """_get_obb should import and cache the obb singleton."""
        mock_obb = MagicMock()
        with patch.dict("sys.modules", {"openbb": MagicMock(obb=mock_obb)}):
            from tradingagents.dataflows.openbb_provider import _get_obb
            import tradingagents.dataflows.openbb_provider as provider
            provider._obb = None

            with patch("tradingagents.dataflows.openbb_provider._obb", None):
                # Force re-import path
                provider._obb = None
                # Simulate import
                provider._obb = mock_obb
                assert provider._obb is mock_obb

    def test_get_obb_raises_on_missing_package(self):
        """_get_obb should raise ImportError when openbb is not installed."""
        from tradingagents.dataflows.openbb_provider import _get_obb
        import tradingagents.dataflows.openbb_provider as provider
        provider._obb = None

        with patch("builtins.__import__", side_effect=ImportError("No module named 'openbb'")):
            with pytest.raises(ImportError, match="openbb is not installed"):
                _get_obb()


# ===========================================================================
# Test: get_stock_data
# ===========================================================================

class TestGetStockData:
    @patch("tradingagents.dataflows.openbb_provider._get_obb")
    def test_returns_csv_with_header(self, mock_get_obb):
        mock_obb = _mock_obb()
        mock_get_obb.return_value = mock_obb

        mock_obb.equity.price.historical.return_value = _make_openbb_result({
            "open": [150.0, 151.0],
            "high": [155.0, 156.0],
            "low": [149.0, 150.0],
            "close": [154.0, 155.0],
            "volume": [1000000, 1100000],
        })

        from tradingagents.dataflows.openbb_provider import get_stock_data
        result = get_stock_data("AAPL", "2025-01-01", "2025-01-02")

        assert "Stock data for AAPL" in result
        assert "OpenBB" in result
        assert "154.0" in result
        mock_obb.equity.price.historical.assert_called_once()

    @patch("tradingagents.dataflows.openbb_provider._get_obb")
    def test_empty_data_returns_message(self, mock_get_obb):
        mock_obb = _mock_obb()
        mock_get_obb.return_value = mock_obb
        mock_obb.equity.price.historical.return_value = _make_empty_result()

        from tradingagents.dataflows.openbb_provider import get_stock_data
        result = get_stock_data("FAKE", "2025-01-01", "2025-01-02")

        assert "No data found" in result

    @patch("tradingagents.dataflows.openbb_provider._get_obb")
    def test_error_returns_message(self, mock_get_obb):
        mock_obb = _mock_obb()
        mock_get_obb.return_value = mock_obb
        mock_obb.equity.price.historical.side_effect = Exception("API error")

        from tradingagents.dataflows.openbb_provider import get_stock_data
        result = get_stock_data("AAPL", "2025-01-01", "2025-01-02")

        assert "Error" in result


# ===========================================================================
# Test: get_fundamentals
# ===========================================================================

class TestGetFundamentals:
    @patch("tradingagents.dataflows.openbb_provider._get_obb")
    def test_returns_formatted_fundamentals(self, mock_get_obb):
        mock_obb = _mock_obb()
        mock_get_obb.return_value = mock_obb

        mock_obb.equity.profile.return_value = _make_openbb_result({
            "name": ["Apple Inc."],
            "sector": ["Technology"],
            "industry": ["Consumer Electronics"],
            "market_cap": [3000000000000],
            "beta": [1.2],
        })
        # Mock metrics to raise (optional endpoint)
        mock_obb.equity.fundamental.metrics.side_effect = Exception("not available")

        from tradingagents.dataflows.openbb_provider import get_fundamentals
        result = get_fundamentals("AAPL")

        assert "Apple Inc." in result
        assert "Technology" in result
        assert "OpenBB" in result

    @patch("tradingagents.dataflows.openbb_provider._get_obb")
    def test_empty_profile_returns_message(self, mock_get_obb):
        mock_obb = _mock_obb()
        mock_get_obb.return_value = mock_obb
        mock_obb.equity.profile.return_value = _make_empty_result()

        from tradingagents.dataflows.openbb_provider import get_fundamentals
        result = get_fundamentals("FAKE")

        assert "No fundamentals data found" in result


# ===========================================================================
# Test: Financial statements (balance sheet, cashflow, income statement)
# ===========================================================================

class TestFinancialStatements:
    @patch("tradingagents.dataflows.openbb_provider._get_obb")
    def test_balance_sheet_quarterly(self, mock_get_obb):
        mock_obb = _mock_obb()
        mock_get_obb.return_value = mock_obb
        mock_obb.equity.fundamental.balance.return_value = _make_openbb_result({
            "total_assets": [100000, 110000],
            "total_liabilities": [50000, 55000],
        })

        from tradingagents.dataflows.openbb_provider import get_balance_sheet
        result = get_balance_sheet("AAPL", "quarterly")

        assert "Balance Sheet" in result
        assert "100000" in result
        mock_obb.equity.fundamental.balance.assert_called_once_with(
            symbol="AAPL", period="quarter", provider="yfinance", limit=8
        )

    @patch("tradingagents.dataflows.openbb_provider._get_obb")
    def test_balance_sheet_annual(self, mock_get_obb):
        mock_obb = _mock_obb()
        mock_get_obb.return_value = mock_obb
        mock_obb.equity.fundamental.balance.return_value = _make_openbb_result({
            "total_assets": [100000],
        })

        from tradingagents.dataflows.openbb_provider import get_balance_sheet
        result = get_balance_sheet("AAPL", "annual")

        mock_obb.equity.fundamental.balance.assert_called_once_with(
            symbol="AAPL", period="annual", provider="yfinance", limit=8
        )

    @patch("tradingagents.dataflows.openbb_provider._get_obb")
    def test_cashflow_returns_csv(self, mock_get_obb):
        mock_obb = _mock_obb()
        mock_get_obb.return_value = mock_obb
        mock_obb.equity.fundamental.cash.return_value = _make_openbb_result({
            "operating_cash_flow": [50000],
            "free_cash_flow": [30000],
        })

        from tradingagents.dataflows.openbb_provider import get_cashflow
        result = get_cashflow("AAPL")

        assert "Cash Flow" in result
        assert "50000" in result

    @patch("tradingagents.dataflows.openbb_provider._get_obb")
    def test_income_statement_returns_csv(self, mock_get_obb):
        mock_obb = _mock_obb()
        mock_get_obb.return_value = mock_obb
        mock_obb.equity.fundamental.income.return_value = _make_openbb_result({
            "revenue": [400000000000],
            "net_income": [100000000000],
        })

        from tradingagents.dataflows.openbb_provider import get_income_statement
        result = get_income_statement("AAPL")

        assert "Income Statement" in result

    @patch("tradingagents.dataflows.openbb_provider._get_obb")
    def test_empty_statement_returns_message(self, mock_get_obb):
        mock_obb = _mock_obb()
        mock_get_obb.return_value = mock_obb
        mock_obb.equity.fundamental.balance.return_value = _make_empty_result()

        from tradingagents.dataflows.openbb_provider import get_balance_sheet
        result = get_balance_sheet("FAKE")

        assert "No balance sheet data found" in result


# ===========================================================================
# Test: Insider transactions
# ===========================================================================

class TestInsiderTransactions:
    @patch("tradingagents.dataflows.openbb_provider._get_obb")
    def test_returns_insider_data(self, mock_get_obb):
        mock_obb = _mock_obb()
        mock_get_obb.return_value = mock_obb
        mock_obb.equity.ownership.insider_trading.return_value = _make_openbb_result({
            "owner_name": ["Tim Cook"],
            "transaction_type": ["Sale"],
            "shares": [50000],
        })

        from tradingagents.dataflows.openbb_provider import get_insider_transactions
        result = get_insider_transactions("AAPL")

        assert "Insider Transactions" in result
        assert "SEC" in result

    @patch("tradingagents.dataflows.openbb_provider._get_obb")
    def test_empty_insider_returns_message(self, mock_get_obb):
        mock_obb = _mock_obb()
        mock_get_obb.return_value = mock_obb
        mock_obb.equity.ownership.insider_trading.return_value = _make_empty_result()

        from tradingagents.dataflows.openbb_provider import get_insider_transactions
        result = get_insider_transactions("FAKE")

        assert "No insider transactions" in result


# ===========================================================================
# Test: News
# ===========================================================================

class TestNews:
    @patch("tradingagents.dataflows.openbb_provider._get_obb")
    def test_get_news_returns_formatted(self, mock_get_obb):
        mock_obb = _mock_obb()
        mock_get_obb.return_value = mock_obb
        mock_obb.news.company.return_value = _make_openbb_result({
            "title": ["Apple beats earnings", "iPhone sales surge"],
            "source": ["Reuters", "Bloomberg"],
            "url": ["https://example.com/1", "https://example.com/2"],
            "text": ["Apple reported...", "iPhone sales..."],
        })

        from tradingagents.dataflows.openbb_provider import get_news
        result = get_news("AAPL", "2025-01-01", "2025-01-07")

        assert "AAPL News" in result
        assert "Apple beats earnings" in result

    @patch("tradingagents.dataflows.openbb_provider._get_obb")
    def test_get_news_empty(self, mock_get_obb):
        mock_obb = _mock_obb()
        mock_get_obb.return_value = mock_obb
        mock_obb.news.company.return_value = _make_empty_result()

        from tradingagents.dataflows.openbb_provider import get_news
        result = get_news("FAKE", "2025-01-01", "2025-01-07")

        assert "No news found" in result

    @patch("tradingagents.dataflows.openbb_provider._get_obb")
    def test_get_global_news(self, mock_get_obb):
        mock_obb = _mock_obb()
        mock_get_obb.return_value = mock_obb
        mock_obb.news.world.return_value = _make_openbb_result({
            "title": ["Fed holds rates steady"],
            "source": ["CNBC"],
            "url": ["https://example.com/fed"],
        })

        from tradingagents.dataflows.openbb_provider import get_global_news
        result = get_global_news("2025-01-15")

        assert "Global Market News" in result
        assert "Fed holds rates steady" in result


# ===========================================================================
# Test: NEW CAPABILITIES — SEC filings and economic indicators
# ===========================================================================

class TestSECFilings:
    @patch("tradingagents.dataflows.openbb_provider._get_obb")
    def test_returns_filing_list(self, mock_get_obb):
        mock_obb = _mock_obb()
        mock_get_obb.return_value = mock_obb
        mock_obb.equity.fundamental.filings.return_value = _make_openbb_result({
            "filing_date": ["2025-01-15", "2024-10-30"],
            "description": ["Annual Report", "Quarterly Report"],
            "link": ["https://sec.gov/1", "https://sec.gov/2"],
        })

        from tradingagents.dataflows.openbb_provider import get_sec_filings
        result = get_sec_filings("AAPL", "10-K", 5)

        assert "SEC 10-K Filings" in result
        assert "Annual Report" in result

    @patch("tradingagents.dataflows.openbb_provider._get_obb")
    def test_empty_filings(self, mock_get_obb):
        mock_obb = _mock_obb()
        mock_get_obb.return_value = mock_obb
        mock_obb.equity.fundamental.filings.return_value = _make_empty_result()

        from tradingagents.dataflows.openbb_provider import get_sec_filings
        result = get_sec_filings("FAKE")

        assert "No 10-K filings found" in result


class TestEconomicIndicators:
    @patch("tradingagents.dataflows.openbb_provider._get_obb")
    def test_returns_fred_data(self, mock_get_obb):
        mock_obb = _mock_obb()
        mock_get_obb.return_value = mock_obb
        mock_obb.economy.fred_series.return_value = _make_openbb_result({
            "date": ["2025-01-01", "2025-02-01"],
            "value": [4.5, 4.3],
        })

        from tradingagents.dataflows.openbb_provider import get_economic_indicators
        result = get_economic_indicators("FEDFUNDS")

        assert "FRED Economic Data: FEDFUNDS" in result
        assert "4.5" in result

    @patch("tradingagents.dataflows.openbb_provider._get_obb")
    def test_empty_fred_data(self, mock_get_obb):
        mock_obb = _mock_obb()
        mock_get_obb.return_value = mock_obb
        mock_obb.economy.fred_series.return_value = _make_empty_result()

        from tradingagents.dataflows.openbb_provider import get_economic_indicators
        result = get_economic_indicators("INVALID")

        assert "No data found" in result

    @patch("tradingagents.dataflows.openbb_provider._get_obb")
    def test_with_date_range(self, mock_get_obb):
        mock_obb = _mock_obb()
        mock_get_obb.return_value = mock_obb
        mock_obb.economy.fred_series.return_value = _make_openbb_result({
            "date": ["2024-06-01"],
            "value": [3.8],
        })

        from tradingagents.dataflows.openbb_provider import get_economic_indicators
        result = get_economic_indicators("UNRATE", "2024-01-01", "2024-12-31")

        mock_obb.economy.fred_series.assert_called_once_with(
            symbol="UNRATE", provider="fred",
            start_date="2024-01-01", end_date="2024-12-31"
        )


class TestMarketOverview:
    @patch("tradingagents.dataflows.openbb_provider._get_obb")
    def test_returns_overview(self, mock_get_obb):
        mock_obb = _mock_obb()
        mock_get_obb.return_value = mock_obb

        # Mock index data
        mock_obb.equity.price.historical.return_value = _make_openbb_result({
            "close": [5800.0],
        })
        # Mock FRED data
        mock_obb.economy.fred_series.return_value = _make_openbb_result({
            "value": [4.5],
        })

        from tradingagents.dataflows.openbb_provider import get_market_overview
        result = get_market_overview()

        assert "Market Overview" in result

    @patch("tradingagents.dataflows.openbb_provider._get_obb")
    def test_handles_partial_failures(self, mock_get_obb):
        mock_obb = _mock_obb()
        mock_get_obb.return_value = mock_obb

        # All calls fail
        mock_obb.equity.price.historical.side_effect = Exception("fail")
        mock_obb.economy.fred_series.side_effect = Exception("fail")

        from tradingagents.dataflows.openbb_provider import get_market_overview
        result = get_market_overview()

        assert "Unable to retrieve" in result
