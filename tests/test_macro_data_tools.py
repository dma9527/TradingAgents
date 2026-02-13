"""Tests for macro_data_tools langchain tool wrappers.

Verifies that the @tool-decorated functions correctly delegate
to route_to_vendor with the right method names.
"""

import pytest
from unittest.mock import patch


class TestMacroToolDelegation:
    """Verify tool wrappers call route_to_vendor correctly."""

    @patch("tradingagents.agents.utils.macro_data_tools.route_to_vendor")
    def test_get_economic_indicators_delegates(self, mock_route):
        mock_route.return_value = "FRED data here"

        from tradingagents.agents.utils.macro_data_tools import get_economic_indicators
        result = get_economic_indicators.invoke({
            "indicator": "FEDFUNDS",
            "start_date": "2025-01-01",
            "end_date": "2025-06-01",
        })

        mock_route.assert_called_once_with(
            "get_economic_indicators", "FEDFUNDS", "2025-01-01", "2025-06-01"
        )
        assert result == "FRED data here"

    @patch("tradingagents.agents.utils.macro_data_tools.route_to_vendor")
    def test_get_market_overview_delegates(self, mock_route):
        mock_route.return_value = "Market overview here"

        from tradingagents.agents.utils.macro_data_tools import get_market_overview
        result = get_market_overview.invoke({})

        mock_route.assert_called_once_with("get_market_overview")
        assert result == "Market overview here"

    @patch("tradingagents.agents.utils.macro_data_tools.route_to_vendor")
    def test_get_sec_filings_delegates(self, mock_route):
        mock_route.return_value = "SEC filings here"

        from tradingagents.agents.utils.macro_data_tools import get_sec_filings
        result = get_sec_filings.invoke({
            "ticker": "AAPL",
            "filing_type": "10-Q",
            "limit": 3,
        })

        mock_route.assert_called_once_with("get_sec_filings", "AAPL", "10-Q", 3)
        assert result == "SEC filings here"


class TestToolMetadata:
    """Verify tool decorators have proper metadata."""

    def test_economic_indicators_is_tool(self):
        from tradingagents.agents.utils.macro_data_tools import get_economic_indicators
        assert hasattr(get_economic_indicators, "name")
        assert get_economic_indicators.name == "get_economic_indicators"

    def test_market_overview_is_tool(self):
        from tradingagents.agents.utils.macro_data_tools import get_market_overview
        assert hasattr(get_market_overview, "name")
        assert get_market_overview.name == "get_market_overview"

    def test_sec_filings_is_tool(self):
        from tradingagents.agents.utils.macro_data_tools import get_sec_filings
        assert hasattr(get_sec_filings, "name")
        assert get_sec_filings.name == "get_sec_filings"
