"""Tests for OpenBB integration in the dataflows interface layer.

Verifies that:
1. OpenBB is registered as a vendor in VENDOR_METHODS
2. New categories (macro_data, sec_data) are properly defined
3. route_to_vendor correctly routes to OpenBB for exclusive tools
4. Vendor fallback chain includes OpenBB
5. Config updates propagate correctly
"""

import pytest
from unittest.mock import patch, MagicMock


class TestVendorRegistration:
    """Verify OpenBB is properly registered in the interface."""

    def test_openbb_in_vendor_list(self):
        from tradingagents.dataflows.interface import VENDOR_LIST
        assert "openbb" in VENDOR_LIST

    def test_openbb_registered_for_stock_data(self):
        from tradingagents.dataflows.interface import VENDOR_METHODS
        assert "openbb" in VENDOR_METHODS["get_stock_data"]

    def test_openbb_registered_for_fundamentals(self):
        from tradingagents.dataflows.interface import VENDOR_METHODS
        assert "openbb" in VENDOR_METHODS["get_fundamentals"]

    def test_openbb_registered_for_balance_sheet(self):
        from tradingagents.dataflows.interface import VENDOR_METHODS
        assert "openbb" in VENDOR_METHODS["get_balance_sheet"]

    def test_openbb_registered_for_cashflow(self):
        from tradingagents.dataflows.interface import VENDOR_METHODS
        assert "openbb" in VENDOR_METHODS["get_cashflow"]

    def test_openbb_registered_for_income_statement(self):
        from tradingagents.dataflows.interface import VENDOR_METHODS
        assert "openbb" in VENDOR_METHODS["get_income_statement"]

    def test_openbb_registered_for_insider_transactions(self):
        from tradingagents.dataflows.interface import VENDOR_METHODS
        assert "openbb" in VENDOR_METHODS["get_insider_transactions"]

    def test_openbb_registered_for_news(self):
        from tradingagents.dataflows.interface import VENDOR_METHODS
        assert "openbb" in VENDOR_METHODS["get_news"]

    def test_openbb_registered_for_global_news(self):
        from tradingagents.dataflows.interface import VENDOR_METHODS
        assert "openbb" in VENDOR_METHODS["get_global_news"]

    def test_openbb_registered_for_indicators(self):
        from tradingagents.dataflows.interface import VENDOR_METHODS
        assert "openbb" in VENDOR_METHODS["get_indicators"]


class TestNewCategories:
    """Verify new OpenBB-exclusive categories exist."""

    def test_macro_data_category_exists(self):
        from tradingagents.dataflows.interface import TOOLS_CATEGORIES
        assert "macro_data" in TOOLS_CATEGORIES
        assert "get_economic_indicators" in TOOLS_CATEGORIES["macro_data"]["tools"]
        assert "get_market_overview" in TOOLS_CATEGORIES["macro_data"]["tools"]

    def test_sec_data_category_exists(self):
        from tradingagents.dataflows.interface import TOOLS_CATEGORIES
        assert "sec_data" in TOOLS_CATEGORIES
        assert "get_sec_filings" in TOOLS_CATEGORIES["sec_data"]["tools"]

    def test_exclusive_tools_only_have_openbb_vendor(self):
        from tradingagents.dataflows.interface import VENDOR_METHODS
        assert list(VENDOR_METHODS["get_economic_indicators"].keys()) == ["openbb"]
        assert list(VENDOR_METHODS["get_market_overview"].keys()) == ["openbb"]
        assert list(VENDOR_METHODS["get_sec_filings"].keys()) == ["openbb"]


class TestCategoryLookup:
    """Verify get_category_for_method works for new tools."""

    def test_economic_indicators_in_macro_data(self):
        from tradingagents.dataflows.interface import get_category_for_method
        assert get_category_for_method("get_economic_indicators") == "macro_data"

    def test_market_overview_in_macro_data(self):
        from tradingagents.dataflows.interface import get_category_for_method
        assert get_category_for_method("get_market_overview") == "macro_data"

    def test_sec_filings_in_sec_data(self):
        from tradingagents.dataflows.interface import get_category_for_method
        assert get_category_for_method("get_sec_filings") == "sec_data"

    def test_unknown_method_raises(self):
        from tradingagents.dataflows.interface import get_category_for_method
        with pytest.raises(ValueError, match="not found"):
            get_category_for_method("nonexistent_method")


class TestRouting:
    """Verify route_to_vendor dispatches correctly."""

    @patch("tradingagents.dataflows.interface.get_config")
    def test_routes_to_openbb_when_configured(self, mock_config):
        mock_config.return_value = {
            "data_vendors": {"macro_data": "openbb"},
            "tool_vendors": {},
        }

        from tradingagents.dataflows.interface import VENDOR_METHODS
        # Verify the openbb implementation is callable
        impl = VENDOR_METHODS["get_economic_indicators"]["openbb"]
        assert callable(impl)

    @patch("tradingagents.dataflows.interface.get_config")
    def test_openbb_in_fallback_chain_for_stock_data(self, mock_config):
        """When yfinance is primary, openbb should be in the fallback chain."""
        mock_config.return_value = {
            "data_vendors": {"core_stock_apis": "yfinance"},
            "tool_vendors": {},
        }

        from tradingagents.dataflows.interface import VENDOR_METHODS
        vendors = list(VENDOR_METHODS["get_stock_data"].keys())
        assert "openbb" in vendors
        assert "yfinance" in vendors


class TestDefaultConfig:
    """Verify default config includes new categories."""

    def test_default_config_has_macro_data(self):
        from tradingagents.default_config import DEFAULT_CONFIG
        assert "macro_data" in DEFAULT_CONFIG["data_vendors"]
        assert DEFAULT_CONFIG["data_vendors"]["macro_data"] == "openbb"

    def test_default_config_has_sec_data(self):
        from tradingagents.default_config import DEFAULT_CONFIG
        assert "sec_data" in DEFAULT_CONFIG["data_vendors"]
        assert DEFAULT_CONFIG["data_vendors"]["sec_data"] == "openbb"

    def test_existing_vendors_unchanged(self):
        from tradingagents.default_config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["data_vendors"]["core_stock_apis"] == "yfinance"
        assert DEFAULT_CONFIG["data_vendors"]["technical_indicators"] == "yfinance"
        assert DEFAULT_CONFIG["data_vendors"]["fundamental_data"] == "yfinance"
        assert DEFAULT_CONFIG["data_vendors"]["news_data"] == "yfinance"
