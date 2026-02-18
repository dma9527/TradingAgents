"""Tests for advanced_indicators module (FSVZO, Hull+Kahlman, NMA, BB, VWAP)."""

import pytest
import numpy as np
import pandas as pd
from tradingagents.dataflows.advanced_indicators import (
    fsvzo,
    fsvzo_signals,
    hull_kahlman_trend,
    nma_3gen,
    bollinger_bands,
    vwap_line,
    compute_all_advanced,
    generate_advanced_report,
    _ema,
    _sma,
    _wma,
    _fourier_smooth,
    _kahlman_filter,
)


@pytest.fixture
def sample_df():
    """Create a realistic OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1_000_000, 10_000_000, n).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class TestUtilities:
    def test_ema_length(self, sample_df):
        result = _ema(sample_df["close"], 10)
        assert len(result) == len(sample_df)
        assert not result.isna().all()

    def test_sma_length(self, sample_df):
        result = _sma(sample_df["close"], 20)
        assert len(result) == len(sample_df)

    def test_wma_length(self, sample_df):
        result = _wma(sample_df["close"], 10)
        assert len(result) == len(sample_df)

    def test_fourier_smooth(self, sample_df):
        result = _fourier_smooth(sample_df["close"], 31)
        assert len(result) == len(sample_df)
        assert not result.isna().all()

    def test_kahlman_filter(self, sample_df):
        result = _kahlman_filter(sample_df["close"], gain=10000)
        assert len(result) == len(sample_df)
        # Kahlman should track close reasonably
        corr = np.corrcoef(sample_df["close"].values, result.values)[0, 1]
        assert corr > 0.85


class TestFSVZO:
    def test_output_columns(self, sample_df):
        result = fsvzo(sample_df)
        assert "fsvzo" in result.columns
        assert "fsvzo_signal" in result.columns
        assert "flow_momentum" in result.columns

    def test_range_bounded(self, sample_df):
        result = fsvzo(sample_df)
        assert result["fsvzo"].max() <= 100
        assert result["fsvzo"].min() >= -100

    def test_signal_length(self, sample_df):
        result = fsvzo(sample_df)
        assert len(result) == len(sample_df)

    def test_custom_params(self, sample_df):
        result = fsvzo(sample_df, length=14, signal_length=5, fourier_length=21)
        assert not result["fsvzo"].isna().all()


class TestFSVZOSignals:
    def test_output_columns(self, sample_df):
        vzo_df = fsvzo(sample_df)
        sigs = fsvzo_signals(sample_df, vzo_df)
        expected = ["bull_signal", "bear_signal", "extreme_bull", "extreme_bear",
                     "overbought", "oversold", "bull_divergence", "bear_divergence"]
        for col in expected:
            assert col in sigs.columns

    def test_signals_are_boolean(self, sample_df):
        vzo_df = fsvzo(sample_df)
        sigs = fsvzo_signals(sample_df, vzo_df)
        assert sigs["bull_signal"].dtype == bool
        assert sigs["bear_signal"].dtype == bool


class TestHullKahlman:
    def test_output_columns(self, sample_df):
        result = hull_kahlman_trend(sample_df)
        assert "hull_a" in result.columns
        assert "hull_b" in result.columns
        assert "hull_trend" in result.columns
        assert "hull_buy" in result.columns
        assert "hull_sell" in result.columns

    def test_trend_values(self, sample_df):
        result = hull_kahlman_trend(sample_df)
        unique = set(result["hull_trend"].dropna().unique())
        assert unique.issubset({-1, 1})

    def test_without_kahlman(self, sample_df):
        result = hull_kahlman_trend(sample_df, use_kahlman=False)
        assert not result["hull_a"].isna().all()


class TestNMA:
    def test_output_length(self, sample_df):
        result = nma_3gen(sample_df)
        assert len(result) == len(sample_df)

    def test_tracks_price(self, sample_df):
        result = nma_3gen(sample_df)
        valid = ~result.isna()
        corr = np.corrcoef(
            sample_df["close"].values[valid], result.values[valid]
        )[0, 1]
        assert corr > 0.8


class TestBollingerBands:
    def test_output_columns(self, sample_df):
        result = bollinger_bands(sample_df)
        for col in ["bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_pct_b"]:
            assert col in result.columns

    def test_upper_above_lower(self, sample_df):
        result = bollinger_bands(sample_df)
        valid = result.dropna()
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()


class TestVWAP:
    def test_output_length(self, sample_df):
        result = vwap_line(sample_df)
        assert len(result) == len(sample_df)


class TestComputeAll:
    def test_all_columns_present(self, sample_df):
        result = compute_all_advanced(sample_df)
        expected_cols = [
            "fsvzo", "fsvzo_signal", "flow_momentum",
            "hull_a", "hull_b", "hull_trend",
            "nma", "bb_middle", "bb_upper", "bb_lower", "vwap",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"


class TestReport:
    def test_report_contains_sections(self, sample_df):
        report = generate_advanced_report(sample_df, "AAPL")
        assert "FSVZO" in report
        assert "Hull" in report
        assert "NMA" in report
        assert "Bollinger" in report
        assert "VWAP" in report
        assert "Summary Table" in report
