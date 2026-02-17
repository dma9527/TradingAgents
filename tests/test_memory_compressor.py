"""Tests for MemoryCompressor.

Covers:
- LLM call failure returns original summary (Requirement 6.4)
- Summary append logic (Requirement 6.2)
- Max length truncation (Requirement 6.3)
- No LLM client fallback
- Empty rounds handling
- LLM client with invoke method (LangChain style)
- LLM client as plain callable

Requirements: 6.1, 6.2, 6.3, 6.4
"""

import logging
from unittest.mock import MagicMock

import pytest

from tradingagents.memory.compressor import MemoryCompressor
from tradingagents.memory.types import AnalysisRound


def _make_round(
    query: str = "AAPL",
    consensus: str = "建议持有",
    key_points: list[str] | None = None,
    timestamp: str = "2024-12-01T10:00:00",
) -> AnalysisRound:
    return AnalysisRound(
        query=query,
        consensus=consensus,
        key_points=key_points if key_points is not None else ["技术面突破", "基本面稳健"],
        timestamp=timestamp,
    )


class TestLLMFailure:
    """LLM call failure should return existing summary unchanged. Req 6.4"""

    def test_invoke_raises_returns_existing_summary(
        self, caplog: pytest.LogCaptureFixture
    ):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM service unavailable")
        compressor = MemoryCompressor(llm_client=llm)

        existing = "之前的摘要内容"
        result = compressor.compress([_make_round()], existing)

        assert result == existing

    def test_invoke_raises_logs_error(self, caplog: pytest.LogCaptureFixture):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("timeout")
        compressor = MemoryCompressor(llm_client=llm)

        with caplog.at_level(logging.ERROR):
            compressor.compress([_make_round()], "old summary")

        assert "LLM compression failed" in caplog.text

    def test_callable_raises_returns_existing_summary(self):
        def bad_llm(_prompt: str) -> str:
            raise ConnectionError("network error")

        compressor = MemoryCompressor(llm_client=bad_llm)
        existing = "保留这段摘要"
        result = compressor.compress([_make_round()], existing)

        assert result == existing


class TestSummaryAppend:
    """New summary should be appended to existing summary. Req 6.2"""

    def test_appends_to_existing_summary(self):
        llm = MagicMock()
        llm.invoke.return_value = "新的压缩摘要"
        compressor = MemoryCompressor(llm_client=llm)

        result = compressor.compress([_make_round()], "已有摘要")

        assert result.startswith("已有摘要")
        assert "新的压缩摘要" in result

    def test_empty_existing_summary_uses_new_only(self):
        llm = MagicMock()
        llm.invoke.return_value = "全新摘要"
        compressor = MemoryCompressor(llm_client=llm)

        result = compressor.compress([_make_round()], "")

        assert result == "全新摘要"

    def test_llm_returns_empty_keeps_existing(self):
        llm = MagicMock()
        llm.invoke.return_value = ""
        compressor = MemoryCompressor(llm_client=llm)

        result = compressor.compress([_make_round()], "已有摘要")

        assert result == "已有摘要"


class TestMaxLengthTruncation:
    """Result must not exceed max_length. Req 6.3"""

    def test_truncates_to_max_length(self):
        llm = MagicMock()
        llm.invoke.return_value = "A" * 600
        compressor = MemoryCompressor(llm_client=llm)

        result = compressor.compress([_make_round()], "", max_length=500)

        assert len(result) <= 500

    def test_merged_result_truncated(self):
        llm = MagicMock()
        llm.invoke.return_value = "B" * 300
        compressor = MemoryCompressor(llm_client=llm)

        existing = "A" * 300
        result = compressor.compress([_make_round()], existing, max_length=100)

        assert len(result) <= 100

    def test_small_max_length(self):
        llm = MagicMock()
        llm.invoke.return_value = "摘要内容"
        compressor = MemoryCompressor(llm_client=llm)

        result = compressor.compress([_make_round()], "已有", max_length=5)

        assert len(result) <= 5

    def test_default_max_length_is_500(self):
        llm = MagicMock()
        llm.invoke.return_value = "X" * 1000
        compressor = MemoryCompressor(llm_client=llm)

        result = compressor.compress([_make_round()], "")

        assert len(result) <= 500


class TestNoLLMFallback:
    """Without an LLM client, compressor should use simple truncation."""

    def test_fallback_produces_summary(self):
        compressor = MemoryCompressor(llm_client=None)
        rounds = [_make_round(query="AAPL", consensus="看涨")]

        result = compressor.compress(rounds, "")

        assert "AAPL" in result
        assert "看涨" in result

    def test_fallback_appends_to_existing(self):
        compressor = MemoryCompressor(llm_client=None)
        rounds = [_make_round(query="TSLA", consensus="看跌")]

        result = compressor.compress(rounds, "旧摘要")

        assert result.startswith("旧摘要")
        assert "TSLA" in result

    def test_fallback_respects_max_length(self):
        compressor = MemoryCompressor(llm_client=None)
        rounds = [_make_round() for _ in range(10)]

        result = compressor.compress(rounds, "已有摘要", max_length=50)

        assert len(result) <= 50


class TestEmptyRounds:
    """Empty rounds list should return existing summary (possibly truncated)."""

    def test_empty_rounds_returns_existing(self):
        llm = MagicMock()
        compressor = MemoryCompressor(llm_client=llm)

        result = compressor.compress([], "保留摘要")

        assert result == "保留摘要"
        llm.invoke.assert_not_called()

    def test_empty_rounds_truncates_existing(self):
        compressor = MemoryCompressor(llm_client=None)

        result = compressor.compress([], "A" * 600, max_length=100)

        assert len(result) <= 100


class TestLLMClientStyles:
    """Support both invoke-method and callable LLM clients."""

    def test_invoke_method_with_content_attr(self):
        """LangChain-style response with .content attribute."""
        response = MagicMock()
        response.content = "LangChain摘要"
        llm = MagicMock()
        llm.invoke.return_value = response
        compressor = MemoryCompressor(llm_client=llm)

        result = compressor.compress([_make_round()], "")

        assert "LangChain摘要" in result

    def test_invoke_method_returns_string(self):
        """LLM client whose invoke returns a plain string."""
        llm = MagicMock()
        llm.invoke.return_value = "纯字符串摘要"
        compressor = MemoryCompressor(llm_client=llm)

        result = compressor.compress([_make_round()], "")

        assert "纯字符串摘要" in result

    def test_callable_client(self):
        """LLM client as a plain callable function."""

        def my_llm(prompt: str) -> str:
            return "callable摘要"

        compressor = MemoryCompressor(llm_client=my_llm)

        result = compressor.compress([_make_round()], "")

        assert "callable摘要" in result


class TestMultipleRounds:
    """Verify that multiple rounds are formatted and compressed."""

    def test_multiple_rounds_all_included_in_prompt(self):
        llm = MagicMock()
        llm.invoke.return_value = "多轮摘要"
        compressor = MemoryCompressor(llm_client=llm)

        rounds = [
            _make_round(query="AAPL", consensus="看涨", timestamp="2024-01-01T00:00:00"),
            _make_round(query="AAPL", consensus="看跌", timestamp="2024-02-01T00:00:00"),
            _make_round(query="AAPL", consensus="持有", timestamp="2024-03-01T00:00:00"),
        ]
        compressor.compress(rounds, "")

        prompt = llm.invoke.call_args[0][0]
        assert "看涨" in prompt
        assert "看跌" in prompt
        assert "持有" in prompt
