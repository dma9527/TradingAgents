"""Tests for StockMemory data model and types.

Covers:
- KeyFact, AnalysisRound, StockMemory, MemoryConfig dataclass creation
- new_stock_memory factory function
- Default values and field validation
- Requirements: 1.1, 1.2, 1.3, 1.4, 8.1, 8.2
"""

from datetime import datetime

from tradingagents.memory.types import (
    AnalysisRound,
    KeyFact,
    MemoryConfig,
    StockMemory,
    new_stock_memory,
)


class TestKeyFact:
    """Tests for KeyFact dataclass. Requirements: 1.3"""

    def test_create_fact_type(self):
        fact = KeyFact(
            type="fact",
            content="Revenue grew 10%",
            source="fundamentals_analyst",
            keywords=["revenue", "growth"],
            timestamp="2024-12-01T10:00:00",
            weight=0.8,
        )
        assert fact.type == "fact"
        assert fact.content == "Revenue grew 10%"
        assert fact.source == "fundamentals_analyst"
        assert fact.keywords == ["revenue", "growth"]
        assert fact.timestamp == "2024-12-01T10:00:00"
        assert fact.weight == 0.8

    def test_create_opinion_type(self):
        fact = KeyFact(
            type="opinion",
            content="Stock is overvalued",
            source="risk_analyst",
            keywords=["overvalued"],
            timestamp="2024-12-01T10:00:00",
            weight=0.5,
        )
        assert fact.type == "opinion"

    def test_create_decision_type(self):
        fact = KeyFact(
            type="decision",
            content="Buy recommendation",
            source="trader",
            keywords=["buy"],
            timestamp="2024-12-01T10:00:00",
            weight=0.9,
        )
        assert fact.type == "decision"

    def test_weight_boundary_zero(self):
        fact = KeyFact(
            type="fact", content="test", source="src",
            keywords=[], timestamp="2024-01-01", weight=0.0,
        )
        assert fact.weight == 0.0

    def test_weight_boundary_one(self):
        fact = KeyFact(
            type="fact", content="test", source="src",
            keywords=[], timestamp="2024-01-01", weight=1.0,
        )
        assert fact.weight == 1.0

    def test_empty_keywords_list(self):
        fact = KeyFact(
            type="fact", content="test", source="src",
            keywords=[], timestamp="2024-01-01", weight=0.5,
        )
        assert fact.keywords == []


class TestAnalysisRound:
    """Tests for AnalysisRound dataclass. Requirements: 1.4"""

    def test_create_round(self):
        round_ = AnalysisRound(
            query="AAPL",
            consensus="Hold, short-term bullish",
            key_points=["Technical breakout", "Strong fundamentals"],
            timestamp="2024-12-01T10:00:00",
        )
        assert round_.query == "AAPL"
        assert round_.consensus == "Hold, short-term bullish"
        assert round_.key_points == ["Technical breakout", "Strong fundamentals"]
        assert round_.timestamp == "2024-12-01T10:00:00"

    def test_empty_key_points(self):
        round_ = AnalysisRound(
            query="TSLA", consensus="No consensus",
            key_points=[], timestamp="2024-01-01",
        )
        assert round_.key_points == []


class TestStockMemory:
    """Tests for StockMemory dataclass. Requirements: 1.1, 1.2"""

    def test_create_stock_memory(self):
        memory = StockMemory(
            stock_code="AAPL",
            summary="Historical summary",
            key_facts=[],
            recent_rounds=[],
            total_rounds=0,
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
        )
        assert memory.stock_code == "AAPL"
        assert memory.summary == "Historical summary"
        assert memory.key_facts == []
        assert memory.recent_rounds == []
        assert memory.total_rounds == 0

    def test_stock_memory_with_facts_and_rounds(self):
        fact = KeyFact(
            type="fact", content="Q3 revenue $94.9B",
            source="fundamentals_analyst", keywords=["revenue"],
            timestamp="2024-12-01T10:00:00", weight=0.8,
        )
        round_ = AnalysisRound(
            query="AAPL", consensus="Buy",
            key_points=["Strong earnings"], timestamp="2024-12-01T10:00:00",
        )
        memory = StockMemory(
            stock_code="AAPL", summary="Good stock",
            key_facts=[fact], recent_rounds=[round_],
            total_rounds=1,
            created_at="2024-01-01", updated_at="2024-12-01",
        )
        assert len(memory.key_facts) == 1
        assert len(memory.recent_rounds) == 1
        assert memory.total_rounds == 1


class TestMemoryConfig:
    """Tests for MemoryConfig dataclass. Requirements: 8.1, 8.2"""

    def test_default_values(self):
        config = MemoryConfig()
        assert config.max_recent_rounds == 3
        assert config.max_key_facts == 20
        assert config.max_summary_length == 500
        assert config.compress_threshold == 5
        assert config.memory_dir == "data/memory"
        assert config.relevance_top_n == 5

    def test_custom_values(self):
        config = MemoryConfig(
            max_recent_rounds=5,
            max_key_facts=50,
            max_summary_length=1000,
            compress_threshold=10,
            memory_dir="/custom/path",
            relevance_top_n=10,
        )
        assert config.max_recent_rounds == 5
        assert config.max_key_facts == 50
        assert config.max_summary_length == 1000
        assert config.compress_threshold == 10
        assert config.memory_dir == "/custom/path"
        assert config.relevance_top_n == 10

    def test_partial_override(self):
        config = MemoryConfig(max_recent_rounds=10, max_key_facts=30)
        assert config.max_recent_rounds == 10
        assert config.max_key_facts == 30
        # Rest should be defaults
        assert config.max_summary_length == 500
        assert config.compress_threshold == 5
        assert config.memory_dir == "data/memory"
        assert config.relevance_top_n == 5


class TestNewStockMemory:
    """Tests for new_stock_memory factory function. Requirements: 1.2"""

    def test_creates_empty_memory(self):
        memory = new_stock_memory("AAPL")
        assert memory.stock_code == "AAPL"
        assert memory.summary == ""
        assert memory.key_facts == []
        assert memory.recent_rounds == []
        assert memory.total_rounds == 0

    def test_timestamps_are_set(self):
        before = datetime.now().isoformat()
        memory = new_stock_memory("TSLA")
        after = datetime.now().isoformat()
        assert before <= memory.created_at <= after
        assert before <= memory.updated_at <= after

    def test_created_and_updated_match(self):
        memory = new_stock_memory("GOOG")
        assert memory.created_at == memory.updated_at

    def test_different_stock_codes(self):
        m1 = new_stock_memory("AAPL")
        m2 = new_stock_memory("TSLA")
        assert m1.stock_code == "AAPL"
        assert m2.stock_code == "TSLA"
