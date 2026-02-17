"""Tests for MemoryManager core logic.

Covers: get_or_create, add_round, add_facts, build_context, save,
compression triggering, deduplication, and capacity enforcement.
"""

import os
import tempfile

import pytest

from tradingagents.memory.manager import MemoryManager
from tradingagents.memory.types import KeyFact, MemoryConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def manager(tmp_dir):
    cfg = MemoryConfig(
        max_recent_rounds=2,
        max_key_facts=5,
        max_summary_length=200,
        compress_threshold=4,
        memory_dir=tmp_dir,
        relevance_top_n=3,
    )
    return MemoryManager(config=cfg)


def _make_fact(content: str, weight: float = 0.5, source: str = "test") -> KeyFact:
    return KeyFact(
        type="fact",
        content=content,
        source=source,
        keywords=[],
        timestamp="2024-01-01T00:00:00",
        weight=weight,
    )


# ---------------------------------------------------------------------------
# get_or_create
# ---------------------------------------------------------------------------

class TestGetOrCreate:
    def test_creates_new_memory(self, manager):
        mem = manager.get_or_create("AAPL")
        assert mem.stock_code == "AAPL"
        assert mem.total_rounds == 0
        assert mem.key_facts == []
        assert mem.recent_rounds == []

    def test_returns_cached_instance(self, manager):
        mem1 = manager.get_or_create("AAPL")
        mem2 = manager.get_or_create("AAPL")
        assert mem1 is mem2

    def test_different_stocks_different_instances(self, manager):
        a = manager.get_or_create("AAPL")
        b = manager.get_or_create("GOOG")
        assert a is not b
        assert a.stock_code == "AAPL"
        assert b.stock_code == "GOOG"

    def test_loads_from_disk(self, manager):
        manager.get_or_create("TSLA")
        manager.add_round("TSLA", "TSLA", "buy", ["strong"])
        manager.save("TSLA")

        # New manager, same dir — should load from file
        mgr2 = MemoryManager(config=manager.config)
        mem = mgr2.get_or_create("TSLA")
        assert mem.total_rounds == 1
        assert mem.recent_rounds[0].consensus == "buy"


# ---------------------------------------------------------------------------
# add_round
# ---------------------------------------------------------------------------

class TestAddRound:
    def test_increments_total_rounds(self, manager):
        manager.get_or_create("X")
        manager.add_round("X", "X", "hold", [])
        assert manager.get_or_create("X").total_rounds == 1
        manager.add_round("X", "X", "sell", [])
        assert manager.get_or_create("X").total_rounds == 2

    def test_appends_to_recent_rounds(self, manager):
        manager.get_or_create("X")
        manager.add_round("X", "X", "buy", ["p1"])
        mem = manager.get_or_create("X")
        assert len(mem.recent_rounds) == 1
        assert mem.recent_rounds[0].consensus == "buy"

    def test_compression_triggered_at_threshold(self, manager):
        """With compress_threshold=4 and max_recent_rounds=2,
        adding 4 rounds should trigger compression leaving 2."""
        manager.get_or_create("Z")
        for i in range(4):
            manager.add_round("Z", "Z", f"c{i}", [f"p{i}"])
        mem = manager.get_or_create("Z")
        assert len(mem.recent_rounds) <= manager.config.max_recent_rounds
        assert mem.total_rounds == 4


# ---------------------------------------------------------------------------
# add_facts
# ---------------------------------------------------------------------------

class TestAddFacts:
    def test_adds_facts(self, manager):
        manager.get_or_create("A")
        manager.add_facts("A", [_make_fact("revenue up")])
        assert len(manager.get_or_create("A").key_facts) == 1

    def test_deduplicates_by_content(self, manager):
        manager.get_or_create("A")
        f1 = _make_fact("revenue up")
        f2 = _make_fact("revenue up")
        manager.add_facts("A", [f1, f2])
        assert len(manager.get_or_create("A").key_facts) == 1

    def test_dedup_across_calls(self, manager):
        manager.get_or_create("A")
        manager.add_facts("A", [_make_fact("revenue up")])
        manager.add_facts("A", [_make_fact("revenue up")])
        assert len(manager.get_or_create("A").key_facts) == 1

    def test_enforces_max_key_facts(self, manager):
        """max_key_facts=5, adding 7 should keep only 5 highest-weight."""
        manager.get_or_create("B")
        facts = [_make_fact(f"fact{i}", weight=i * 0.1) for i in range(7)]
        manager.add_facts("B", facts)
        mem = manager.get_or_create("B")
        assert len(mem.key_facts) == 5
        # Lowest weight facts should have been removed
        weights = [f.weight for f in mem.key_facts]
        assert min(weights) >= 0.2  # 0.0 and 0.1 removed

    def test_keeps_highest_weight(self, manager):
        manager.get_or_create("C")
        facts = [
            _make_fact("low", weight=0.1),
            _make_fact("mid", weight=0.5),
            _make_fact("high", weight=0.9),
            _make_fact("very_high", weight=1.0),
            _make_fact("medium", weight=0.6),
            _make_fact("extra", weight=0.8),
        ]
        manager.add_facts("C", facts)
        mem = manager.get_or_create("C")
        contents = {f.content for f in mem.key_facts}
        assert "low" not in contents  # weight 0.1 removed


# ---------------------------------------------------------------------------
# build_context
# ---------------------------------------------------------------------------

class TestBuildContext:
    def test_empty_memory_returns_empty_string(self, manager):
        manager.get_or_create("E")
        assert manager.build_context("E", "E") == ""

    def test_includes_summary(self, manager):
        mem = manager.get_or_create("F")
        mem.summary = "Historical summary text"
        ctx = manager.build_context("F", "F")
        assert "Historical summary text" in ctx
        assert "历史摘要" in ctx

    def test_includes_facts(self, manager):
        manager.get_or_create("G")
        manager.add_facts("G", [_make_fact("Q3 revenue 94B", source="analyst")])
        ctx = manager.build_context("G", "revenue")
        assert "Q3 revenue 94B" in ctx
        assert "相关历史事实" in ctx

    def test_includes_recent_rounds(self, manager):
        manager.get_or_create("H")
        manager.add_round("H", "H", "bullish outlook", ["strong earnings"])
        ctx = manager.build_context("H", "H")
        assert "bullish outlook" in ctx
        assert "近期分析" in ctx

    def test_header_contains_stock_code(self, manager):
        manager.get_or_create("MSFT")
        manager.add_round("MSFT", "MSFT", "hold", [])
        ctx = manager.build_context("MSFT", "MSFT")
        assert "MSFT" in ctx


# ---------------------------------------------------------------------------
# save / persistence
# ---------------------------------------------------------------------------

class TestSave:
    def test_save_creates_file(self, manager, tmp_dir):
        manager.get_or_create("SV")
        manager.add_round("SV", "SV", "buy", [])
        manager.save("SV")
        assert os.path.exists(os.path.join(tmp_dir, "SV.json"))

    def test_save_noop_for_unknown_stock(self, manager):
        # Should not raise
        manager.save("UNKNOWN")


# ---------------------------------------------------------------------------
# save_async
# ---------------------------------------------------------------------------

class TestSaveAsync:
    @pytest.mark.asyncio
    async def test_async_save_creates_file(self, manager, tmp_dir):
        manager.get_or_create("AS")
        manager.add_round("AS", "AS", "sell", [])
        await manager.save_async("AS")
        assert os.path.exists(os.path.join(tmp_dir, "AS.json"))
