"""Tests for MemoryStorage persistence layer.

Covers:
- File not found returns None
- Corrupted JSON returns None with warning
- Normal save/load roundtrip
- Directory auto-creation
- Serialize/deserialize correctness

Requirements: 4.1, 4.3, 4.4, 4.5
"""

import json
import logging
import os

import pytest

from tradingagents.memory.storage import MemoryStorage
from tradingagents.memory.types import (
    AnalysisRound,
    KeyFact,
    StockMemory,
    new_stock_memory,
)


@pytest.fixture
def storage(tmp_path: str) -> MemoryStorage:
    """Create a MemoryStorage backed by a temporary directory."""
    return MemoryStorage(base_dir=str(tmp_path))


def _sample_memory() -> StockMemory:
    """Build a StockMemory with one fact and one round for testing."""
    return StockMemory(
        stock_code="AAPL",
        summary="Historical summary",
        key_facts=[
            KeyFact(
                type="fact",
                content="Q3 revenue $94.9B",
                source="fundamentals_analyst",
                keywords=["revenue", "Q3"],
                timestamp="2024-12-01T10:00:00",
                weight=0.8,
            ),
        ],
        recent_rounds=[
            AnalysisRound(
                query="AAPL",
                consensus="Hold, short-term bullish",
                key_points=["Technical breakout", "Strong fundamentals"],
                timestamp="2024-12-01T10:00:00",
            ),
        ],
        total_rounds=1,
        created_at="2024-11-01T08:00:00",
        updated_at="2024-12-01T10:00:00",
    )


class TestLoad:
    """Tests for MemoryStorage.load. Requirements: 4.4"""

    def test_file_not_found_returns_none(self, storage: MemoryStorage):
        result = storage.load("NONEXISTENT")
        assert result is None

    def test_corrupted_json_returns_none(
        self, storage: MemoryStorage, caplog: pytest.LogCaptureFixture
    ):
        os.makedirs(storage.base_dir, exist_ok=True)
        path = storage._file_path("BAD")
        with open(path, "w") as f:
            f.write("{invalid json content!!!")

        with caplog.at_level(logging.WARNING):
            result = storage.load("BAD")

        assert result is None
        assert "Corrupted memory file" in caplog.text

    def test_missing_fields_returns_none(
        self, storage: MemoryStorage, caplog: pytest.LogCaptureFixture
    ):
        os.makedirs(storage.base_dir, exist_ok=True)
        path = storage._file_path("PARTIAL")
        with open(path, "w") as f:
            json.dump({"stock_code": "PARTIAL"}, f)

        with caplog.at_level(logging.WARNING):
            result = storage.load("PARTIAL")

        assert result is None
        assert "Corrupted memory file" in caplog.text


class TestSave:
    """Tests for MemoryStorage.save. Requirements: 4.1"""

    def test_creates_directory_if_missing(self, tmp_path: str):
        nested = os.path.join(str(tmp_path), "a", "b", "c")
        storage = MemoryStorage(base_dir=nested)
        memory = new_stock_memory("TSLA")

        storage.save(memory)

        assert os.path.isdir(nested)
        assert os.path.isfile(os.path.join(nested, "TSLA.json"))

    def test_file_path_matches_stock_code(self, storage: MemoryStorage):
        memory = new_stock_memory("GOOG")
        storage.save(memory)

        expected = os.path.join(storage.base_dir, "GOOG.json")
        assert os.path.isfile(expected)

    def test_saved_file_is_valid_json(self, storage: MemoryStorage):
        memory = _sample_memory()
        storage.save(memory)

        path = storage._file_path("AAPL")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["stock_code"] == "AAPL"
        assert len(data["key_facts"]) == 1
        assert len(data["recent_rounds"]) == 1


class TestRoundtrip:
    """Tests for save → load roundtrip. Requirements: 4.3, 4.5"""

    def test_empty_memory_roundtrip(self, storage: MemoryStorage):
        original = new_stock_memory("MSFT")
        storage.save(original)
        loaded = storage.load("MSFT")

        assert loaded is not None
        assert loaded.stock_code == original.stock_code
        assert loaded.summary == original.summary
        assert loaded.key_facts == original.key_facts
        assert loaded.recent_rounds == original.recent_rounds
        assert loaded.total_rounds == original.total_rounds
        assert loaded.created_at == original.created_at
        assert loaded.updated_at == original.updated_at

    def test_populated_memory_roundtrip(self, storage: MemoryStorage):
        original = _sample_memory()
        storage.save(original)
        loaded = storage.load("AAPL")

        assert loaded is not None
        assert loaded.stock_code == original.stock_code
        assert loaded.summary == original.summary
        assert loaded.total_rounds == original.total_rounds

        assert len(loaded.key_facts) == 1
        fact = loaded.key_facts[0]
        assert fact.type == "fact"
        assert fact.content == "Q3 revenue $94.9B"
        assert fact.weight == 0.8
        assert fact.keywords == ["revenue", "Q3"]

        assert len(loaded.recent_rounds) == 1
        rnd = loaded.recent_rounds[0]
        assert rnd.query == "AAPL"
        assert rnd.consensus == "Hold, short-term bullish"
        assert rnd.key_points == ["Technical breakout", "Strong fundamentals"]

    def test_chinese_content_roundtrip(self, storage: MemoryStorage):
        memory = new_stock_memory("600519")
        memory.summary = "贵州茅台历史分析摘要"
        memory.key_facts.append(
            KeyFact(
                type="fact",
                content="2024年Q3营收达到949亿元",
                source="基本面分析师",
                keywords=["营收", "Q3", "949亿"],
                timestamp="2024-12-01T10:00:00",
                weight=0.9,
            )
        )
        storage.save(memory)
        loaded = storage.load("600519")

        assert loaded is not None
        assert loaded.summary == "贵州茅台历史分析摘要"
        assert loaded.key_facts[0].content == "2024年Q3营收达到949亿元"


class TestSerializeDeserialize:
    """Tests for serialize/deserialize methods directly. Requirements: 4.3, 4.5"""

    def test_serialize_returns_valid_json(self, storage: MemoryStorage):
        memory = _sample_memory()
        json_str = storage.serialize(memory)
        data = json.loads(json_str)
        assert data["stock_code"] == "AAPL"

    def test_deserialize_reconstructs_nested_types(self, storage: MemoryStorage):
        memory = _sample_memory()
        json_str = storage.serialize(memory)
        restored = storage.deserialize(json_str)

        assert isinstance(restored, StockMemory)
        assert isinstance(restored.key_facts[0], KeyFact)
        assert isinstance(restored.recent_rounds[0], AnalysisRound)

    def test_roundtrip_via_serialize_deserialize(self, storage: MemoryStorage):
        original = _sample_memory()
        json_str = storage.serialize(original)
        restored = storage.deserialize(json_str)

        assert restored == original

    def test_deserialize_invalid_json_raises(self, storage: MemoryStorage):
        with pytest.raises(json.JSONDecodeError):
            storage.deserialize("not json")

    def test_deserialize_missing_key_raises(self, storage: MemoryStorage):
        with pytest.raises(KeyError):
            storage.deserialize('{"stock_code": "X"}')
