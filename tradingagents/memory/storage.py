"""Persistence layer for the Stock Memory System.

Handles JSON file read/write for StockMemory objects.
Each stock gets its own file at {memory_dir}/{stock_code}.json.

Requirements: 4.1, 4.3, 4.4, 4.5
"""

import json
import logging
import os
from dataclasses import asdict

from tradingagents.memory.types import (
    AnalysisRound,
    KeyFact,
    StockMemory,
)

logger = logging.getLogger(__name__)


class MemoryStorage:
    """Handles serialization and file I/O for StockMemory objects."""

    def __init__(self, base_dir: str = "data/memory"):
        """Initialize with the base directory for memory files.

        Args:
            base_dir: Directory path where memory JSON files are stored.
        """
        self.base_dir = base_dir

    def _file_path(self, stock_code: str) -> str:
        """Return the JSON file path for a given stock code."""
        return os.path.join(self.base_dir, f"{stock_code}.json")

    def load(self, stock_code: str) -> StockMemory | None:
        """Load a StockMemory from its JSON file.

        Args:
            stock_code: The stock ticker symbol.

        Returns:
            The deserialized StockMemory, or None if the file doesn't
            exist or contains invalid JSON.
        """
        path = self._file_path(stock_code)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                json_str = f.read()
            return self.deserialize(json_str)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Corrupted memory file for %s: %s", stock_code, e)
            return None

    def save(self, memory: StockMemory) -> None:
        """Serialize a StockMemory and write it to its JSON file.

        Creates the directory if it doesn't exist.

        Args:
            memory: The StockMemory object to persist.
        """
        os.makedirs(self.base_dir, exist_ok=True)
        path = self._file_path(memory.stock_code)
        json_str = self.serialize(memory)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_str)

    def serialize(self, memory: StockMemory) -> str:
        """Convert a StockMemory to a JSON string.

        Args:
            memory: The StockMemory object to serialize.

        Returns:
            A JSON string representation of the memory.
        """
        return json.dumps(asdict(memory), ensure_ascii=False, indent=2)

    def deserialize(self, json_str: str) -> StockMemory:
        """Reconstruct a StockMemory from a JSON string.

        Args:
            json_str: A JSON string previously produced by serialize().

        Returns:
            The reconstructed StockMemory object with nested dataclasses.

        Raises:
            json.JSONDecodeError: If the string is not valid JSON.
            KeyError: If required fields are missing.
        """
        data = json.loads(json_str)
        key_facts = [KeyFact(**kf) for kf in data["key_facts"]]
        recent_rounds = [AnalysisRound(**rr) for rr in data["recent_rounds"]]
        return StockMemory(
            stock_code=data["stock_code"],
            summary=data["summary"],
            key_facts=key_facts,
            recent_rounds=recent_rounds,
            total_rounds=data["total_rounds"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )
