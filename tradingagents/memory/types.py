"""Data types for the Stock Memory System.

Defines the core data structures: KeyFact, AnalysisRound, StockMemory, and MemoryConfig.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class KeyFact:
    """A key fact extracted from stock analysis.

    Attributes:
        type: Category of the fact - "fact", "opinion", or "decision".
        content: The actual content of the fact.
        source: Which analyst or agent produced this fact.
        keywords: Keywords for relevance matching.
        timestamp: ISO format timestamp of when the fact was recorded.
        weight: Importance weight between 0.0 and 1.0.
    """

    type: Literal["fact", "opinion", "decision"]
    content: str
    source: str
    keywords: list[str]
    timestamp: str
    weight: float  # 0.0 ~ 1.0


@dataclass
class AnalysisRound:
    """A single round of stock analysis.

    Attributes:
        query: The query or stock code that was analyzed.
        consensus: The consensus conclusion from the analysis.
        key_points: List of key points from the analysis.
        timestamp: ISO format timestamp of when the round occurred.
    """

    query: str
    consensus: str
    key_points: list[str]
    timestamp: str


@dataclass
class StockMemory:
    """Complete memory space for a single stock.

    Attributes:
        stock_code: The stock ticker symbol.
        summary: Compressed historical analysis summary.
        key_facts: List of key facts extracted from past analyses.
        recent_rounds: Recent analysis rounds (kept within threshold).
        total_rounds: Total number of analysis rounds performed.
        created_at: ISO format timestamp of when this memory was created.
        updated_at: ISO format timestamp of the last update.
    """

    stock_code: str
    summary: str
    key_facts: list[KeyFact]
    recent_rounds: list[AnalysisRound]
    total_rounds: int
    created_at: str
    updated_at: str


@dataclass
class MemoryConfig:
    """Configuration for the memory system.

    Attributes:
        max_recent_rounds: Maximum number of recent rounds to keep.
        max_key_facts: Maximum number of key facts to store per stock.
        max_summary_length: Maximum character length for the summary.
        compress_threshold: Number of rounds that triggers compression.
        memory_dir: Directory path for storing memory JSON files.
        relevance_top_n: Number of top relevant facts to retrieve.
    """

    max_recent_rounds: int = 3
    max_key_facts: int = 20
    max_summary_length: int = 500
    compress_threshold: int = 5
    memory_dir: str = "data/memory"
    relevance_top_n: int = 5


def new_stock_memory(stock_code: str) -> StockMemory:
    """Create a new empty StockMemory for the given stock code.

    Args:
        stock_code: The stock ticker symbol.

    Returns:
        A fresh StockMemory with empty summary, no facts, no rounds, and
        created_at/updated_at set to the current time.
    """
    now = datetime.now().isoformat()
    return StockMemory(
        stock_code=stock_code,
        summary="",
        key_facts=[],
        recent_rounds=[],
        total_rounds=0,
        created_at=now,
        updated_at=now,
    )
