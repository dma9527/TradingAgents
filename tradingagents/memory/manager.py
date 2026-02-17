"""Core MemoryManager for the Stock Memory System.

Orchestrates memory lifecycle: load/create, add rounds, add facts,
build context, save. This is the single entry point for external callers.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.4, 4.2
"""

import asyncio
import logging
from datetime import datetime

from tradingagents.memory.compressor import MemoryCompressor
from tradingagents.memory.relevance import RelevanceMatcher
from tradingagents.memory.storage import MemoryStorage
from tradingagents.memory.types import (
    AnalysisRound,
    KeyFact,
    MemoryConfig,
    StockMemory,
    new_stock_memory,
)

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages per-stock memory lifecycle.

    Provides get_or_create, add_round, add_facts, build_context, save,
    and save_async. All operations are safe — failures are logged but
    never propagate to callers.
    """

    def __init__(self, config: MemoryConfig | None = None, llm_client=None):  # noqa: ANN001
        """Initialise the manager.

        Args:
            config: Optional configuration. Uses defaults when *None*.
            llm_client: Optional LLM client for memory compression.
        """
        self.config = config or MemoryConfig()
        self.storage = MemoryStorage(base_dir=self.config.memory_dir)
        self.relevance = RelevanceMatcher()
        self.compressor = MemoryCompressor(llm_client=llm_client)
        self._cache: dict[str, StockMemory] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_create(self, stock_code: str) -> StockMemory:
        """Load existing memory or create a new empty one.

        Results are cached in-memory so repeated calls for the same
        stock_code return the same instance.

        Args:
            stock_code: The stock ticker symbol.

        Returns:
            The StockMemory for this stock.
        """
        if stock_code in self._cache:
            return self._cache[stock_code]

        mem = self.storage.load(stock_code)
        if mem is None:
            logger.info("Creating new memory for %s", stock_code)
            mem = new_stock_memory(stock_code)

        self._cache[stock_code] = mem
        return mem


    def add_round(
        self,
        stock_code: str,
        query: str,
        consensus: str,
        key_points: list[str],
    ) -> None:
        """Add an analysis round and trigger compression if needed.

        Args:
            stock_code: The stock ticker symbol.
            query: The query or ticker that was analysed.
            consensus: The consensus conclusion.
            key_points: Key points from the analysis.
        """
        mem = self.get_or_create(stock_code)
        mem.total_rounds += 1
        mem.recent_rounds.append(
            AnalysisRound(
                query=query,
                consensus=consensus,
                key_points=key_points,
                timestamp=datetime.now().isoformat(),
            )
        )
        mem.updated_at = datetime.now().isoformat()

        # Compress if threshold exceeded
        if len(mem.recent_rounds) >= self.config.compress_threshold:
            self._compress(mem)

    def add_facts(self, stock_code: str, facts: list[KeyFact]) -> None:
        """Add key facts with deduplication and capacity enforcement.

        Duplicate facts (same content) are skipped. When the total exceeds
        max_key_facts, the lowest-weight facts are removed.

        Args:
            stock_code: The stock ticker symbol.
            facts: New facts to add.
        """
        mem = self.get_or_create(stock_code)
        existing_contents = {f.content for f in mem.key_facts}

        for fact in facts:
            if fact.content not in existing_contents:
                mem.key_facts.append(fact)
                existing_contents.add(fact.content)

        # Enforce capacity — keep highest-weight facts
        if len(mem.key_facts) > self.config.max_key_facts:
            mem.key_facts.sort(key=lambda f: f.weight, reverse=True)
            mem.key_facts = mem.key_facts[: self.config.max_key_facts]

        mem.updated_at = datetime.now().isoformat()

    def build_context(self, stock_code: str, query: str) -> str:
        """Build a formatted context string from the stock's memory.

        Returns an empty string when the memory is completely empty
        (no summary, no facts, no rounds).

        Args:
            stock_code: The stock ticker symbol.
            query: Current query for relevance matching.

        Returns:
            Formatted context string ready for prompt injection.
        """
        mem = self.get_or_create(stock_code)

        has_summary = bool(mem.summary)
        has_facts = bool(mem.key_facts)
        has_rounds = bool(mem.recent_rounds)

        if not has_summary and not has_facts and not has_rounds:
            return ""

        parts: list[str] = [f"=== 股票历史记忆: {stock_code} ==="]

        if has_summary:
            parts.append(f"\n【历史摘要】\n{mem.summary}")

        if has_facts:
            relevant = self.relevance.match(
                mem.key_facts, query, top_n=self.config.relevance_top_n
            )
            if relevant:
                lines = [
                    f"- [{f.type}] {f.content} (来源: {f.source})"
                    for f in relevant
                ]
                parts.append("\n【相关历史事实】\n" + "\n".join(lines))

        if has_rounds:
            round_lines: list[str] = []
            for r in mem.recent_rounds:
                points = "、".join(r.key_points) if r.key_points else "无"
                round_lines.append(
                    f"第 {mem.total_rounds - len(mem.recent_rounds) + mem.recent_rounds.index(r) + 1} 轮 ({r.timestamp}):\n"
                    f"  结论: {r.consensus}\n"
                    f"  要点: {points}"
                )
            parts.append("\n【近期分析】\n" + "\n".join(round_lines))

        return "\n".join(parts)

    def save(self, stock_code: str) -> None:
        """Synchronously save the stock's memory to disk.

        Args:
            stock_code: The stock ticker symbol.
        """
        mem = self._cache.get(stock_code)
        if mem is None:
            return
        self.storage.save(mem)

    async def save_async(self, stock_code: str) -> None:
        """Asynchronously save the stock's memory to disk.

        Runs the blocking file I/O in a thread executor so it doesn't
        block the event loop.

        Args:
            stock_code: The stock ticker symbol.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.save, stock_code)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compress(self, mem: StockMemory) -> None:
        """Compress old rounds into the summary, keeping only recent ones."""
        keep = self.config.max_recent_rounds
        if len(mem.recent_rounds) <= keep:
            return

        to_compress = mem.recent_rounds[:-keep] if keep > 0 else mem.recent_rounds
        to_keep = mem.recent_rounds[-keep:] if keep > 0 else []

        mem.summary = self.compressor.compress(
            to_compress,
            mem.summary,
            max_length=self.config.max_summary_length,
        )
        mem.recent_rounds = to_keep
