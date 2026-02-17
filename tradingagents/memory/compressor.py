"""Memory compressor for the Stock Memory System.

Uses an LLM client to compress old analysis rounds into a summary.
Falls back to simple truncation when no LLM client is available.
On LLM failure, returns the existing summary unchanged and logs an error.

Requirements: 6.1, 6.2, 6.3, 6.4
"""

import logging

from tradingagents.memory.types import AnalysisRound

logger = logging.getLogger(__name__)


class MemoryCompressor:
    """Compresses old analysis rounds into a text summary using an LLM.

    The LLM client should be a callable or an object with an ``invoke`` method
    (LangChain style).  When no client is provided the compressor falls back to
    a simple text-truncation strategy.

    Attributes:
        llm_client: Optional LLM client used for summarisation.
    """

    def __init__(self, llm_client=None):  # noqa: ANN001
        """Initialise the compressor.

        Args:
            llm_client: A callable or object with an ``invoke(prompt)`` method.
                        If *None*, the compressor uses simple truncation.
        """
        self.llm_client = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(
        self,
        rounds: list[AnalysisRound],
        existing_summary: str,
        max_length: int = 500,
    ) -> str:
        """Compress *rounds* into a summary and append to *existing_summary*.

        Steps:
        1. Format the rounds into readable text.
        2. Ask the LLM to summarise (or fall back to truncation).
        3. Merge the new summary with the existing one.
        4. Truncate the result to *max_length* characters.

        If the LLM call fails the method returns *existing_summary* unchanged
        and logs an ``ERROR``.

        Args:
            rounds: Analysis rounds to compress.
            existing_summary: The current summary text to append to.
            max_length: Maximum allowed length of the returned summary.

        Returns:
            The merged summary string, guaranteed to be at most *max_length*
            characters long.
        """
        if not rounds:
            return existing_summary[:max_length]

        rounds_text = self._format_rounds(rounds)

        if self.llm_client is None:
            new_summary = self._fallback_compress(rounds)
            return self._merge_and_truncate(existing_summary, new_summary, max_length)

        prompt = self._build_prompt(rounds_text, existing_summary)

        try:
            new_summary = self._call_llm(prompt)
        except Exception:
            logger.error(
                "LLM compression failed, returning existing summary unchanged",
                exc_info=True,
            )
            return existing_summary

        return self._merge_and_truncate(existing_summary, new_summary, max_length)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_rounds(self, rounds: list[AnalysisRound]) -> str:
        """Format analysis rounds into human-readable text."""
        parts: list[str] = []
        for i, r in enumerate(rounds, 1):
            key_points_str = "、".join(r.key_points) if r.key_points else "无"
            parts.append(
                f"第{i}轮 ({r.timestamp}):\n"
                f"  查询: {r.query}\n"
                f"  结论: {r.consensus}\n"
                f"  要点: {key_points_str}"
            )
        return "\n\n".join(parts)

    def _build_prompt(self, rounds_text: str, existing_summary: str) -> str:
        """Build the LLM prompt for summarisation."""
        context = ""
        if existing_summary:
            context = f"已有摘要:\n{existing_summary}\n\n"

        return (
            "你是一个股票分析记忆压缩助手。请将以下历史分析轮次压缩为简洁的中文摘要，"
            "保留关键结论和重要事实。摘要应尽量简短。\n\n"
            f"{context}"
            f"需要压缩的分析轮次:\n{rounds_text}\n\n"
            "请输出压缩后的摘要:"
        )

    def _call_llm(self, prompt: str) -> str:
        """Invoke the LLM client and return the response text."""
        if hasattr(self.llm_client, "invoke"):
            response = self.llm_client.invoke(prompt)
            # LangChain-style: response may be an object with .content
            if hasattr(response, "content"):
                return str(response.content)
            return str(response)
        # Treat as a plain callable
        return str(self.llm_client(prompt))

    def _fallback_compress(self, rounds: list[AnalysisRound]) -> str:
        """Simple truncation fallback when no LLM is available."""
        parts: list[str] = []
        for r in rounds:
            points = "、".join(r.key_points) if r.key_points else ""
            parts.append(f"{r.query}: {r.consensus}。{points}")
        return "; ".join(parts)

    def _merge_and_truncate(
        self, existing: str, new_text: str, max_length: int
    ) -> str:
        """Merge *new_text* into *existing* and truncate to *max_length*."""
        if not existing:
            return new_text[:max_length]
        if not new_text:
            return existing[:max_length]
        merged = f"{existing}; {new_text}"
        return merged[:max_length]
