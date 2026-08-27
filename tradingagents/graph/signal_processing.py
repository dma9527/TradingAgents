# TradingAgents/graph/signal_processing.py

import re

from langchain_openai import ChatOpenAI

# The trader's final text nearly always states the decision explicitly, most
# often as "FINAL TRANSACTION PROPOSAL: **BUY**". Matching that structurally
# avoids one LLM call per analysis whose entire output is a single word.
_DECISIONS = ("BUY", "SELL", "HOLD")

_EXPLICIT_PATTERNS = (
    re.compile(
        r"final\s+transaction\s+proposal\s*:?\s*\**\s*(BUY|SELL|HOLD)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"final\s+(?:trade\s+)?(?:decision|recommendation)\s*:?\s*\**\s*"
        r"(BUY|SELL|HOLD)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\*\*\s*(BUY|SELL|HOLD)\s*\*\*",
    ),
)


def extract_decision(full_signal: str) -> str | None:
    """Extract BUY/SELL/HOLD from *full_signal* without an LLM call.

    Returns None when the text is missing, or when it mentions more than one
    distinct decision with no explicit "final proposal" marker to disambiguate.
    The caller then falls back to the LLM.
    """
    if not full_signal or not isinstance(full_signal, str):
        return None

    for pattern in _EXPLICIT_PATTERNS:
        match = pattern.search(full_signal)
        if match:
            return match.group(1).upper()

    # No explicit marker: accept a bare mention only when it is unambiguous.
    mentioned = {
        word
        for word in _DECISIONS
        if re.search(rf"\b{word}\b", full_signal, re.IGNORECASE)
    }
    if len(mentioned) == 1:
        return mentioned.pop()
    return None


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """Initialize with an LLM for processing."""
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """
        Process a full trading signal to extract the core decision.

        Tries structural extraction first and only calls the LLM when the text
        is ambiguous, which removes one LLM round trip from the common path.

        Args:
            full_signal: Complete trading signal text

        Returns:
            Extracted decision (BUY, SELL, or HOLD)
        """
        extracted = extract_decision(full_signal)
        if extracted:
            return extracted

        if not full_signal:
            return "HOLD"

        messages = [
            (
                "system",
                "You are an efficient assistant designed to analyze paragraphs or financial reports provided by a group of analysts. Your task is to extract the investment decision: SELL, BUY, or HOLD. Provide only the extracted decision (SELL, BUY, or HOLD) as your output, without adding any additional text or information.",
            ),
            ("human", full_signal),
        ]

        response = self.quick_thinking_llm.invoke(messages).content
        return extract_decision(response) or "HOLD"
