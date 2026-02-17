"""BM25-based relevance matching for key facts.

Uses rank_bm25 to index KeyFact content and retrieve the most relevant
facts for a given query. Supports Chinese + English mixed tokenization.

Requirements: 5.1, 5.2, 5.3, 5.4
"""

import re

from rank_bm25 import BM25Okapi

from tradingagents.memory.types import KeyFact


class RelevanceMatcher:
    """Match key facts to a query using BM25 relevance scoring."""

    def match(
        self, facts: list[KeyFact], query: str, top_n: int = 5
    ) -> list[KeyFact]:
        """Return the top_n most relevant facts for the query, ordered by BM25 score descending.

        Args:
            facts: List of KeyFact instances to search through.
            query: The query string to match against.
            top_n: Maximum number of results to return.

        Returns:
            List of KeyFact instances sorted by BM25 relevance (highest first).
            Returns empty list if facts is empty.
        """
        if not facts:
            return []

        tokenized_corpus = [self._tokenize(fact.content) for fact in facts]
        bm25 = BM25Okapi(tokenized_corpus)

        query_tokens = self._tokenize(query)
        scores = bm25.get_scores(query_tokens)

        scored_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_n]

        return [facts[i] for i in scored_indices]

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text supporting Chinese characters and English words.

        Chinese characters are split individually; English/numeric words are
        kept as whole tokens. Everything is lowercased.

        Args:
            text: Input text (may contain Chinese, English, or mixed).

        Returns:
            List of token strings.
        """
        tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text.lower())
        return tokens
