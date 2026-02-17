"""Tests for RelevanceMatcher BM25-based relevance matching.

Covers:
- Empty facts list returns empty
- Single fact matching
- Chinese + English mixed query
- Top-N limiting
- Requirements: 5.1, 5.2, 5.3, 5.4
"""

from tradingagents.memory.relevance import RelevanceMatcher
from tradingagents.memory.types import KeyFact


def _make_fact(content: str, **kwargs) -> KeyFact:
    """Helper to create a KeyFact with sensible defaults."""
    defaults = {
        "type": "fact",
        "source": "test",
        "keywords": [],
        "timestamp": "2024-01-01T00:00:00",
        "weight": 0.5,
    }
    defaults.update(kwargs)
    return KeyFact(content=content, **defaults)


class TestRelevanceMatcherTokenize:
    """Tests for _tokenize method. Requirements: 5.4"""

    def setup_method(self):
        self.matcher = RelevanceMatcher()

    def test_english_words(self):
        tokens = self.matcher._tokenize("Revenue grew 10 percent")
        assert tokens == ["revenue", "grew", "10", "percent"]

    def test_chinese_characters(self):
        tokens = self.matcher._tokenize("营收增长")
        assert tokens == ["营", "收", "增", "长"]

    def test_mixed_chinese_english(self):
        tokens = self.matcher._tokenize("AAPL的营收grew by 10%")
        assert "aapl" in tokens
        assert "grew" in tokens
        assert "by" in tokens
        assert "10" in tokens
        assert "营" in tokens
        assert "收" in tokens

    def test_empty_string(self):
        tokens = self.matcher._tokenize("")
        assert tokens == []

    def test_lowercases_english(self):
        tokens = self.matcher._tokenize("Apple REVENUE Growth")
        assert tokens == ["apple", "revenue", "growth"]

    def test_strips_punctuation(self):
        tokens = self.matcher._tokenize("hello, world! test.")
        assert tokens == ["hello", "world", "test"]


class TestRelevanceMatcherMatch:
    """Tests for match method. Requirements: 5.1, 5.2, 5.3"""

    def setup_method(self):
        self.matcher = RelevanceMatcher()

    def test_empty_facts_returns_empty(self):
        result = self.matcher.match([], "some query")
        assert result == []

    def test_single_fact_returned(self):
        facts = [_make_fact("Revenue grew by 10 percent last quarter")]
        result = self.matcher.match(facts, "revenue growth")
        assert len(result) == 1
        assert result[0].content == "Revenue grew by 10 percent last quarter"

    def test_top_n_limits_results(self):
        facts = [
            _make_fact("Apple revenue increased significantly"),
            _make_fact("Tesla stock price dropped sharply"),
            _make_fact("Google cloud revenue grew fast"),
            _make_fact("Amazon profit margins improved"),
            _make_fact("Microsoft Azure growth accelerated"),
        ]
        result = self.matcher.match(facts, "revenue growth", top_n=2)
        assert len(result) == 2

    def test_top_n_larger_than_facts(self):
        facts = [
            _make_fact("Revenue grew"),
            _make_fact("Stock dropped"),
        ]
        result = self.matcher.match(facts, "revenue", top_n=10)
        assert len(result) == 2

    def test_relevance_ordering(self):
        facts = [
            _make_fact("Tesla electric vehicle production numbers"),
            _make_fact("Apple revenue grew by 10 percent in Q3"),
            _make_fact("Apple iPhone revenue reached new highs"),
        ]
        result = self.matcher.match(facts, "Apple revenue", top_n=3)
        # The two Apple-revenue facts should rank higher than Tesla
        top_contents = [f.content for f in result[:2]]
        assert any("Apple" in c and "revenue" in c for c in top_contents)

    def test_chinese_query_matching(self):
        facts = [
            _make_fact("苹果公司营收增长百分之十"),
            _make_fact("特斯拉股价大幅下跌"),
            _make_fact("谷歌云计算业务快速发展"),
        ]
        result = self.matcher.match(facts, "营收增长", top_n=2)
        assert len(result) == 2
        # The fact about 营收增长 should be ranked first
        assert "营收增长" in result[0].content

    def test_mixed_chinese_english_query(self):
        facts = [
            _make_fact("AAPL的营收在Q3增长了10%"),
            _make_fact("TSLA电动车产量创新高"),
            _make_fact("GOOG云计算收入快速增长"),
        ]
        result = self.matcher.match(facts, "AAPL营收增长", top_n=2)
        assert len(result) == 2
        # AAPL fact should rank first since it matches both AAPL and 营收增长
        assert "AAPL" in result[0].content

    def test_returns_keyfact_instances(self):
        facts = [_make_fact("test content", weight=0.9, source="analyst")]
        result = self.matcher.match(facts, "test")
        assert isinstance(result[0], KeyFact)
        assert result[0].weight == 0.9
        assert result[0].source == "analyst"

    def test_default_top_n_is_five(self):
        facts = [_make_fact(f"fact number {i}") for i in range(10)]
        result = self.matcher.match(facts, "fact number")
        assert len(result) == 5
