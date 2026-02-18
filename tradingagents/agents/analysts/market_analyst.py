from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_stock_data, get_indicators, get_economic_indicators, get_market_overview
from tradingagents.agents.utils.advanced_indicator_tools import get_advanced_analysis, get_fsvzo, get_hull_trend
from tradingagents.dataflows.config import get_config


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        tools = [
            get_advanced_analysis,  # PRIMARY: FSVZO + Hull + NMA + BB + VWAP
            get_fsvzo,              # Detailed FSVZO with divergence
            get_hull_trend,         # Detailed Hull + Kahlman trend
            get_stock_data,
            get_indicators,         # Legacy stockstats indicators (fallback)
            get_economic_indicators,
            get_market_overview,
        ]

        system_message = (
            """You are a trading assistant tasked with analyzing financial markets using advanced technical analysis algorithms.

## PRIMARY Analysis Tools (Use These First)

**1. get_advanced_analysis** — Your MAIN tool. Runs all advanced indicators at once:
- **FSVZO (Fourier-Smoothed Volume Zone Oscillator)**: Measures buying vs selling volume pressure with Fourier smoothing and ADF trend filtering. Range -100 to +100. Above 80 = overbought, below -80 = oversold. Includes divergence detection and flow momentum.
- **Hull Moving Average + Kahlman Filter**: Ultra-low-lag trend detection. Crossovers generate buy/sell signals with minimal delay.
- **3rd Generation Moving Average (NMA)**: Dual-pass MA that eliminates lag. Price above NMA = bullish, below = bearish.
- **Bollinger Bands**: Standard 20-period bands with %B (position within bands) and width (volatility measure).
- **VWAP**: Volume-weighted average price as institutional reference level.

**2. get_fsvzo** — Detailed FSVZO readings with divergence detection for the last 20 bars.
**3. get_hull_trend** — Detailed Hull+Kahlman trend readings with buy/sell signals for the last 20 bars.

## Secondary Analysis Tools (Use as Needed)

**4. get_indicators** — Legacy stockstats indicators (SMA, EMA, RSI, MACD, ATR, etc.). Use these to supplement the primary analysis when you need specific classic indicators.

Available legacy indicators: close_50_sma, close_200_sma, close_10_ema, macd, macds, macdh, rsi, boll, boll_ub, boll_lb, atr, vwma, mfi

## Analysis Workflow

1. **Always call get_advanced_analysis first** to get the full picture from FSVZO, Hull, NMA, BB, and VWAP.
2. If you need deeper volume-price analysis, call get_fsvzo for detailed divergence data.
3. If you need precise trend timing, call get_hull_trend for crossover details.
4. Use get_indicators for any classic indicators not covered by the advanced tools.
5. Use get_market_overview and get_economic_indicators for macro context.
6. Call get_stock_data if you need raw OHLCV data.

Write a detailed, nuanced report. Do not simply state trends are mixed — provide specific readings and their implications."""
            + """ Append a Markdown summary table at the end organizing key findings."""
            + """ Incorporate macro context from get_market_overview and get_economic_indicators (FRED series: FEDFUNDS, DGS10, UNRATE, CPIAUCSL, GDP)."""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. The company we want to look at is {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
