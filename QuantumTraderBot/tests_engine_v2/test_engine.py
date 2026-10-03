import asyncio
from decimal import Decimal

from engine.adapters import PaperExecutionAdapter, StaticQuoteProvider, UnknownExecutionOutcome
from engine.ledger import SQLiteTradeLedger
from engine.models import TradeIntent, TradeSide, TradeState
from engine.risk import RiskEngine, RiskPolicy, RiskSnapshot
from engine.service import TradingEngine


def run(coro):
    return asyncio.run(coro)


def make_engine(tmp_path, *, adapter=None, liquidity="1000000", impact=25):
    ledger = SQLiteTradeLedger(tmp_path / "qt.db")
    quote_provider = StaticQuoteProvider(
        prices_usd={"TOKEN": Decimal("2")},
        liquidity_usd=Decimal(liquidity),
        price_impact_bps=impact,
    )
    execution_adapter = adapter or PaperExecutionAdapter(slippage_bps=10)
    risk = RiskEngine(
        RiskPolicy(
            max_notional_usd=Decimal("100"),
            max_open_exposure_usd=Decimal("500"),
            min_liquidity_usd=Decimal("10000"),
            max_price_impact_bps=300,
            max_slippage_bps=300,
            max_daily_loss_usd=Decimal("100"),
            max_quote_age_seconds=20,
        )
    )
    return TradingEngine(
        quote_provider=quote_provider,
        execution_adapter=execution_adapter,
        risk_engine=risk,
        ledger=ledger,
    ), ledger


def test_paper_buy_reaches_reconciled(tmp_path):
    engine, ledger = make_engine(tmp_path)
    intent = TradeIntent.create(
        asset="TOKEN",
        side=TradeSide.BUY,
        notional_usd="25",
        max_slippage_bps=100,
    )

    result = run(engine.execute_intent(intent, RiskSnapshot()))

    assert result.state is TradeState.RECONCILED
    assert result.record["execution_mode"] == "paper"
    assert result.record["external_ref"].startswith("paper:")
    assert Decimal(result.record["filled_quote_usd"]) == Decimal("25")

    states = [event["to_state"] for event in ledger.events(intent.intent_id)]
    assert states == [
        "created",
        "risk_approved",
        "quoted",
        "execution_pending",
        "submitted",
        "confirmed",
        "reconciled",
    ]


class CountingPaperAdapter(PaperExecutionAdapter):
    def __init__(self):
        super().__init__(slippage_bps=10)
        self.calls = 0

    async def execute(self, intent, quote):
        self.calls += 1
        return await super().execute(intent, quote)


def test_duplicate_intent_is_idempotent(tmp_path):
    adapter = CountingPaperAdapter()
    engine, _ = make_engine(tmp_path, adapter=adapter)
    intent = TradeIntent.create(
        asset="TOKEN",
        side=TradeSide.BUY,
        notional_usd="10",
        max_slippage_bps=100,
    )

    first = run(engine.execute_intent(intent, RiskSnapshot()))
    second = run(engine.execute_intent(intent, RiskSnapshot()))

    assert first.state is TradeState.RECONCILED
    assert second.state is TradeState.RECONCILED
    assert second.existing is True
    assert adapter.calls == 1
    assert first.record["execution_id"] == second.record["execution_id"]


def test_pre_trade_risk_rejects_oversized_notional(tmp_path):
    adapter = CountingPaperAdapter()
    engine, _ = make_engine(tmp_path, adapter=adapter)
    intent = TradeIntent.create(
        asset="TOKEN",
        side=TradeSide.BUY,
        notional_usd="101",
        max_slippage_bps=100,
    )

    result = run(engine.execute_intent(intent, RiskSnapshot()))

    assert result.state is TradeState.REJECTED
    assert "max_notional_exceeded" in result.reasons
    assert adapter.calls == 0


def test_quote_risk_rejects_low_liquidity(tmp_path):
    adapter = CountingPaperAdapter()
    engine, _ = make_engine(tmp_path, adapter=adapter, liquidity="500")
    intent = TradeIntent.create(
        asset="TOKEN",
        side=TradeSide.BUY,
        notional_usd="10",
        max_slippage_bps=100,
    )

    result = run(engine.execute_intent(intent, RiskSnapshot()))

    assert result.state is TradeState.REJECTED
    assert "insufficient_liquidity" in result.reasons
    assert adapter.calls == 0


class UnknownAdapter:
    mode = "test-unknown"

    def __init__(self):
        self.calls = 0

    async def execute(self, intent, quote):
        self.calls += 1
        raise UnknownExecutionOutcome("socket lost after submit", external_ref="candidate-123")


def test_unknown_outcome_is_not_retried(tmp_path):
    adapter = UnknownAdapter()
    engine, _ = make_engine(tmp_path, adapter=adapter)
    intent = TradeIntent.create(
        asset="TOKEN",
        side=TradeSide.BUY,
        notional_usd="10",
        max_slippage_bps=100,
    )

    first = run(engine.execute_intent(intent, RiskSnapshot()))
    second = run(engine.execute_intent(intent, RiskSnapshot()))

    assert first.state is TradeState.UNKNOWN
    assert second.state is TradeState.UNKNOWN
    assert second.existing is True
    assert adapter.calls == 1


def test_daily_loss_circuit_breaker_is_fail_closed(tmp_path):
    engine, _ = make_engine(tmp_path)
    intent = TradeIntent.create(
        asset="TOKEN",
        side=TradeSide.SELL,
        notional_usd="10",
        max_slippage_bps=100,
    )

    result = run(
        engine.execute_intent(
            intent,
            RiskSnapshot(realized_pnl_today_usd=Decimal("-100")),
        )
    )

    assert result.state is TradeState.REJECTED
    assert "daily_loss_limit_reached" in result.reasons
