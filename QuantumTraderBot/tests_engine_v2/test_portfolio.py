import asyncio
from datetime import datetime, timezone
from decimal import Decimal

from engine.adapters import PaperExecutionAdapter, StaticQuoteProvider
from engine.ledger import SQLiteTradeLedger
from engine.models import ExecutionReceipt, TradeIntent, TradeSide, TradeState
from engine.risk import RiskEngine, RiskPolicy, RiskSnapshot
from engine.service import TradingEngine


def run(coro):
    return asyncio.run(coro)


def make_engine(path, *, price="10", slippage_bps=0, fee_bps=25):
    ledger = SQLiteTradeLedger(path)
    quotes = StaticQuoteProvider(
        prices_usd={"TOKEN": Decimal(price)},
        liquidity_usd=Decimal("1000000"),
        price_impact_bps=0,
        fee_bps=fee_bps,
    )
    engine = TradingEngine(
        quote_provider=quotes,
        execution_adapter=PaperExecutionAdapter(
            slippage_bps=slippage_bps,
            fee_bps=fee_bps,
        ),
        risk_engine=RiskEngine(
            RiskPolicy(
                max_notional_usd=Decimal("1000"),
                max_open_exposure_usd=Decimal("5000"),
                min_liquidity_usd=Decimal("10000"),
                max_price_impact_bps=300,
                max_slippage_bps=300,
                max_daily_loss_usd=Decimal("500"),
                max_quote_age_seconds=20,
            )
        ),
        ledger=ledger,
    )
    return engine, ledger, quotes


def test_buy_fill_creates_position_with_fee_in_cost_basis(tmp_path):
    engine, ledger, _ = make_engine(tmp_path / "paper.db")
    intent = TradeIntent.create(
        asset="TOKEN",
        side=TradeSide.BUY,
        notional_usd="100",
        max_slippage_bps=100,
    )

    result = run(engine.execute_intent(intent, RiskSnapshot()))
    position = ledger.get_position("TOKEN")

    assert result.state is TradeState.RECONCILED
    assert position.quantity == Decimal("10")
    assert position.average_cost_usd == Decimal("10.025")
    assert position.realized_pnl_usd == Decimal("0")
    assert ledger.portfolio_fill_count() == 1


def test_profitable_sell_updates_quantity_and_realized_pnl(tmp_path):
    engine, ledger, quotes = make_engine(tmp_path / "paper.db")

    buy = TradeIntent.create(
        asset="TOKEN",
        side=TradeSide.BUY,
        notional_usd="100",
        max_slippage_bps=100,
    )
    assert run(engine.execute_intent(buy, RiskSnapshot())).state is TradeState.RECONCILED

    quotes.prices_usd["TOKEN"] = Decimal("12")
    sell = TradeIntent.create(
        asset="TOKEN",
        side=TradeSide.SELL,
        notional_usd="60",
        max_slippage_bps=100,
    )
    result = run(engine.execute_intent(sell, engine.risk_snapshot()))
    position = ledger.get_position("TOKEN")
    metrics = ledger.portfolio_metrics()

    assert result.state is TradeState.RECONCILED
    assert position.quantity == Decimal("5")
    assert position.realized_pnl_usd == Decimal("9.725")
    assert metrics.fills == 2
    assert metrics.buys == 1
    assert metrics.sells == 1
    assert metrics.winning_sells == 1
    assert metrics.win_rate == Decimal("1")


def test_sell_without_position_is_rejected_before_execution(tmp_path):
    class CountingAdapter(PaperExecutionAdapter):
        def __init__(self):
            super().__init__(slippage_bps=0)
            self.calls = 0

        async def execute(self, intent, quote):
            self.calls += 1
            return await super().execute(intent, quote)

    adapter = CountingAdapter()
    ledger = SQLiteTradeLedger(tmp_path / "paper.db")
    engine = TradingEngine(
        quote_provider=StaticQuoteProvider(
            prices_usd={"TOKEN": Decimal("10")},
            liquidity_usd=Decimal("1000000"),
            price_impact_bps=0,
        ),
        execution_adapter=adapter,
        risk_engine=RiskEngine(
            RiskPolicy(
                max_notional_usd=Decimal("1000"),
                max_open_exposure_usd=Decimal("5000"),
                min_liquidity_usd=Decimal("10000"),
                max_price_impact_bps=300,
                max_slippage_bps=300,
                max_daily_loss_usd=Decimal("500"),
            )
        ),
        ledger=ledger,
    )
    sell = TradeIntent.create(
        asset="TOKEN",
        side=TradeSide.SELL,
        notional_usd="25",
        max_slippage_bps=100,
    )

    result = run(engine.execute_intent(sell, RiskSnapshot()))

    assert result.state is TradeState.REJECTED
    assert "insufficient_position" in result.reasons
    assert adapter.calls == 0
    assert ledger.portfolio_fill_count() == 0


def test_restart_recovers_confirmed_fill_exactly_once(tmp_path):
    path = tmp_path / "paper.db"
    engine, ledger, _ = make_engine(path)
    intent = TradeIntent.create(
        asset="TOKEN",
        side=TradeSide.BUY,
        notional_usd="20",
        max_slippage_bps=100,
    )
    ledger.create_intent(intent)
    ledger.transition(intent.intent_id, TradeState.RISK_APPROVED)
    quote = run(engine.quote_provider.quote(intent))
    ledger.attach_quote(intent.intent_id, quote)
    ledger.transition(intent.intent_id, TradeState.QUOTED)
    ledger.transition(intent.intent_id, TradeState.EXECUTION_PENDING)
    receipt = run(engine.execution_adapter.execute(intent, quote))
    ledger.attach_execution(intent.intent_id, receipt)
    ledger.transition(intent.intent_id, TradeState.SUBMITTED)
    ledger.transition(intent.intent_id, TradeState.CONFIRMED)
    ledger.close()

    recovered_engine, recovered_ledger, _ = make_engine(path)
    first = recovered_engine.recover_confirmed()
    second = recovered_engine.recover_confirmed()

    assert len(first) == 1
    assert first[0].state is TradeState.RECONCILED
    assert second == []
    assert recovered_ledger.portfolio_fill_count() == 1
    assert recovered_ledger.get_position("TOKEN").quantity == Decimal("2")


def test_reconciliation_is_idempotent_if_fill_exists_before_state_repair(tmp_path):
    engine, ledger, _ = make_engine(tmp_path / "paper.db")
    intent = TradeIntent.create(
        asset="TOKEN",
        side=TradeSide.BUY,
        notional_usd="20",
        max_slippage_bps=100,
    )
    ledger.create_intent(intent)
    ledger.transition(intent.intent_id, TradeState.RISK_APPROVED)
    quote = run(engine.quote_provider.quote(intent))
    ledger.attach_quote(intent.intent_id, quote)
    ledger.transition(intent.intent_id, TradeState.QUOTED)
    ledger.transition(intent.intent_id, TradeState.EXECUTION_PENDING)
    receipt = run(engine.execution_adapter.execute(intent, quote))
    ledger.attach_execution(intent.intent_id, receipt)
    ledger.transition(intent.intent_id, TradeState.SUBMITTED)
    ledger.transition(intent.intent_id, TradeState.CONFIRMED)

    first = ledger.reconcile_confirmed_execution(intent, receipt)
    second = ledger.reconcile_confirmed_execution(intent, receipt)

    assert first.existing is False
    assert second.existing is True
    assert ledger.portfolio_fill_count() == 1
    assert ledger.get_position("TOKEN").quantity == Decimal("2")
