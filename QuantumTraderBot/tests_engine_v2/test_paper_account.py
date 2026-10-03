import asyncio
from decimal import Decimal

import pytest

from engine.adapters import PaperExecutionAdapter, StaticQuoteProvider
from engine.ledger import SQLiteTradeLedger
from engine.models import TradeIntent, TradeSide, TradeState
from engine.risk import RiskEngine, RiskPolicy
from engine.service import TradingEngine


def run(coro):
    return asyncio.run(coro)


def make_engine(path, *, starting_cash="1000", price="10"):
    ledger = SQLiteTradeLedger(path)
    ledger.initialize_paper_account(starting_cash)
    quotes = StaticQuoteProvider(
        prices_usd={"TOKEN": Decimal(price)},
        liquidity_usd=Decimal("1000000"),
        price_impact_bps=0,
        fee_bps=25,
    )
    engine = TradingEngine(
        quote_provider=quotes,
        execution_adapter=PaperExecutionAdapter(slippage_bps=0, fee_bps=25),
        risk_engine=RiskEngine(
            RiskPolicy(
                max_notional_usd=Decimal("1000"),
                max_open_exposure_usd=Decimal("5000"),
                min_liquidity_usd=Decimal("10000"),
                max_price_impact_bps=300,
                max_slippage_bps=300,
                max_daily_loss_usd=Decimal("500"),
                buy_fee_reserve_bps=100,
            )
        ),
        ledger=ledger,
    )
    return engine, ledger, quotes


def test_paper_account_initialization_is_stable(tmp_path):
    ledger = SQLiteTradeLedger(tmp_path / "paper.db")
    first = ledger.initialize_paper_account("1000")
    second = ledger.initialize_paper_account(Decimal("1000"))

    assert first.cash_usd == Decimal("1000")
    assert second == first

    with pytest.raises(ValueError, match="different starting balance"):
        ledger.initialize_paper_account("2000")


def test_buy_debits_real_paper_cash_and_fee(tmp_path):
    engine, ledger, _ = make_engine(tmp_path / "paper.db", starting_cash="1000")
    intent = TradeIntent.create(
        asset="TOKEN",
        side=TradeSide.BUY,
        notional_usd="100",
        max_slippage_bps=100,
    )

    result = run(engine.execute_intent(intent, engine.risk_snapshot()))
    account = ledger.get_paper_account()

    assert result.state is TradeState.RECONCILED
    assert account is not None
    assert account.cash_usd == Decimal("899.75")


def test_buying_power_rejects_before_quote_execution(tmp_path):
    class CountingAdapter(PaperExecutionAdapter):
        def __init__(self):
            super().__init__(slippage_bps=0, fee_bps=25)
            self.calls = 0

        async def execute(self, intent, quote):
            self.calls += 1
            return await super().execute(intent, quote)

    ledger = SQLiteTradeLedger(tmp_path / "paper.db")
    ledger.initialize_paper_account("100")
    adapter = CountingAdapter()
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
                buy_fee_reserve_bps=100,
            )
        ),
        ledger=ledger,
    )
    intent = TradeIntent.create(
        asset="TOKEN",
        side=TradeSide.BUY,
        notional_usd="100",
        max_slippage_bps=100,
    )

    result = run(engine.execute_intent(intent, engine.risk_snapshot()))

    assert result.state is TradeState.REJECTED
    assert "insufficient_cash" in result.reasons
    assert adapter.calls == 0
    assert ledger.get_paper_account().cash_usd == Decimal("100")


def test_sell_credits_cash_and_realized_pnl(tmp_path):
    engine, ledger, quotes = make_engine(tmp_path / "paper.db", starting_cash="1000")
    buy = TradeIntent.create(
        asset="TOKEN",
        side=TradeSide.BUY,
        notional_usd="100",
        max_slippage_bps=100,
    )
    assert run(engine.execute_intent(buy, engine.risk_snapshot())).state is TradeState.RECONCILED

    quotes.prices_usd["TOKEN"] = Decimal("12")
    sell = TradeIntent.create(
        asset="TOKEN",
        side=TradeSide.SELL,
        notional_usd="60",
        max_slippage_bps=100,
    )
    result = run(engine.execute_intent(sell, engine.risk_snapshot()))
    account = ledger.get_paper_account()

    assert result.state is TradeState.RECONCILED
    assert account is not None
    assert account.cash_usd == Decimal("959.60")
    assert ledger.realized_pnl_today_utc() == Decimal("9.725")


def test_mark_to_market_tracks_equity_and_drawdown(tmp_path):
    engine, ledger, _ = make_engine(tmp_path / "paper.db", starting_cash="1000")
    buy = TradeIntent.create(
        asset="TOKEN",
        side=TradeSide.BUY,
        notional_usd="100",
        max_slippage_bps=100,
    )
    assert run(engine.execute_intent(buy, engine.risk_snapshot())).state is TradeState.RECONCILED

    at_entry = ledger.mark_to_market({"TOKEN": Decimal("10")})
    lower = ledger.mark_to_market({"TOKEN": Decimal("8")})

    assert at_entry.total_equity_usd == Decimal("999.75")
    assert lower.total_equity_usd == Decimal("979.75")
    assert lower.drawdown_pct > 0
    assert lower.max_drawdown_pct == lower.drawdown_pct
