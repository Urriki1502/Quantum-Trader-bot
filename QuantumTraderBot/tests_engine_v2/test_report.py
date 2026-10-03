import asyncio
from decimal import Decimal

import pytest

from engine.adapters import PaperExecutionAdapter, StaticQuoteProvider
from engine.ledger import ReconciliationError, SQLiteTradeLedger
from engine.models import TradeIntent, TradeSide, TradeState
from engine.report import build_performance_report
from engine.risk import RiskEngine, RiskPolicy
from engine.service import TradingEngine


def run(coro):
    return asyncio.run(coro)


def make_engine(path):
    ledger = SQLiteTradeLedger(path)
    ledger.initialize_paper_account("1000")
    quotes = StaticQuoteProvider(
        prices_usd={"TOKEN": Decimal("10")},
        liquidity_usd=Decimal("1000000"),
        price_impact_bps=0,
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
            )
        ),
        ledger=ledger,
    )
    return engine, ledger, quotes


def test_report_combines_cash_mark_to_market_and_realized_metrics(tmp_path):
    engine, ledger, quotes = make_engine(tmp_path / "paper.db")
    buy = TradeIntent.create(
        asset="TOKEN",
        side=TradeSide.BUY,
        notional_usd="100",
        max_slippage_bps=100,
    )
    assert run(engine.execute_intent(buy, engine.risk_snapshot())).state is TradeState.RECONCILED

    report = build_performance_report(
        ledger,
        prices_usd={"TOKEN": Decimal("12")},
    )

    assert report.cash_usd == Decimal("899.75")
    assert report.positions_value_usd == Decimal("120")
    assert report.total_equity_usd == Decimal("1019.75")
    assert report.net_pnl_usd == Decimal("19.75")
    assert report.unrealized_pnl_usd == Decimal("19.75")
    assert report.fills == 1
    assert '"total_equity_usd": "1019.75"' in report.as_json()


def test_report_requires_marks_for_every_open_position(tmp_path):
    engine, ledger, _ = make_engine(tmp_path / "paper.db")
    buy = TradeIntent.create(
        asset="TOKEN",
        side=TradeSide.BUY,
        notional_usd="100",
        max_slippage_bps=100,
    )
    run(engine.execute_intent(buy, engine.risk_snapshot()))

    with pytest.raises(ReconciliationError, match="missing report mark price"):
        build_performance_report(ledger, prices_usd={})
