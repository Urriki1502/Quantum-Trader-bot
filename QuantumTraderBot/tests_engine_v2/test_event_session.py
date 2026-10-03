import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from engine.adapters import PaperExecutionAdapter, StaticQuoteProvider
from engine.journal import SQLiteMarketJournal
from engine.ledger import SQLiteTradeLedger
from engine.market import MarketTick
from engine.models import TradeState
from engine.risk import RiskEngine, RiskPolicy
from engine.service import TradingEngine
from engine.session import PaperTradingSession
from engine.signals import SignalAction, StrategySignal


def run(coro):
    return asyncio.run(coro)


class BuyOnceStrategy:
    strategy_id = "buy-once"

    def __init__(self):
        self.calls = 0

    def prime(self, ticks):
        return None

    def on_tick(self, tick, context):
        self.calls += 1
        if context.position_quantity <= 0:
            return StrategySignal(SignalAction.BUY, "test_buy")
        return StrategySignal(SignalAction.HOLD, "already_in_position")


def make_session(tmp_path):
    db = tmp_path / "paper.db"
    ledger = SQLiteTradeLedger(db)
    ledger.initialize_paper_account("1000")
    engine = TradingEngine(
        quote_provider=StaticQuoteProvider(
            prices_usd={"TOKEN": Decimal("10")},
            liquidity_usd=Decimal("1000000"),
            price_impact_bps=0,
        ),
        execution_adapter=PaperExecutionAdapter(slippage_bps=0, fee_bps=25),
        risk_engine=RiskEngine(
            RiskPolicy(
                max_notional_usd=Decimal("100"),
                max_open_exposure_usd=Decimal("1000"),
                min_liquidity_usd=Decimal("10000"),
                max_price_impact_bps=100,
                max_slippage_bps=100,
                max_daily_loss_usd=Decimal("100"),
            )
        ),
        ledger=ledger,
    )
    strategy = BuyOnceStrategy()
    journal = SQLiteMarketJournal(db)
    session = PaperTradingSession(
        engine=engine,
        journal=journal,
        strategy=strategy,
        buy_notional_usd="25",
        max_slippage_bps=100,
    )
    return session, ledger, journal, strategy


def tick(event_id="e1", price="10", when=None):
    return MarketTick(
        event_id=event_id,
        asset="TOKEN",
        price_usd=Decimal(price),
        liquidity_usd=Decimal("100000"),
        price_impact_bps=1,
        observed_at=when or datetime.now(timezone.utc),
        source="fixture",
    )


def test_duplicate_market_event_never_creates_second_intent(tmp_path):
    session, ledger, journal, strategy = make_session(tmp_path)
    market_tick = tick()

    first = run(session.process_tick(market_tick))
    second = run(session.process_tick(market_tick))

    assert first.engine_result is not None
    assert first.engine_result.state is TradeState.RECONCILED
    assert second.duplicate_event is True
    assert second.engine_result is None
    assert strategy.calls == 1
    assert journal.count() == 1
    assert ledger.portfolio_fill_count() == 1


def test_stale_live_tick_fails_closed(tmp_path):
    session, _, _, _ = make_session(tmp_path)
    old = tick(
        event_id="old",
        when=datetime.now(timezone.utc) - timedelta(minutes=5),
    )

    decision = run(session.process_tick(old))

    assert decision.engine_result is not None
    assert decision.engine_result.state is TradeState.REJECTED
    assert "market_data_stale" in decision.engine_result.reasons


def test_replay_allows_historical_tick_without_weakening_live_path(tmp_path):
    session, ledger, _, _ = make_session(tmp_path)
    old = tick(
        event_id="historic",
        when=datetime.now(timezone.utc) - timedelta(days=30),
    )

    decision = run(session.process_tick(old, replay=True))

    assert decision.engine_result is not None
    assert decision.engine_result.state is TradeState.RECONCILED
    assert ledger.portfolio_fill_count() == 1


def test_restart_journal_preserves_event_dedup(tmp_path):
    session, ledger, journal, _ = make_session(tmp_path)
    market_tick = tick(event_id="persisted")
    first = run(session.process_tick(market_tick))
    assert first.engine_result.state is TradeState.RECONCILED
    journal.close()
    ledger.close()

    session2, ledger2, journal2, strategy2 = make_session(tmp_path)
    second = run(session2.process_tick(market_tick))

    assert second.duplicate_event is True
    assert strategy2.calls == 0
    assert journal2.count() == 1
    assert ledger2.portfolio_fill_count() == 1
