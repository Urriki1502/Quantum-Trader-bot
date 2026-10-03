from datetime import datetime, timezone
from decimal import Decimal

from engine.market import MarketTick
from engine.signals import (
    MovingAverageCrossStrategy,
    SignalAction,
    StrategyContext,
)


def tick(i, price):
    return MarketTick(
        event_id=f"e{i}",
        asset="TOKEN",
        price_usd=Decimal(str(price)),
        liquidity_usd=Decimal("100000"),
        price_impact_bps=1,
        observed_at=datetime.now(timezone.utc),
        source="fixture",
    )


def context(position="0"):
    return StrategyContext(
        position_quantity=Decimal(position),
        available_cash_usd=Decimal("1000"),
        realized_pnl_today_usd=Decimal("0"),
    )


def test_ma_cross_emits_buy_only_on_actual_cross():
    strategy = MovingAverageCrossStrategy(short_window=2, long_window=3)
    prices = [3, 2, 1, 4]
    signals = [strategy.on_tick(tick(i, p), context()) for i, p in enumerate(prices)]

    assert signals[0].action is SignalAction.HOLD
    assert signals[1].action is SignalAction.HOLD
    assert signals[2].action is SignalAction.HOLD
    assert signals[3].action is SignalAction.BUY


def test_ma_cross_emits_sell_when_position_exists_and_crosses_down():
    strategy = MovingAverageCrossStrategy(short_window=2, long_window=3)
    strategy.prime([tick(0, 1), tick(1, 2), tick(2, 3)])
    signal = strategy.on_tick(tick(3, 0.5), context(position="10"))

    assert signal.action is SignalAction.SELL
