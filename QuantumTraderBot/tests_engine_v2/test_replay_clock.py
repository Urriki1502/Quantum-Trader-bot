from datetime import datetime, timezone
from decimal import Decimal

from engine.models import Quote, TradeIntent, TradeSide
from engine.risk import RiskEngine, RiskPolicy, RiskSnapshot


def test_replay_evaluation_time_is_timezone_checked_and_used_by_caller():
    when = datetime(2025, 1, 1, tzinfo=timezone.utc)
    snapshot = RiskSnapshot(evaluation_time=when)
    assert snapshot.evaluation_time == when


def test_historical_quote_can_be_evaluated_at_historical_as_of_time():
    when = datetime(2025, 1, 1, tzinfo=timezone.utc)
    intent = TradeIntent.create(
        asset="TOKEN",
        side=TradeSide.BUY,
        notional_usd=Decimal("10"),
        max_slippage_bps=100,
    )
    quote = Quote(
        quote_id="q",
        provider="replay",
        asset="TOKEN",
        side=TradeSide.BUY,
        price_usd=Decimal("2"),
        estimated_base_amount=Decimal("5"),
        notional_usd=Decimal("10"),
        liquidity_usd=Decimal("100000"),
        price_impact_bps=0,
        estimated_fee_usd=Decimal("0"),
        created_at=when,
        expires_at=when.replace(hour=1),
    )
    engine = RiskEngine(RiskPolicy(max_quote_age_seconds=20))

    decision = engine.evaluate_quote(intent, quote, now=when)

    assert decision.allowed is True
