from decimal import Decimal

import pytest

from engine.ledger import SQLiteTradeLedger
from engine.models import InvalidStateTransition, TradeIntent, TradeSide, TradeState


def test_ledger_survives_restart(tmp_path):
    path = tmp_path / "ledger.db"
    intent = TradeIntent.create(
        asset="TOKEN",
        side=TradeSide.BUY,
        notional_usd=Decimal("5"),
        max_slippage_bps=50,
    )

    ledger = SQLiteTradeLedger(path)
    ledger.create_intent(intent)
    ledger.transition(intent.intent_id, TradeState.RISK_APPROVED, reason="test")
    ledger.close()

    reopened = SQLiteTradeLedger(path)
    record = reopened.get_trade(intent.intent_id)
    events = reopened.events(intent.intent_id)

    assert record is not None
    assert record["state"] == TradeState.RISK_APPROVED.value
    assert [e["to_state"] for e in events] == ["created", "risk_approved"]
    reopened.close()


def test_invalid_state_transition_is_rejected(tmp_path):
    ledger = SQLiteTradeLedger(tmp_path / "ledger.db")
    intent = TradeIntent.create(
        asset="TOKEN",
        side=TradeSide.BUY,
        notional_usd=Decimal("5"),
        max_slippage_bps=50,
    )
    ledger.create_intent(intent)

    with pytest.raises(InvalidStateTransition):
        ledger.transition(intent.intent_id, TradeState.RECONCILED)
