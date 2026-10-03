from decimal import Decimal

import pytest

from engine.ledger import ReconciliationError, SQLiteTradeLedger
from engine.models import TradeIntent, TradeSide, TradeState


def prepared_trade(ledger):
    intent = TradeIntent.create(
        asset="TOKEN",
        side=TradeSide.BUY,
        notional_usd=Decimal("10"),
        max_slippage_bps=100,
    )
    ledger.create_intent(intent)
    ledger.transition(intent.intent_id, TradeState.RISK_APPROVED)
    ledger.transition(intent.intent_id, TradeState.QUOTED)
    ledger.transition(intent.intent_id, TradeState.EXECUTION_PENDING)
    return intent


def test_execution_attempt_is_reserved_once_and_survives_restart(tmp_path):
    path = tmp_path / "attempt.db"
    ledger = SQLiteTradeLedger(path)
    intent = prepared_trade(ledger)

    first = ledger.reserve_execution_attempt(
        intent.intent_id,
        attempt_id="attempt-1",
        tx_identity="sha256:abc",
        recent_blockhash="blockhash-1",
        last_valid_block_height=123,
    )
    second = ledger.reserve_execution_attempt(
        intent.intent_id,
        attempt_id="attempt-1",
        tx_identity="sha256:abc",
        recent_blockhash="blockhash-1",
        last_valid_block_height=123,
    )

    assert first.existing is False
    assert second.existing is True
    assert ledger.execution_attempt_count() == 1
    ledger.close()

    reopened = SQLiteTradeLedger(path)
    recovered = reopened.get_execution_attempt(intent.intent_id)
    assert recovered is not None
    assert recovered.tx_identity == "sha256:abc"
    assert recovered.last_valid_block_height == 123
    assert reopened.execution_attempt_count() == 1


def test_attempt_identity_cannot_change_for_same_intent(tmp_path):
    ledger = SQLiteTradeLedger(tmp_path / "attempt.db")
    intent = prepared_trade(ledger)
    ledger.reserve_execution_attempt(
        intent.intent_id,
        attempt_id="attempt-1",
        tx_identity="sha256:abc",
        recent_blockhash="blockhash-1",
        last_valid_block_height=123,
    )

    with pytest.raises(ReconciliationError, match="identity collision"):
        ledger.reserve_execution_attempt(
            intent.intent_id,
            attempt_id="attempt-2",
            tx_identity="sha256:def",
            recent_blockhash="blockhash-2",
            last_valid_block_height=456,
        )


def test_external_signature_is_bound_once(tmp_path):
    ledger = SQLiteTradeLedger(tmp_path / "attempt.db")
    intent = prepared_trade(ledger)
    ledger.reserve_execution_attempt(
        intent.intent_id,
        attempt_id="attempt-1",
        tx_identity="sha256:abc",
        recent_blockhash="blockhash-1",
        last_valid_block_height=123,
    )

    first = ledger.attach_attempt_external_ref(intent.intent_id, "signature-1")
    second = ledger.attach_attempt_external_ref(intent.intent_id, "signature-1")

    assert first.external_ref == "signature-1"
    assert second.existing is True

    with pytest.raises(ReconciliationError, match="different external reference"):
        ledger.attach_attempt_external_ref(intent.intent_id, "signature-2")
