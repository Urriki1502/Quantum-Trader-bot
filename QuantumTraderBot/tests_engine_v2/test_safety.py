from decimal import Decimal

import pytest

from engine.ledger import SQLiteTradeLedger
from engine.models import TradeIntent, TradeSide
from engine.safety import LiveExecutionPolicy, LiveSafetyViolation


def intent(asset="TOKEN", usd="2"):
    return TradeIntent.create(
        asset=asset,
        side=TradeSide.BUY,
        notional_usd=Decimal(usd),
        max_slippage_bps=100,
    )


def test_runtime_safety_defaults_fail_closed_and_persist(tmp_path):
    path = tmp_path / "safety.db"
    ledger = SQLiteTradeLedger(path)
    initial = ledger.get_runtime_safety()

    assert initial.live_submission_enabled is False
    assert initial.kill_switch_engaged is True
    assert initial.reason == "default_fail_closed"

    ledger.set_live_submission_enabled(True, reason="explicit-test-enable")
    ledger.engage_kill_switch(reason="operator-stop")
    ledger.close()

    reopened = SQLiteTradeLedger(path)
    state = reopened.get_runtime_safety()
    assert state.live_submission_enabled is True
    assert state.kill_switch_engaged is True
    assert state.reason == "operator-stop"


def test_safety_changes_require_reason(tmp_path):
    ledger = SQLiteTradeLedger(tmp_path / "safety.db")
    with pytest.raises(ValueError):
        ledger.set_live_submission_enabled(True, reason="")
    with pytest.raises(ValueError):
        ledger.release_kill_switch(reason="")


def test_policy_hard_caps_and_allowlist():
    policy = LiveExecutionPolicy(
        max_notional_usd=Decimal("5"),
        allowed_assets=frozenset({"TOKEN"}),
    )

    policy.require_intent_allowed(intent("TOKEN", "5"))

    with pytest.raises(LiveSafetyViolation, match="hard cap"):
        policy.require_intent_allowed(intent("TOKEN", "5.01"))

    with pytest.raises(LiveSafetyViolation, match="allowlist"):
        policy.require_intent_allowed(intent("OTHER", "1"))
