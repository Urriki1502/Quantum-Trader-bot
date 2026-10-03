import asyncio
import base64
from datetime import timedelta
from decimal import Decimal

import pytest

from engine.ledger import ReconciliationError, SQLiteTradeLedger
from engine.live_guard import (
    GuardedLiveExecutionBoundary,
    LivePreparationError,
    UnsignedExecutionBundle,
)
from engine.models import Quote, TradeIntent, TradeSide, TradeState, utc_now
from engine.safety import LiveExecutionPolicy, LiveSafetyViolation
from engine.signer import SignedTransaction


def run(coro):
    return asyncio.run(coro)


class FakeBuilder:
    def __init__(self, *, count=1):
        self.count = count
        self.calls = 0

    async def build(self, intent, quote, *, wallet_pubkey):
        self.calls += 1
        txs = tuple(
            base64.b64encode(f"unsigned-{i}".encode()).decode()
            for i in range(self.count)
        )
        return UnsignedExecutionBundle(
            request_id="build-1",
            transactions_base64=txs,
            recent_blockhash="blockhash-1",
            last_valid_block_height=120,
        )


class FakeRpc:
    def __init__(self, *, height=100, valid=True, sim_err=None):
        self.height = height
        self.valid = valid
        self.sim_err = sim_err
        self.sim_calls = 0

    async def get_block_height(self, *, commitment="confirmed"):
        return self.height

    async def is_blockhash_valid(self, blockhash, *, commitment="confirmed"):
        return self.valid

    async def simulate_transaction(
        self,
        tx,
        *,
        commitment="confirmed",
        replace_recent_blockhash=True,
        sig_verify=False,
    ):
        self.sim_calls += 1
        return {
            "err": self.sim_err,
            "unitsConsumed": 123456,
            "logs": [],
        }


class FakeSigner:
    def __init__(self, *, signed_bytes=b"signed-fixed", pubkey="wallet-1"):
        self.public_key = pubkey
        self.signed_bytes = signed_bytes
        self.calls = 0

    async def sign_transaction(self, unsigned_transaction_base64):
        self.calls += 1
        return SignedTransaction(
            transaction_base64=base64.b64encode(self.signed_bytes).decode(),
            signer_pubkey=self.public_key,
        )


def prepared_trade(ledger):
    now = utc_now()
    intent = TradeIntent.create(
        asset="TOKEN",
        side=TradeSide.BUY,
        notional_usd="2",
        max_slippage_bps=100,
    )
    quote = Quote(
        quote_id="q1",
        provider="fixture",
        asset="TOKEN",
        side=TradeSide.BUY,
        price_usd=Decimal("1"),
        estimated_base_amount=Decimal("2"),
        notional_usd=Decimal("2"),
        liquidity_usd=Decimal("100000"),
        price_impact_bps=0,
        estimated_fee_usd=Decimal("0"),
        created_at=now,
        expires_at=now + timedelta(seconds=30),
    )
    ledger.create_intent(intent)
    ledger.transition(intent.intent_id, TradeState.RISK_APPROVED)
    ledger.attach_quote(intent.intent_id, quote)
    ledger.transition(intent.intent_id, TradeState.QUOTED)
    ledger.transition(intent.intent_id, TradeState.EXECUTION_PENDING)
    return intent, quote


def boundary(ledger, *, signer=None, builder=None, rpc=None):
    return GuardedLiveExecutionBoundary(
        ledger=ledger,
        policy=LiveExecutionPolicy(
            max_notional_usd=Decimal("5"),
            allowed_assets=frozenset({"TOKEN"}),
        ),
        builder=builder or FakeBuilder(),
        rpc=rpc or FakeRpc(),
        signer=signer or FakeSigner(),
    )


def test_prepare_is_allowed_while_live_submission_remains_locked(tmp_path):
    ledger = SQLiteTradeLedger(tmp_path / "live.db")
    intent, quote = prepared_trade(ledger)
    guard = boundary(ledger)

    plan = run(guard.prepare(intent, quote))

    assert plan.submission_authorized is False
    assert plan.simulation_units_consumed == 123456
    assert ledger.execution_attempt_count() == 0


def test_signer_is_not_called_until_both_live_flag_and_kill_switch_allow_it(tmp_path):
    ledger = SQLiteTradeLedger(tmp_path / "live.db")
    intent, quote = prepared_trade(ledger)
    signer = FakeSigner()
    guard = boundary(ledger, signer=signer)
    plan = run(guard.prepare(intent, quote))

    with pytest.raises(LiveSafetyViolation, match="kill switch"):
        run(guard.sign(plan))
    assert signer.calls == 0

    ledger.set_live_submission_enabled(True, reason="test-enable")
    with pytest.raises(LiveSafetyViolation, match="kill switch"):
        run(guard.sign(plan))
    assert signer.calls == 0

    ledger.release_kill_switch(reason="test-release")
    signed = run(guard.sign(plan))

    assert signer.calls == 1
    assert signed.tx_identity.startswith("sha256:")
    assert signed.attempt.existing is False
    assert ledger.execution_attempt_count() == 1


def test_same_signed_bytes_are_idempotent_but_changed_bytes_collide(tmp_path):
    ledger = SQLiteTradeLedger(tmp_path / "live.db")
    intent, quote = prepared_trade(ledger)
    signer = FakeSigner(signed_bytes=b"same-signed-bytes")
    guard = boundary(ledger, signer=signer)
    plan = run(guard.prepare(intent, quote))
    ledger.set_live_submission_enabled(True, reason="test-enable")
    ledger.release_kill_switch(reason="test-release")

    first = run(guard.sign(plan))
    second = run(guard.sign(plan))

    assert first.attempt.existing is False
    assert second.attempt.existing is True
    assert ledger.execution_attempt_count() == 1

    signer.signed_bytes = b"different-signed-bytes"
    with pytest.raises(ReconciliationError, match="identity collision"):
        run(guard.sign(plan))


def test_multi_transaction_bundle_fails_until_multi_leg_accounting_exists(tmp_path):
    ledger = SQLiteTradeLedger(tmp_path / "live.db")
    intent, quote = prepared_trade(ledger)
    guard = boundary(ledger, builder=FakeBuilder(count=2))

    with pytest.raises(LivePreparationError, match="multi-transaction"):
        run(guard.prepare(intent, quote))


def test_preflight_failure_fails_before_signing(tmp_path):
    ledger = SQLiteTradeLedger(tmp_path / "live.db")
    intent, quote = prepared_trade(ledger)
    signer = FakeSigner()
    guard = boundary(
        ledger,
        signer=signer,
        rpc=FakeRpc(sim_err={"InstructionError": [0, "Custom"]}),
    )

    with pytest.raises(LivePreparationError, match="preflight"):
        run(guard.prepare(intent, quote))
    assert signer.calls == 0


def test_expired_blockhash_fails_before_signing(tmp_path):
    ledger = SQLiteTradeLedger(tmp_path / "live.db")
    intent, quote = prepared_trade(ledger)
    signer = FakeSigner()
    guard = boundary(
        ledger,
        signer=signer,
        rpc=FakeRpc(height=121, valid=True),
    )

    with pytest.raises(Exception, match="blockhash"):
        run(guard.prepare(intent, quote))
    assert signer.calls == 0
