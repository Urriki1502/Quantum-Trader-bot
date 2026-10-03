import asyncio

import pytest

from engine.outcome import (
    ExpiredBlockhashError,
    OutcomeState,
    SolanaBlockhashGuard,
    SolanaOutcomeReconciler,
)


def run(coro):
    return asyncio.run(coro)


class FakeRpc:
    def __init__(self, *, status=None, height=100, blockhash_valid=True):
        self.status = status
        self.height = height
        self.blockhash_valid = blockhash_valid
        self.status_calls = 0

    async def get_signature_statuses(self, signatures, *, search_transaction_history=True):
        self.status_calls += 1
        return [self.status]

    async def get_block_height(self, *, commitment="confirmed"):
        return self.height

    async def is_blockhash_valid(self, blockhash, *, commitment="confirmed"):
        return self.blockhash_valid


def test_confirmed_signature_resolves_without_waiting_for_expiry():
    rpc = FakeRpc(
        status={
            "slot": 1,
            "err": None,
            "confirmationStatus": "confirmed",
        },
        height=999,
    )
    reconciler = SolanaOutcomeReconciler(rpc)

    result = run(
        reconciler.resolve("sig", last_valid_block_height=123)
    )

    assert result.state is OutcomeState.CONFIRMED
    assert result.found is True
    assert result.current_block_height is None


def test_failed_signature_is_not_retried_as_pending():
    rpc = FakeRpc(
        status={
            "slot": 1,
            "err": {"InstructionError": [0, "Custom"]},
            "confirmationStatus": "confirmed",
        }
    )
    result = run(
        SolanaOutcomeReconciler(rpc).resolve(
            "sig",
            last_valid_block_height=123,
        )
    )

    assert result.state is OutcomeState.FAILED
    assert result.err is not None


def test_missing_signature_becomes_expired_only_after_last_valid_height():
    before = run(
        SolanaOutcomeReconciler(FakeRpc(status=None, height=123)).resolve(
            "sig",
            last_valid_block_height=123,
        )
    )
    after = run(
        SolanaOutcomeReconciler(FakeRpc(status=None, height=124)).resolve(
            "sig",
            last_valid_block_height=123,
        )
    )

    assert before.state is OutcomeState.PENDING
    assert before.found is False
    assert after.state is OutcomeState.EXPIRED
    assert after.found is False


def test_blockhash_guard_requires_both_height_and_rpc_validity():
    valid = FakeRpc(height=100, blockhash_valid=True)
    guard = SolanaBlockhashGuard(valid)

    result = run(guard.require_valid("hash", 120))
    assert result.valid_for_submission is True

    invalid = SolanaBlockhashGuard(
        FakeRpc(height=100, blockhash_valid=False)
    )
    with pytest.raises(ExpiredBlockhashError):
        run(invalid.require_valid("hash", 120))

    too_late = SolanaBlockhashGuard(
        FakeRpc(height=121, blockhash_valid=True)
    )
    with pytest.raises(ExpiredBlockhashError):
        run(too_late.require_valid("hash", 120))
