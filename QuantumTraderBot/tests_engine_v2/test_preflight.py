import asyncio

import pytest

from engine.preflight import (
    PreflightSimulationError,
    RaydiumPreflightSimulator,
)
from engine.raydium_api import RaydiumBuiltSwap, RaydiumSwapQuote
from datetime import datetime, timezone
from decimal import Decimal


def run(coro):
    return asyncio.run(coro)


class FakeTradeApi:
    async def build_swap_base_in(self, quote, **kwargs):
        return RaydiumBuiltSwap(
            request_id="build-1",
            transactions_base64=("tx-a", "tx-b"),
        )


class FakeRpc:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    async def simulate_transaction(self, tx, **kwargs):
        self.calls.append((tx, kwargs))
        return self.values.pop(0)


def quote():
    return RaydiumSwapQuote(
        request_id="q",
        input_mint="IN",
        output_mint="OUT",
        input_amount_atomic=100,
        output_amount_atomic=200,
        other_amount_threshold_atomic=190,
        slippage_bps=50,
        price_impact_pct=Decimal("0.01"),
        route_plan=(),
        raw_data={},
        created_at=datetime.now(timezone.utc),
    )


def test_preflight_simulates_every_built_transaction_without_signature_verification():
    rpc = FakeRpc(
        [
            {"err": None, "unitsConsumed": 100000, "logs": ["ok-a"]},
            {"err": None, "unitsConsumed": 120000, "logs": ["ok-b"]},
        ]
    )
    simulator = RaydiumPreflightSimulator(
        trade_api=FakeTradeApi(),
        rpc=rpc,
    )

    result = run(
        simulator.simulate_base_in(
            quote(),
            wallet_pubkey="public-wallet-only",
            compute_unit_price_micro_lamports=50000,
        )
    )

    assert result.all_passed is True
    assert len(result.transactions) == 2
    assert result.transactions[0].units_consumed == 100000
    assert all(call[1]["sig_verify"] is False for call in rpc.calls)
    assert all(call[1]["replace_recent_blockhash"] is True for call in rpc.calls)


def test_preflight_failure_is_visible_and_require_pass_fails_closed():
    rpc = FakeRpc(
        [
            {"err": None, "unitsConsumed": 100000, "logs": []},
            {"err": {"InstructionError": [1, "Custom"]}, "unitsConsumed": 90000, "logs": ["boom"]},
        ]
    )
    simulator = RaydiumPreflightSimulator(
        trade_api=FakeTradeApi(),
        rpc=rpc,
    )

    result = run(simulator.simulate_base_in(quote(), wallet_pubkey="pubkey"))

    assert result.all_passed is False
    assert result.transactions[1].ok is False
    with pytest.raises(PreflightSimulationError):
        simulator.require_pass(result)
