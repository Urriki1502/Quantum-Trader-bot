import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from engine.raydium_api import RaydiumSwapQuote, RaydiumTradeApiClient, RaydiumTradeApiError
from engine.solana_rpc import SolanaRpcClient, SolanaRpcError


def run(coro):
    return asyncio.run(coro)


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def request_json(self, method, url, *, params=None, json_body=None, timeout=10.0):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "params": params,
                "json_body": json_body,
                "timeout": timeout,
            }
        )
        if not self.responses:
            raise AssertionError("unexpected HTTP request")
        return self.responses.pop(0)


def test_solana_latest_blockhash_parses_and_uses_confirmed():
    transport = FakeTransport(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "context": {"slot": 123},
                    "value": {"blockhash": "abc", "lastValidBlockHeight": 456},
                },
            }
        ]
    )
    client = SolanaRpcClient(transport=transport)

    result = run(client.get_latest_blockhash())

    assert result.blockhash == "abc"
    assert result.context_slot == 123
    assert result.last_valid_block_height == 456
    assert transport.calls[0]["json_body"]["method"] == "getLatestBlockhash"
    assert transport.calls[0]["json_body"]["params"][0]["commitment"] == "confirmed"


def test_solana_rpc_error_fails_closed():
    transport = FakeTransport(
        [{"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "boom"}}]
    )
    client = SolanaRpcClient(transport=transport)

    with pytest.raises(SolanaRpcError):
        run(client.get_health())


def test_raydium_quote_base_in_parses_public_api_shape():
    transport = FakeTransport(
        [
            {
                "id": "quote-1",
                "success": True,
                "data": {
                    "swapType": "BaseIn",
                    "inputMint": "IN",
                    "inputAmount": "1000",
                    "outputMint": "OUT",
                    "outputAmount": "2450",
                    "otherAmountThreshold": "2400",
                    "slippageBps": 50,
                    "priceImpactPct": 0.0012,
                    "routePlan": [{"poolId": "pool-1"}],
                },
            }
        ]
    )
    client = RaydiumTradeApiClient(transport=transport)

    quote = run(
        client.quote_base_in(
            input_mint="IN",
            output_mint="OUT",
            amount_atomic=1000,
            slippage_bps=50,
        )
    )

    assert quote.request_id == "quote-1"
    assert quote.output_amount_atomic == 2450
    assert quote.other_amount_threshold_atomic == 2400
    assert len(quote.route_plan) == 1
    assert transport.calls[0]["method"] == "GET"
    assert transport.calls[0]["params"]["txVersion"] == "V0"


def test_raydium_logical_error_fails_closed():
    transport = FakeTransport(
        [{"id": "quote-x", "success": False, "msg": "No route found", "data": None}]
    )
    client = RaydiumTradeApiClient(transport=transport)

    with pytest.raises(RaydiumTradeApiError, match="No route found"):
        run(
            client.quote_base_in(
                input_mint="IN",
                output_mint="OUT",
                amount_atomic=1000,
                slippage_bps=50,
            )
        )


def _quote(created_at):
    return RaydiumSwapQuote(
        request_id="q",
        input_mint="IN",
        output_mint="OUT",
        input_amount_atomic=1000,
        output_amount_atomic=2000,
        other_amount_threshold_atomic=1900,
        slippage_bps=50,
        price_impact_pct=0,
        route_plan=(),
        raw_data={
            "inputMint": "IN",
            "outputMint": "OUT",
            "inputAmount": "1000",
            "outputAmount": "2000",
            "otherAmountThreshold": "1900",
            "slippageBps": 50,
            "routePlan": [],
        },
        created_at=created_at,
    )


def test_raydium_build_rejects_stale_quote_without_http_call():
    transport = FakeTransport([])
    client = RaydiumTradeApiClient(transport=transport)
    stale = _quote(datetime.now(timezone.utc) - timedelta(seconds=31))

    with pytest.raises(RaydiumTradeApiError, match="stale"):
        run(
            client.build_swap_base_in(
                stale,
                wallet_pubkey="wallet",
                compute_unit_price_micro_lamports=0,
            )
        )

    assert transport.calls == []


def test_raydium_build_returns_all_unsigned_transactions():
    transport = FakeTransport(
        [
            {
                "id": "build-1",
                "success": True,
                "data": [
                    {"transaction": "base64-a"},
                    {"transaction": "base64-b"},
                ],
            }
        ]
    )
    client = RaydiumTradeApiClient(transport=transport)
    quote = _quote(datetime.now(timezone.utc))

    built = run(
        client.build_swap_base_in(
            quote,
            wallet_pubkey="wallet",
            compute_unit_price_micro_lamports=50000,
        )
    )

    assert built.transactions_base64 == ("base64-a", "base64-b")
    body = transport.calls[0]["json_body"]
    assert body["wallet"] == "wallet"
    assert body["computeUnitPriceMicroLamports"] == "50000"
    assert body["swapResponse"]["inputMint"] == "IN"
