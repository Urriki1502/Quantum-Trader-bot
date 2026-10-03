import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from engine.fill import (
    FillReconciliationError,
    SolanaSwapFillReconciler,
    observed_fill_to_receipt,
)
from engine.models import Quote, TradeSide


OWNER = "Wallet111"
ASSET = "Asset111"
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


def token_balance(owner, mint, amount, decimals):
    return {
        "accountIndex": 1,
        "mint": mint,
        "owner": owner,
        "uiTokenAmount": {
            "amount": str(amount),
            "decimals": decimals,
            "uiAmount": None,
            "uiAmountString": "0",
        },
    }


def tx(pre, post, *, fee=5000, err=None, slot=123):
    return {
        "slot": slot,
        "meta": {
            "err": err,
            "fee": fee,
            "preTokenBalances": pre,
            "postTokenBalances": post,
        },
        "transaction": {"signatures": ["sig"]},
    }


def reconciler():
    return SolanaSwapFillReconciler(rpc=None)


def test_buy_fill_uses_wallet_token_deltas_not_quote_values():
    transaction = tx(
        [
            token_balance(OWNER, ASSET, 1_000_000, 6),
            token_balance(OWNER, USDC, 20_000_000, 6),
        ],
        [
            token_balance(OWNER, ASSET, 6_000_000, 6),
            token_balance(OWNER, USDC, 10_000_000, 6),
        ],
    )

    fill = reconciler().from_transaction(
        transaction,
        signature="sig-buy",
        wallet_owner=OWNER,
        asset_mint=ASSET,
        asset_decimals=6,
        side=TradeSide.BUY,
    )

    assert fill.base_amount == Decimal("5")
    assert fill.quote_amount == Decimal("10")
    assert fill.effective_price_usd == Decimal("2")
    assert fill.network_fee_lamports == 5000


def test_sell_fill_aggregates_multiple_wallet_token_accounts():
    transaction = tx(
        [
            token_balance(OWNER, ASSET, 4_000_000, 6),
            token_balance(OWNER, ASSET, 3_000_000, 6),
            token_balance(OWNER, USDC, 1_000_000, 6),
        ],
        [
            token_balance(OWNER, ASSET, 1_000_000, 6),
            token_balance(OWNER, ASSET, 1_000_000, 6),
            token_balance(OWNER, USDC, 11_000_000, 6),
        ],
    )

    fill = reconciler().from_transaction(
        transaction,
        signature="sig-sell",
        wallet_owner=OWNER,
        asset_mint=ASSET,
        asset_decimals=6,
        side=TradeSide.SELL,
    )

    assert fill.base_amount == Decimal("5")
    assert fill.quote_amount == Decimal("10")
    assert fill.effective_price_usd == Decimal("2")


def test_wrong_balance_direction_fails_closed():
    transaction = tx(
        [token_balance(OWNER, ASSET, 1_000_000, 6)],
        [token_balance(OWNER, ASSET, 2_000_000, 6)],
    )

    with pytest.raises(FillReconciliationError, match="BUY balance deltas"):
        reconciler().from_transaction(
            transaction,
            signature="sig",
            wallet_owner=OWNER,
            asset_mint=ASSET,
            asset_decimals=6,
            side=TradeSide.BUY,
        )


def test_failed_chain_transaction_cannot_be_reconciled():
    transaction = tx([], [], err={"InstructionError": [0, "Custom"]})

    with pytest.raises(FillReconciliationError, match="failed on chain"):
        reconciler().from_transaction(
            transaction,
            signature="sig",
            wallet_owner=OWNER,
            asset_mint=ASSET,
            asset_decimals=6,
            side=TradeSide.BUY,
        )


def test_observed_fill_to_receipt_calculates_adverse_slippage():
    transaction = tx(
        [
            token_balance(OWNER, ASSET, 0, 6),
            token_balance(OWNER, USDC, 11_000_000, 6),
        ],
        [
            token_balance(OWNER, ASSET, 5_000_000, 6),
            token_balance(OWNER, USDC, 1_000_000, 6),
        ],
    )
    fill = reconciler().from_transaction(
        transaction,
        signature="sig",
        wallet_owner=OWNER,
        asset_mint=ASSET,
        asset_decimals=6,
        side=TradeSide.BUY,
    )
    now = datetime.now(timezone.utc)
    quote = Quote(
        quote_id="q",
        provider="fixture",
        asset="TOKEN",
        side=TradeSide.BUY,
        price_usd=Decimal("1.90"),
        estimated_base_amount=Decimal("5.2"),
        notional_usd=Decimal("10"),
        liquidity_usd=Decimal("100000"),
        price_impact_bps=1,
        estimated_fee_usd=Decimal("0"),
        created_at=now,
        expires_at=now.replace(year=now.year + 1),
    )

    receipt = observed_fill_to_receipt(
        fill,
        quote=quote,
        execution_id="exec-1",
        submitted_at=now,
        confirmed_at=now,
    )

    assert receipt.average_price_usd == Decimal("2")
    assert receipt.filled_base_amount == Decimal("5")
    assert receipt.filled_quote_usd == Decimal("10")
    assert receipt.actual_slippage_bps == 527
    assert receipt.external_ref == "sig"


class FakeRpc:
    async def get_transaction(self, signature, *, commitment="confirmed"):
        return tx(
            [
                token_balance(OWNER, ASSET, 0, 6),
                token_balance(OWNER, USDC, 10_000_000, 6),
            ],
            [
                token_balance(OWNER, ASSET, 5_000_000, 6),
                token_balance(OWNER, USDC, 0, 6),
            ],
        )


def test_async_reconcile_reads_confirmed_transaction():
    fill = asyncio.run(
        SolanaSwapFillReconciler(FakeRpc()).reconcile(
            "sig",
            wallet_owner=OWNER,
            asset_mint=ASSET,
            asset_decimals=6,
            side=TradeSide.BUY,
        )
    )

    assert fill.base_amount == Decimal("5")
    assert fill.quote_amount == Decimal("10")
