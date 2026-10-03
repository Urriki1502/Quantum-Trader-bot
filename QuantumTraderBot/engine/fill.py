from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_CEILING
from typing import Any, Mapping, Sequence

from .assets import USDC_MINT
from .models import ExecutionReceipt, Quote, TradeSide, TradeState
from .solana_rpc import SolanaRpcClient


class FillReconciliationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ObservedSwapFill:
    signature: str
    slot: int
    side: TradeSide
    asset_mint: str
    quote_mint: str
    base_amount: Decimal
    quote_amount: Decimal
    effective_price_usd: Decimal
    network_fee_lamports: int


class SolanaSwapFillReconciler:
    """Derive a swap fill from confirmed wallet token-balance deltas.

    The reconciler trusts transaction metadata, not the pre-submit quote.
    Current scope is SPL-token <-> USDC. Native SOL/wSOL accounting is kept out
    until a separate lamport/wrapped-SOL reconciliation path is validated.
    """

    def __init__(
        self,
        rpc: SolanaRpcClient,
        *,
        quote_mint: str = USDC_MINT,
        quote_decimals: int = 6,
    ) -> None:
        self.rpc = rpc
        self.quote_mint = quote_mint
        self.quote_decimals = quote_decimals

    async def reconcile(
        self,
        signature: str,
        *,
        wallet_owner: str,
        asset_mint: str,
        asset_decimals: int,
        side: TradeSide,
    ) -> ObservedSwapFill:
        transaction = await self.rpc.get_transaction(
            signature,
            commitment="confirmed",
        )
        if transaction is None:
            raise FillReconciliationError("confirmed transaction body is not available")
        return self.from_transaction(
            transaction,
            signature=signature,
            wallet_owner=wallet_owner,
            asset_mint=asset_mint,
            asset_decimals=asset_decimals,
            side=side,
        )

    def from_transaction(
        self,
        transaction: Mapping[str, Any],
        *,
        signature: str,
        wallet_owner: str,
        asset_mint: str,
        asset_decimals: int,
        side: TradeSide,
    ) -> ObservedSwapFill:
        if not signature or not wallet_owner or not asset_mint:
            raise FillReconciliationError("signature, wallet_owner and asset_mint are required")
        if asset_decimals < 0 or asset_decimals > 18:
            raise FillReconciliationError("asset_decimals is invalid")

        meta = transaction.get("meta")
        if not isinstance(meta, Mapping):
            raise FillReconciliationError("transaction meta is missing")
        if meta.get("err") is not None:
            raise FillReconciliationError(f"transaction failed on chain: {meta['err']}")

        pre = self._owner_balances(meta.get("preTokenBalances"), wallet_owner)
        post = self._owner_balances(meta.get("postTokenBalances"), wallet_owner)

        asset_delta_atomic = post.get(asset_mint, 0) - pre.get(asset_mint, 0)
        quote_delta_atomic = post.get(self.quote_mint, 0) - pre.get(self.quote_mint, 0)

        asset_scale = Decimal(10) ** asset_decimals
        quote_scale = Decimal(10) ** self.quote_decimals

        if side is TradeSide.BUY:
            if asset_delta_atomic <= 0 or quote_delta_atomic >= 0:
                raise FillReconciliationError(
                    "BUY balance deltas do not show asset received and quote spent"
                )
            base_amount = Decimal(asset_delta_atomic) / asset_scale
            quote_amount = Decimal(-quote_delta_atomic) / quote_scale
        else:
            if asset_delta_atomic >= 0 or quote_delta_atomic <= 0:
                raise FillReconciliationError(
                    "SELL balance deltas do not show asset spent and quote received"
                )
            base_amount = Decimal(-asset_delta_atomic) / asset_scale
            quote_amount = Decimal(quote_delta_atomic) / quote_scale

        if base_amount <= 0 or quote_amount <= 0:
            raise FillReconciliationError("observed swap amounts must be positive")

        slot = transaction.get("slot")
        fee = meta.get("fee")
        if not isinstance(slot, int):
            raise FillReconciliationError("transaction slot is missing")
        if not isinstance(fee, int) or fee < 0:
            raise FillReconciliationError("transaction fee is missing or invalid")

        return ObservedSwapFill(
            signature=signature,
            slot=slot,
            side=side,
            asset_mint=asset_mint,
            quote_mint=self.quote_mint,
            base_amount=base_amount,
            quote_amount=quote_amount,
            effective_price_usd=quote_amount / base_amount,
            network_fee_lamports=fee,
        )

    @staticmethod
    def _owner_balances(raw: Any, owner: str) -> dict[str, int]:
        if raw is None:
            return {}
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise FillReconciliationError("token balances must be a list")

        totals: dict[str, int] = {}
        for entry in raw:
            if not isinstance(entry, Mapping):
                continue
            if entry.get("owner") != owner:
                continue
            mint = entry.get("mint")
            amount_info = entry.get("uiTokenAmount")
            if not isinstance(mint, str) or not isinstance(amount_info, Mapping):
                continue
            raw_amount = amount_info.get("amount")
            if raw_amount is None:
                continue
            try:
                amount = int(str(raw_amount))
            except (TypeError, ValueError) as exc:
                raise FillReconciliationError("invalid token balance amount") from exc
            totals[mint] = totals.get(mint, 0) + amount
        return totals


def observed_fill_to_receipt(
    fill: ObservedSwapFill,
    *,
    quote: Quote,
    execution_id: str,
    submitted_at: datetime,
    confirmed_at: datetime,
    mode: str = "solana-observed",
) -> ExecutionReceipt:
    if quote.side is not fill.side:
        raise FillReconciliationError("quote side does not match observed fill")
    if quote.price_usd <= 0:
        raise FillReconciliationError("quote price must be positive")

    if fill.side is TradeSide.BUY:
        adverse = (fill.effective_price_usd - quote.price_usd) / quote.price_usd
    else:
        adverse = (quote.price_usd - fill.effective_price_usd) / quote.price_usd
    adverse_bps = int(
        (max(Decimal("0"), adverse) * Decimal("10000")).to_integral_value(
            rounding=ROUND_CEILING
        )
    )

    return ExecutionReceipt(
        execution_id=execution_id,
        quote_id=quote.quote_id,
        mode=mode,
        state=TradeState.CONFIRMED,
        external_ref=fill.signature,
        average_price_usd=fill.effective_price_usd,
        filled_base_amount=fill.base_amount,
        filled_quote_usd=fill.quote_amount,
        # USDC delta already represents the wallet's actual swap cashflow.
        # SOL network fees are exposed separately on ObservedSwapFill until an
        # independently sourced SOL/USD conversion is added.
        fee_usd=Decimal("0"),
        actual_slippage_bps=adverse_bps,
        submitted_at=submitted_at,
        confirmed_at=confirmed_at,
    )
