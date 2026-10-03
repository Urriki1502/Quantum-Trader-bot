from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from typing import Protocol
from uuid import uuid4

from .models import ExecutionReceipt, Quote, TradeIntent, TradeSide, TradeState, utc_now


class QuoteProvider(Protocol):
    async def quote(self, intent: TradeIntent) -> Quote:
        ...


class ExecutionAdapter(Protocol):
    mode: str

    async def execute(self, intent: TradeIntent, quote: Quote) -> ExecutionReceipt:
        ...


class UnknownExecutionOutcome(RuntimeError):
    """Raised only when submission may have happened but the result is unknown.

    The engine records UNKNOWN and must not blindly retry the intent.
    """

    def __init__(self, message: str, *, external_ref: str = "") -> None:
        super().__init__(message)
        self.external_ref = external_ref


class DeterministicExecutionFailure(RuntimeError):
    """A fail-closed execution failure where no transaction was accepted."""


@dataclass(slots=True)
class StaticQuoteProvider:
    """Deterministic no-network provider for tests, replay and paper mode."""

    prices_usd: dict[str, Decimal]
    liquidity_usd: Decimal = Decimal("1000000")
    price_impact_bps: int = 25
    fee_bps: int = 25
    ttl_seconds: int = 10
    provider_name: str = "static-paper"

    def __post_init__(self) -> None:
        self.prices_usd = {k: Decimal(str(v)) for k, v in self.prices_usd.items()}
        self.liquidity_usd = Decimal(str(self.liquidity_usd))

    async def quote(self, intent: TradeIntent) -> Quote:
        if intent.asset not in self.prices_usd:
            raise DeterministicExecutionFailure(f"no price configured for {intent.asset}")
        price = self.prices_usd[intent.asset]
        if price <= 0:
            raise DeterministicExecutionFailure("price must be positive")

        now = utc_now()
        base_amount = intent.notional_usd / price
        fee = intent.notional_usd * Decimal(self.fee_bps) / Decimal(10000)
        return Quote(
            quote_id=str(uuid4()),
            provider=self.provider_name,
            asset=intent.asset,
            side=intent.side,
            price_usd=price,
            estimated_base_amount=base_amount,
            notional_usd=intent.notional_usd,
            liquidity_usd=self.liquidity_usd,
            price_impact_bps=self.price_impact_bps,
            estimated_fee_usd=fee,
            created_at=now,
            expires_at=now + timedelta(seconds=self.ttl_seconds),
        )


@dataclass(slots=True)
class PaperExecutionAdapter:
    """Paper execution using the same lifecycle contract intended for live adapters."""

    slippage_bps: int = 10
    fee_bps: int = 25
    mode: str = "paper"

    async def execute(self, intent: TradeIntent, quote: Quote) -> ExecutionReceipt:
        if self.slippage_bps > intent.max_slippage_bps:
            raise DeterministicExecutionFailure("paper slippage exceeds intent limit")
        if utc_now() >= quote.expires_at:
            raise DeterministicExecutionFailure("quote expired before paper execution")

        slippage = Decimal(self.slippage_bps) / Decimal(10000)
        if intent.side is TradeSide.BUY:
            average_price = quote.price_usd * (Decimal("1") + slippage)
            filled_quote = intent.notional_usd
            filled_base = filled_quote / average_price
        else:
            average_price = quote.price_usd * (Decimal("1") - slippage)
            filled_base = quote.estimated_base_amount
            filled_quote = filled_base * average_price

        fee = filled_quote * Decimal(self.fee_bps) / Decimal(10000)
        submitted_at = utc_now()
        seed = f"{intent.intent_id}|{quote.quote_id}|{self.mode}".encode()
        external_ref = "paper:" + hashlib.sha256(seed).hexdigest()[:32]

        return ExecutionReceipt(
            execution_id=str(uuid4()),
            quote_id=quote.quote_id,
            mode=self.mode,
            state=TradeState.CONFIRMED,
            external_ref=external_ref,
            average_price_usd=average_price,
            filled_base_amount=filled_base,
            filled_quote_usd=filled_quote,
            fee_usd=fee,
            actual_slippage_bps=self.slippage_bps,
            submitted_at=submitted_at,
            confirmed_at=utc_now(),
        )
