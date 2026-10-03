from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import AsyncIterator, Protocol

from .models import TradeIntent, TradeSide
from .raydium_quote_provider import RaydiumUsdcQuoteProvider


@dataclass(frozen=True, slots=True)
class MarketTick:
    event_id: str
    asset: str
    price_usd: Decimal
    liquidity_usd: Decimal
    price_impact_bps: int
    observed_at: datetime
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "price_usd", Decimal(str(self.price_usd)))
        object.__setattr__(self, "liquidity_usd", Decimal(str(self.liquidity_usd)))
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        if self.price_usd <= 0:
            raise ValueError("price_usd must be positive")
        if self.liquidity_usd < 0:
            raise ValueError("liquidity_usd must be >= 0")


class MarketSource(Protocol):
    async def stream(self, *, limit: int | None = None) -> AsyncIterator[MarketTick]:
        ...


@dataclass(slots=True)
class RaydiumPollingMarketSource:
    quote_provider: RaydiumUsdcQuoteProvider
    asset: str
    probe_notional_usd: Decimal = Decimal("10")
    max_slippage_bps: int = 100
    interval_seconds: float = 30.0

    def __post_init__(self) -> None:
        self.probe_notional_usd = Decimal(str(self.probe_notional_usd))
        if self.probe_notional_usd <= 0:
            raise ValueError("probe_notional_usd must be positive")
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")

    async def stream(self, *, limit: int | None = None) -> AsyncIterator[MarketTick]:
        emitted = 0
        while limit is None or emitted < limit:
            probe = TradeIntent.create(
                asset=self.asset,
                side=TradeSide.BUY,
                notional_usd=self.probe_notional_usd,
                max_slippage_bps=self.max_slippage_bps,
                metadata={"purpose": "market_probe"},
            )
            quote = await self.quote_provider.quote(probe)
            event_seed = (
                f"{quote.provider}|{quote.asset}|{quote.created_at.isoformat()}|"
                f"{quote.price_usd}|{quote.liquidity_usd}|{quote.price_impact_bps}"
            ).encode()
            event_id = hashlib.sha256(event_seed).hexdigest()
            yield MarketTick(
                event_id=event_id,
                asset=quote.asset,
                price_usd=quote.price_usd,
                liquidity_usd=quote.liquidity_usd,
                price_impact_bps=quote.price_impact_bps,
                observed_at=quote.created_at,
                source=quote.provider,
            )
            emitted += 1
            if limit is None or emitted < limit:
                await asyncio.sleep(self.interval_seconds)
