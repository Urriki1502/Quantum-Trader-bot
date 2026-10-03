from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import AsyncIterator, Mapping
from uuid import uuid4

from .market import MarketTick
from .models import Quote, TradeIntent
from .report import PerformanceReport, build_performance_report
from .session import PaperTradingSession, SessionDecision


class ReplayFormatError(ValueError):
    pass


@dataclass(slots=True)
class ReplayQuoteProvider:
    """Deterministic quote provider driven only by the current replay tick."""

    fee_bps: int = 25
    ttl_seconds: int = 3600
    provider_name: str = "replay-market"
    _ticks: dict[str, MarketTick] | None = None

    def __post_init__(self) -> None:
        self._ticks = {}

    def update(self, tick: MarketTick) -> None:
        assert self._ticks is not None
        self._ticks[tick.asset] = tick

    async def quote(self, intent: TradeIntent) -> Quote:
        assert self._ticks is not None
        tick = self._ticks.get(intent.asset)
        if tick is None:
            raise ReplayFormatError(f"no replay tick available for {intent.asset}")
        base_amount = intent.notional_usd / tick.price_usd
        fee = intent.notional_usd * Decimal(self.fee_bps) / Decimal("10000")
        return Quote(
            quote_id=str(uuid4()),
            provider=self.provider_name,
            asset=intent.asset,
            side=intent.side,
            price_usd=tick.price_usd,
            estimated_base_amount=base_amount,
            notional_usd=intent.notional_usd,
            liquidity_usd=tick.liquidity_usd,
            price_impact_bps=tick.price_impact_bps,
            estimated_fee_usd=fee,
            created_at=tick.observed_at,
            expires_at=tick.observed_at + timedelta(seconds=self.ttl_seconds),
        )


@dataclass(slots=True)
class JsonlMarketReplay:
    path: str | Path

    async def stream(self) -> AsyncIterator[MarketTick]:
        previous_time: datetime | None = None
        with Path(self.path).open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ReplayFormatError(
                        f"line {line_number}: invalid JSON"
                    ) from exc
                if not isinstance(payload, Mapping):
                    raise ReplayFormatError(
                        f"line {line_number}: event must be a JSON object"
                    )

                try:
                    asset = str(payload["asset"])
                    price = Decimal(str(payload["price_usd"]))
                    liquidity = Decimal(str(payload["liquidity_usd"]))
                    impact = int(payload.get("price_impact_bps", 0))
                    observed_at = datetime.fromisoformat(
                        str(payload["observed_at"]).replace("Z", "+00:00")
                    )
                    source = str(payload.get("source", "jsonl-replay"))
                except (KeyError, ValueError, TypeError) as exc:
                    raise ReplayFormatError(
                        f"line {line_number}: invalid market event fields"
                    ) from exc

                if observed_at.tzinfo is None:
                    raise ReplayFormatError(
                        f"line {line_number}: observed_at must include timezone"
                    )
                if previous_time is not None and observed_at < previous_time:
                    raise ReplayFormatError(
                        f"line {line_number}: replay timestamps must be monotonic"
                    )
                previous_time = observed_at

                event_id = payload.get("event_id")
                if not event_id:
                    canonical = (
                        f"{asset}|{price}|{liquidity}|{impact}|"
                        f"{observed_at.isoformat()}|{source}"
                    ).encode()
                    event_id = hashlib.sha256(canonical).hexdigest()

                yield MarketTick(
                    event_id=str(event_id),
                    asset=asset,
                    price_usd=price,
                    liquidity_usd=liquidity,
                    price_impact_bps=impact,
                    observed_at=observed_at,
                    source=source,
                )


@dataclass(frozen=True, slots=True)
class ReplaySummary:
    ticks: int
    duplicate_events: int
    buy_signals: int
    sell_signals: int
    rejected_trades: int
    executed_trades: int
    report: PerformanceReport

    def as_json(self, *, indent: int = 2) -> str:
        payload = json.loads(self.report.as_json(indent=indent))
        payload.update(
            {
                "ticks": self.ticks,
                "duplicate_events": self.duplicate_events,
                "buy_signals": self.buy_signals,
                "sell_signals": self.sell_signals,
                "rejected_trades": self.rejected_trades,
                "executed_trades": self.executed_trades,
            }
        )
        return json.dumps(payload, indent=indent, sort_keys=True)


async def run_replay(
    *,
    source: JsonlMarketReplay,
    session: PaperTradingSession,
    quote_provider: ReplayQuoteProvider,
) -> ReplaySummary:
    ticks = 0
    duplicate_events = 0
    buy_signals = 0
    sell_signals = 0
    rejected = 0
    executed = 0
    latest_prices: dict[str, Decimal] = {}

    async for tick in source.stream():
        ticks += 1
        latest_prices[tick.asset] = tick.price_usd
        quote_provider.update(tick)
        decision: SessionDecision = await session.process_tick(tick, replay=True)

        if decision.duplicate_event:
            duplicate_events += 1
        if decision.signal.action.value == "buy":
            buy_signals += 1
        elif decision.signal.action.value == "sell":
            sell_signals += 1

        if decision.engine_result is not None:
            if decision.engine_result.state.value == "rejected":
                rejected += 1
            elif decision.engine_result.state.value == "reconciled":
                executed += 1

        # Persist the equity curve after each replay event. If an open position
        # exists for an asset with no mark yet, fail rather than invent a value.
        session.engine.ledger.mark_to_market(latest_prices)

    if ticks == 0:
        raise ReplayFormatError("replay input contained no market events")

    report = build_performance_report(
        session.engine.ledger,
        prices_usd=latest_prices,
    )
    return ReplaySummary(
        ticks=ticks,
        duplicate_events=duplicate_events,
        buy_signals=buy_signals,
        sell_signals=sell_signals,
        rejected_trades=rejected,
        executed_trades=executed,
        report=report,
    )
