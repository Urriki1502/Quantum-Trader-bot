from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Mapping
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TradeSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class TradeState(str, Enum):
    CREATED = "created"
    REJECTED = "rejected"
    RISK_APPROVED = "risk_approved"
    QUOTED = "quoted"
    EXECUTION_PENDING = "execution_pending"
    SUBMITTED = "submitted"
    UNKNOWN = "unknown"
    CONFIRMED = "confirmed"
    RECONCILED = "reconciled"
    FAILED = "failed"
    EXPIRED = "expired"


_ALLOWED_TRANSITIONS: dict[TradeState, frozenset[TradeState]] = {
    TradeState.CREATED: frozenset({
        TradeState.RISK_APPROVED,
        TradeState.REJECTED,
        TradeState.FAILED,
    }),
    TradeState.RISK_APPROVED: frozenset({
        TradeState.QUOTED,
        TradeState.REJECTED,
        TradeState.FAILED,
    }),
    TradeState.QUOTED: frozenset({
        TradeState.EXECUTION_PENDING,
        TradeState.REJECTED,
        TradeState.EXPIRED,
        TradeState.FAILED,
    }),
    TradeState.EXECUTION_PENDING: frozenset({
        TradeState.SUBMITTED,
        TradeState.UNKNOWN,
        TradeState.EXPIRED,
        TradeState.FAILED,
    }),
    TradeState.SUBMITTED: frozenset({
        TradeState.CONFIRMED,
        TradeState.UNKNOWN,
        TradeState.EXPIRED,
        TradeState.FAILED,
    }),
    TradeState.UNKNOWN: frozenset({
        TradeState.SUBMITTED,
        TradeState.CONFIRMED,
        TradeState.EXPIRED,
        TradeState.FAILED,
    }),
    TradeState.CONFIRMED: frozenset({
        TradeState.RECONCILED,
    }),
    TradeState.REJECTED: frozenset(),
    TradeState.RECONCILED: frozenset(),
    TradeState.FAILED: frozenset(),
    TradeState.EXPIRED: frozenset(),
}


class InvalidStateTransition(ValueError):
    pass


def assert_transition(current: TradeState, target: TradeState) -> None:
    if target not in _ALLOWED_TRANSITIONS[current]:
        raise InvalidStateTransition(f"invalid trade transition: {current.value} -> {target.value}")


@dataclass(frozen=True, slots=True)
class TradeIntent:
    intent_id: str
    asset: str
    side: TradeSide
    notional_usd: Decimal
    max_slippage_bps: int
    created_at: datetime = field(default_factory=utc_now)
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "notional_usd", Decimal(str(self.notional_usd)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        if not self.asset.strip():
            raise ValueError("asset must be non-empty")
        if self.max_slippage_bps < 0:
            raise ValueError("max_slippage_bps must be >= 0")

    @classmethod
    def create(
        cls,
        *,
        asset: str,
        side: TradeSide,
        notional_usd: Decimal | str | int | float,
        max_slippage_bps: int,
        metadata: Mapping[str, str] | None = None,
    ) -> "TradeIntent":
        return cls(
            intent_id=str(uuid4()),
            asset=asset,
            side=side,
            notional_usd=Decimal(str(notional_usd)),
            max_slippage_bps=max_slippage_bps,
            metadata=metadata or {},
        )


@dataclass(frozen=True, slots=True)
class Quote:
    quote_id: str
    provider: str
    asset: str
    side: TradeSide
    price_usd: Decimal
    estimated_base_amount: Decimal
    notional_usd: Decimal
    liquidity_usd: Decimal
    price_impact_bps: int
    estimated_fee_usd: Decimal
    created_at: datetime
    expires_at: datetime

    def __post_init__(self) -> None:
        for name in (
            "price_usd",
            "estimated_base_amount",
            "notional_usd",
            "liquidity_usd",
            "estimated_fee_usd",
        ):
            object.__setattr__(self, name, Decimal(str(getattr(self, name))))
        if self.created_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("quote timestamps must be timezone-aware")
        if self.expires_at <= self.created_at:
            raise ValueError("quote expires_at must be after created_at")


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    execution_id: str
    quote_id: str
    mode: str
    state: TradeState
    external_ref: str
    average_price_usd: Decimal
    filled_base_amount: Decimal
    filled_quote_usd: Decimal
    fee_usd: Decimal
    actual_slippage_bps: int
    submitted_at: datetime
    confirmed_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in (
            "average_price_usd",
            "filled_base_amount",
            "filled_quote_usd",
            "fee_usd",
        ):
            object.__setattr__(self, name, Decimal(str(getattr(self, name))))
        if self.submitted_at.tzinfo is None:
            raise ValueError("submitted_at must be timezone-aware")
        if self.confirmed_at is not None and self.confirmed_at.tzinfo is None:
            raise ValueError("confirmed_at must be timezone-aware")
        if self.state not in {TradeState.SUBMITTED, TradeState.CONFIRMED}:
            raise ValueError("execution receipt state must be submitted or confirmed")
