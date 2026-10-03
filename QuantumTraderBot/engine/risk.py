from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from .models import Quote, TradeIntent, TradeSide


@dataclass(frozen=True, slots=True)
class RiskPolicy:
    max_notional_usd: Decimal = Decimal("100")
    max_open_exposure_usd: Decimal = Decimal("500")
    min_liquidity_usd: Decimal = Decimal("10000")
    max_price_impact_bps: int = 300
    max_slippage_bps: int = 300
    max_daily_loss_usd: Decimal = Decimal("100")
    max_quote_age_seconds: int = 20
    buy_fee_reserve_bps: int = 100

    def __post_init__(self) -> None:
        if self.buy_fee_reserve_bps < 0:
            raise ValueError("buy_fee_reserve_bps must be >= 0")
        for name in (
            "max_notional_usd",
            "max_open_exposure_usd",
            "min_liquidity_usd",
            "max_daily_loss_usd",
        ):
            object.__setattr__(self, name, Decimal(str(getattr(self, name))))


@dataclass(frozen=True, slots=True)
class RiskSnapshot:
    open_exposure_usd: Decimal = Decimal("0")
    realized_pnl_today_usd: Decimal = Decimal("0")
    available_cash_usd: Decimal | None = None
    trading_enabled: bool = True
    data_fresh: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "open_exposure_usd", Decimal(str(self.open_exposure_usd)))
        object.__setattr__(self, "realized_pnl_today_usd", Decimal(str(self.realized_pnl_today_usd)))
        if self.available_cash_usd is not None:
            object.__setattr__(
                self,
                "available_cash_usd",
                Decimal(str(self.available_cash_usd)),
            )


@dataclass(frozen=True, slots=True)
class RiskDecision:
    allowed: bool
    reasons: tuple[str, ...] = ()


class RiskEngine:
    """Deterministic, fail-closed pre-trade risk gate."""

    def __init__(self, policy: RiskPolicy) -> None:
        self.policy = policy

    def evaluate_intent(self, intent: TradeIntent, snapshot: RiskSnapshot) -> RiskDecision:
        reasons: list[str] = []

        if not snapshot.trading_enabled:
            reasons.append("trading_disabled")
        if not snapshot.data_fresh:
            reasons.append("market_data_stale")
        if intent.notional_usd <= 0:
            reasons.append("notional_must_be_positive")
        if intent.notional_usd > self.policy.max_notional_usd:
            reasons.append("max_notional_exceeded")
        if snapshot.open_exposure_usd + intent.notional_usd > self.policy.max_open_exposure_usd:
            reasons.append("max_open_exposure_exceeded")
        if snapshot.realized_pnl_today_usd <= -self.policy.max_daily_loss_usd:
            reasons.append("daily_loss_limit_reached")
        if intent.side is TradeSide.BUY and snapshot.available_cash_usd is not None:
            reserve = (
                intent.notional_usd
                * Decimal(self.policy.buy_fee_reserve_bps)
                / Decimal("10000")
            )
            if intent.notional_usd + reserve > snapshot.available_cash_usd:
                reasons.append("insufficient_cash")
        if intent.max_slippage_bps > self.policy.max_slippage_bps:
            reasons.append("requested_slippage_exceeds_policy")

        return RiskDecision(allowed=not reasons, reasons=tuple(reasons))

    def evaluate_quote(
        self,
        intent: TradeIntent,
        quote: Quote,
        *,
        now: datetime | None = None,
    ) -> RiskDecision:
        now = now or datetime.now(timezone.utc)
        reasons: list[str] = []

        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        if quote.asset != intent.asset or quote.side != intent.side:
            reasons.append("quote_intent_mismatch")
        if quote.notional_usd != intent.notional_usd:
            reasons.append("quote_notional_mismatch")
        if now >= quote.expires_at:
            reasons.append("quote_expired")
        quote_age = (now - quote.created_at).total_seconds()
        if quote_age < 0 or quote_age > self.policy.max_quote_age_seconds:
            reasons.append("quote_stale")
        if quote.liquidity_usd < self.policy.min_liquidity_usd:
            reasons.append("insufficient_liquidity")
        if quote.price_impact_bps > self.policy.max_price_impact_bps:
            reasons.append("price_impact_too_high")
        if quote.price_impact_bps > intent.max_slippage_bps:
            reasons.append("price_impact_exceeds_requested_slippage")
        if quote.price_usd <= 0 or quote.estimated_base_amount <= 0:
            reasons.append("invalid_quote_amounts")

        return RiskDecision(allowed=not reasons, reasons=tuple(reasons))
