from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Iterable, Protocol

from .market import MarketTick


class SignalAction(str, Enum):
    HOLD = "hold"
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class StrategyContext:
    position_quantity: Decimal
    available_cash_usd: Decimal | None
    realized_pnl_today_usd: Decimal


@dataclass(frozen=True, slots=True)
class StrategySignal:
    action: SignalAction
    reason: str
    confidence: Decimal = Decimal("0")


class Strategy(Protocol):
    strategy_id: str

    def on_tick(self, tick: MarketTick, context: StrategyContext) -> StrategySignal:
        ...

    def prime(self, ticks: Iterable[MarketTick]) -> None:
        ...


@dataclass(slots=True)
class MovingAverageCrossStrategy:
    """Minimal deterministic reference strategy for engine validation.

    This is deliberately simple. It exists to exercise the event/execution
    pipeline, not to claim predictive edge.
    """

    short_window: int = 3
    long_window: int = 8
    strategy_id: str = "ma-cross-v1"
    _prices: deque[Decimal] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.short_window <= 0 or self.long_window <= 0:
            raise ValueError("moving-average windows must be positive")
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be smaller than long_window")
        self._prices = deque(maxlen=self.long_window + 1)

    def prime(self, ticks: Iterable[MarketTick]) -> None:
        for tick in ticks:
            self._prices.append(tick.price_usd)

    def on_tick(self, tick: MarketTick, context: StrategyContext) -> StrategySignal:
        previous = list(self._prices)
        self._prices.append(tick.price_usd)
        current = list(self._prices)

        if len(current) < self.long_window:
            return StrategySignal(SignalAction.HOLD, "warming_up")

        current_short = sum(current[-self.short_window :], Decimal("0")) / Decimal(
            self.short_window
        )
        current_long = sum(current[-self.long_window :], Decimal("0")) / Decimal(
            self.long_window
        )

        if len(previous) < self.long_window:
            return StrategySignal(SignalAction.HOLD, "baseline_ready")

        previous_short = sum(
            previous[-self.short_window :], Decimal("0")
        ) / Decimal(self.short_window)
        previous_long = sum(
            previous[-self.long_window :], Decimal("0")
        ) / Decimal(self.long_window)

        crossed_up = previous_short <= previous_long and current_short > current_long
        crossed_down = previous_short >= previous_long and current_short < current_long

        if crossed_up and context.position_quantity <= 0:
            spread = (current_short - current_long) / current_long
            return StrategySignal(
                SignalAction.BUY,
                "short_ma_crossed_above_long_ma",
                confidence=max(Decimal("0"), min(Decimal("1"), spread * Decimal("100"))),
            )

        if crossed_down and context.position_quantity > 0:
            spread = (current_long - current_short) / current_long
            return StrategySignal(
                SignalAction.SELL,
                "short_ma_crossed_below_long_ma",
                confidence=max(Decimal("0"), min(Decimal("1"), spread * Decimal("100"))),
            )

        return StrategySignal(SignalAction.HOLD, "no_cross")
