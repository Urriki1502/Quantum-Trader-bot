from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from .journal import SQLiteMarketJournal
from .market import MarketTick
from .models import TradeIntent, TradeSide
from .service import EngineResult, TradingEngine
from .signals import SignalAction, Strategy, StrategyContext, StrategySignal


@dataclass(frozen=True, slots=True)
class SessionDecision:
    tick: MarketTick
    signal: StrategySignal
    engine_result: EngineResult | None
    duplicate_event: bool = False


class PaperTradingSession:
    """Event-driven paper session using the same TradingEngine as future live mode."""

    def __init__(
        self,
        *,
        engine: TradingEngine,
        journal: SQLiteMarketJournal,
        strategy: Strategy,
        buy_notional_usd: Decimal | str | int | float,
        sell_notional_usd: Decimal | str | int | float | None = None,
        max_slippage_bps: int = 100,
        max_tick_age_seconds: int = 90,
    ) -> None:
        self.engine = engine
        self.journal = journal
        self.strategy = strategy
        self.buy_notional_usd = Decimal(str(buy_notional_usd))
        self.sell_notional_usd = Decimal(
            str(sell_notional_usd if sell_notional_usd is not None else buy_notional_usd)
        )
        self.max_slippage_bps = max_slippage_bps
        self.max_tick_age_seconds = max_tick_age_seconds
        if self.buy_notional_usd <= 0 or self.sell_notional_usd <= 0:
            raise ValueError("paper notionals must be positive")

    def prime_from_journal(self, asset: str, *, limit: int) -> None:
        self.strategy.prime(self.journal.recent(asset, limit=limit))

    async def process_tick(
        self,
        tick: MarketTick,
        *,
        replay: bool = False,
    ) -> SessionDecision:
        if not self.journal.record(tick):
            return SessionDecision(
                tick=tick,
                signal=StrategySignal(SignalAction.HOLD, "duplicate_event"),
                engine_result=None,
                duplicate_event=True,
            )

        position = self.engine.ledger.get_position(tick.asset)
        account = self.engine.ledger.get_paper_account()
        context = StrategyContext(
            position_quantity=position.quantity,
            available_cash_usd=account.cash_usd if account else None,
            realized_pnl_today_usd=self.engine.ledger.realized_pnl_today_utc(),
        )
        signal = self.strategy.on_tick(tick, context)
        if signal.action is SignalAction.HOLD:
            return SessionDecision(tick=tick, signal=signal, engine_result=None)

        side = TradeSide.BUY if signal.action is SignalAction.BUY else TradeSide.SELL
        if side is TradeSide.BUY:
            notional = self.buy_notional_usd
        else:
            marked_position = position.quantity * tick.price_usd
            # Leave a small mark-to-quote buffer to avoid accidental paper oversell
            # when the execution quote differs slightly from the signal tick.
            max_sell = marked_position * Decimal("0.95")
            notional = min(self.sell_notional_usd, max_sell)
            if notional <= 0:
                return SessionDecision(
                    tick=tick,
                    signal=StrategySignal(SignalAction.HOLD, "no_sellable_position"),
                    engine_result=None,
                )

        intent_id = self._intent_id(tick.event_id, signal.action)
        intent = TradeIntent(
            intent_id=intent_id,
            asset=tick.asset,
            side=side,
            notional_usd=notional,
            max_slippage_bps=self.max_slippage_bps,
            created_at=tick.observed_at,
            metadata={
                "market_event_id": tick.event_id,
                "strategy_id": self.strategy.strategy_id,
                "signal_reason": signal.reason,
                "mode": "paper",
            },
        )

        now = datetime.now(timezone.utc)
        age_seconds = (now - tick.observed_at).total_seconds()
        data_fresh = replay or (0 <= age_seconds <= self.max_tick_age_seconds)
        result = await self.engine.execute_intent(
            intent,
            self.engine.risk_snapshot(
                data_fresh=data_fresh,
                evaluation_time=tick.observed_at if replay else None,
            ),
        )
        return SessionDecision(tick=tick, signal=signal, engine_result=result)

    def _intent_id(self, event_id: str, action: SignalAction) -> str:
        seed = f"{self.strategy.strategy_id}|{event_id}|{action.value}".encode()
        return "paper-v2:" + hashlib.sha256(seed).hexdigest()
