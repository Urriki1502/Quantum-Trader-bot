from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .adapters import (
    DeterministicExecutionFailure,
    ExecutionAdapter,
    QuoteProvider,
    UnknownExecutionOutcome,
)
from .ledger import (
    DuplicateIntentError,
    InsufficientPositionError,
    ReconciliationError,
    SQLiteTradeLedger,
)
from .models import TradeIntent, TradeSide, TradeState
from .risk import RiskEngine, RiskSnapshot


@dataclass(frozen=True, slots=True)
class EngineResult:
    intent_id: str
    state: TradeState
    existing: bool
    record: dict[str, Any]
    reasons: tuple[str, ...] = ()


class TradingEngine:
    """Provider-agnostic execution coordinator.

    Exactly-once rule:
      * intent_id is durable and unique.
      * duplicate calls never re-execute an existing intent.
      * ambiguous submission becomes UNKNOWN, never an automatic retry.
      * confirmed executions are reconciled into the portfolio atomically.
    """

    def __init__(
        self,
        *,
        quote_provider: QuoteProvider,
        execution_adapter: ExecutionAdapter,
        risk_engine: RiskEngine,
        ledger: SQLiteTradeLedger,
    ) -> None:
        self.quote_provider = quote_provider
        self.execution_adapter = execution_adapter
        self.risk_engine = risk_engine
        self.ledger = ledger

    async def execute_intent(
        self,
        intent: TradeIntent,
        snapshot: RiskSnapshot,
    ) -> EngineResult:
        try:
            self.ledger.create_intent(intent)
        except DuplicateIntentError:
            existing = self.ledger.get_trade(intent.intent_id)
            if existing is None:
                raise RuntimeError("duplicate intent exists but ledger row is missing")
            return EngineResult(
                intent_id=intent.intent_id,
                state=TradeState(existing["state"]),
                existing=True,
                record=existing,
            )

        pre = self.risk_engine.evaluate_intent(intent, snapshot)
        if not pre.allowed:
            return self._reject(intent.intent_id, pre.reasons)

        self.ledger.transition(
            intent.intent_id,
            TradeState.RISK_APPROVED,
            reason="pre_trade_risk_pass",
        )

        try:
            quote = await self.quote_provider.quote(intent)
        except Exception as exc:
            self.ledger.set_error(intent.intent_id, f"quote_error:{exc}")
            self.ledger.transition(intent.intent_id, TradeState.FAILED, reason="quote_error")
            return self._result(intent.intent_id, reasons=("quote_error",))

        self.ledger.attach_quote(intent.intent_id, quote)
        self.ledger.transition(intent.intent_id, TradeState.QUOTED, reason=quote.provider)

        quote_risk = self.risk_engine.evaluate_quote(intent, quote)
        if not quote_risk.allowed:
            return self._reject(intent.intent_id, quote_risk.reasons)

        if intent.side is TradeSide.SELL:
            position = self.ledger.get_position(intent.asset)
            if quote.estimated_base_amount > position.quantity:
                return self._reject(
                    intent.intent_id,
                    ("insufficient_position",),
                    error=(
                        f"sell quantity {quote.estimated_base_amount} exceeds "
                        f"position {position.quantity}"
                    ),
                )

        self.ledger.transition(
            intent.intent_id,
            TradeState.EXECUTION_PENDING,
            reason=f"adapter:{self.execution_adapter.mode}",
        )

        try:
            receipt = await self.execution_adapter.execute(intent, quote)
        except UnknownExecutionOutcome as exc:
            message = f"unknown_execution_outcome:{exc}"
            if exc.external_ref:
                message += f":{exc.external_ref}"
            self.ledger.set_error(intent.intent_id, message)
            self.ledger.transition(intent.intent_id, TradeState.UNKNOWN, reason=message)
            return self._result(intent.intent_id, reasons=("unknown_execution_outcome",))
        except DeterministicExecutionFailure as exc:
            self.ledger.set_error(intent.intent_id, str(exc))
            self.ledger.transition(intent.intent_id, TradeState.FAILED, reason=str(exc))
            return self._result(intent.intent_id, reasons=("execution_failed",))
        except Exception as exc:
            message = f"unexpected_execution_exception:{type(exc).__name__}:{exc}"
            self.ledger.set_error(intent.intent_id, message)
            self.ledger.transition(intent.intent_id, TradeState.UNKNOWN, reason=message)
            return self._result(intent.intent_id, reasons=("unknown_execution_outcome",))

        self.ledger.attach_execution(intent.intent_id, receipt)
        self.ledger.transition(
            intent.intent_id,
            TradeState.SUBMITTED,
            reason=receipt.external_ref,
        )

        if receipt.state is TradeState.SUBMITTED:
            return self._result(intent.intent_id)

        self.ledger.transition(
            intent.intent_id,
            TradeState.CONFIRMED,
            reason=receipt.external_ref,
        )
        try:
            self.ledger.reconcile_confirmed_execution(intent, receipt)
        except (ReconciliationError, InsufficientPositionError) as exc:
            # The external execution is already confirmed; never rewrite this as FAILED.
            # Leave the durable state at CONFIRMED for deterministic startup recovery.
            self.ledger.set_error(intent.intent_id, f"reconciliation_required:{exc}")
            return self._result(
                intent.intent_id,
                reasons=("reconciliation_required",),
            )
        return self._result(intent.intent_id)

    def recover_confirmed(self) -> list[EngineResult]:
        """Recover accounting after a crash between confirmation and reconciliation."""
        results: list[EngineResult] = []
        for row in self.ledger.list_trades_by_state(TradeState.CONFIRMED):
            intent_id = row["intent_id"]
            try:
                intent = self.ledger.load_intent(intent_id)
                receipt = self.ledger.load_execution_receipt(intent_id)
                if receipt.state is not TradeState.CONFIRMED:
                    raise ReconciliationError("stored confirmed trade lacks confirmed receipt")
                self.ledger.reconcile_confirmed_execution(intent, receipt)
                results.append(self._result(intent_id))
            except ReconciliationError as exc:
                self.ledger.set_error(intent_id, f"reconciliation_required:{exc}")
                results.append(
                    self._result(intent_id, reasons=("reconciliation_required",))
                )
        return results

    def risk_snapshot(
        self,
        *,
        realized_pnl_today_usd: Decimal | str | int | float | None = None,
        trading_enabled: bool = True,
        data_fresh: bool = True,
    ) -> RiskSnapshot:
        """Build a deterministic snapshot from persisted portfolio accounting.

        Open exposure is cost-basis exposure, not an optimistic mark-to-market
        estimate. A future market-data adapter may supply a separate marked
        exposure metric after it is independently validated.
        """
        realized = (
            Decimal(str(realized_pnl_today_usd))
            if realized_pnl_today_usd is not None
            else self.ledger.realized_pnl_today_utc()
        )
        exposure = self.ledger.open_cost_basis_exposure()
        account = self.ledger.get_paper_account()
        return RiskSnapshot(
            open_exposure_usd=exposure,
            realized_pnl_today_usd=realized,
            available_cash_usd=account.cash_usd if account is not None else None,
            trading_enabled=trading_enabled,
            data_fresh=data_fresh,
        )

    def _reject(
        self,
        intent_id: str,
        reasons: tuple[str, ...],
        *,
        error: str | None = None,
    ) -> EngineResult:
        message = error or ",".join(reasons)
        self.ledger.set_error(intent_id, message)
        self.ledger.transition(intent_id, TradeState.REJECTED, reason=message)
        return self._result(intent_id, reasons=reasons)

    def _result(self, intent_id: str, reasons: tuple[str, ...] = ()) -> EngineResult:
        record = self.ledger.get_trade(intent_id)
        if record is None:
            raise RuntimeError(f"ledger lost intent {intent_id}")
        return EngineResult(
            intent_id=intent_id,
            state=TradeState(record["state"]),
            existing=False,
            record=record,
            reasons=reasons,
        )
