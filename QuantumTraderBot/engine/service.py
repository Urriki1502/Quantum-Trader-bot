from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .adapters import (
    DeterministicExecutionFailure,
    ExecutionAdapter,
    QuoteProvider,
    UnknownExecutionOutcome,
)
from .ledger import DuplicateIntentError, SQLiteTradeLedger
from .models import TradeIntent, TradeState
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
            reason = ",".join(pre.reasons)
            self.ledger.set_error(intent.intent_id, reason)
            self.ledger.transition(intent.intent_id, TradeState.REJECTED, reason=reason)
            return self._result(intent.intent_id, reasons=pre.reasons)

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
            reason = ",".join(quote_risk.reasons)
            self.ledger.set_error(intent.intent_id, reason)
            self.ledger.transition(intent.intent_id, TradeState.REJECTED, reason=reason)
            return self._result(intent.intent_id, reasons=quote_risk.reasons)

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
        self.ledger.transition(
            intent.intent_id,
            TradeState.RECONCILED,
            reason="execution_receipt_reconciled",
        )
        return self._result(intent.intent_id)

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
