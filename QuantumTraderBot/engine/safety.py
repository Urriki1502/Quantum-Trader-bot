from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .ledger import RuntimeSafetySnapshot
from .models import TradeIntent


class LiveSafetyViolation(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LiveExecutionPolicy:
    """Hard policy boundary for any future real-money path.

    The policy is intentionally independent from strategy configuration. A
    strategy cannot raise these limits at runtime.
    """

    max_notional_usd: Decimal = Decimal("5")
    allowed_assets: frozenset[str] = frozenset()
    max_transactions_per_intent: int = 1
    require_preflight: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_notional_usd",
            Decimal(str(self.max_notional_usd)),
        )
        object.__setattr__(
            self,
            "allowed_assets",
            frozenset(str(item) for item in self.allowed_assets),
        )
        if self.max_notional_usd <= 0:
            raise ValueError("max_notional_usd must be positive")
        if self.max_transactions_per_intent <= 0:
            raise ValueError("max_transactions_per_intent must be positive")

    def require_intent_allowed(self, intent: TradeIntent) -> None:
        if intent.notional_usd <= 0:
            raise LiveSafetyViolation("live notional must be positive")
        if intent.notional_usd > self.max_notional_usd:
            raise LiveSafetyViolation(
                f"live notional {intent.notional_usd} exceeds hard cap "
                f"{self.max_notional_usd}"
            )
        if self.allowed_assets and intent.asset not in self.allowed_assets:
            raise LiveSafetyViolation(
                f"asset {intent.asset} is not on the live allowlist"
            )

    def require_signing_authorized(
        self,
        controls: RuntimeSafetySnapshot,
    ) -> None:
        if controls.kill_switch_engaged:
            raise LiveSafetyViolation(
                f"kill switch is engaged: {controls.reason}"
            )
        if not controls.live_submission_enabled:
            raise LiveSafetyViolation(
                f"live submission is disabled: {controls.reason}"
            )
