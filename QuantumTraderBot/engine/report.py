from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Mapping

from .ledger import ReconciliationError, SQLiteTradeLedger


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    starting_cash_usd: Decimal
    cash_usd: Decimal
    positions_value_usd: Decimal
    total_equity_usd: Decimal
    net_pnl_usd: Decimal
    total_return_pct: Decimal
    realized_pnl_usd: Decimal
    unrealized_pnl_usd: Decimal
    max_drawdown_pct: Decimal
    fills: int
    buys: int
    sells: int
    winning_sells: int
    losing_sells: int
    win_rate: Decimal

    def as_json(self, *, indent: int = 2) -> str:
        payload = {
            key: str(value) if isinstance(value, Decimal) else value
            for key, value in asdict(self).items()
        }
        return json.dumps(payload, indent=indent, sort_keys=True)


def build_performance_report(
    ledger: SQLiteTradeLedger,
    *,
    prices_usd: Mapping[str, Decimal | str | int | float],
) -> PerformanceReport:
    account = ledger.get_paper_account()
    if account is None:
        raise ReconciliationError("paper account is not initialized")

    prices = {key: Decimal(str(value)) for key, value in prices_usd.items()}
    if any(value <= 0 for value in prices.values()):
        raise ValueError("report mark prices must be positive")

    positions_value = Decimal("0")
    unrealized = Decimal("0")
    for position in ledger.list_positions():
        if position.quantity == 0:
            continue
        if position.asset not in prices:
            raise ReconciliationError(
                f"missing report mark price for {position.asset}"
            )
        mark = prices[position.asset]
        positions_value += position.quantity * mark
        unrealized += position.quantity * (mark - position.average_cost_usd)

    equity = account.cash_usd + positions_value
    net_pnl = equity - account.starting_cash_usd
    total_return = (
        Decimal("0")
        if account.starting_cash_usd == 0
        else (net_pnl / account.starting_cash_usd) * Decimal("100")
    )
    metrics = ledger.portfolio_metrics()

    return PerformanceReport(
        starting_cash_usd=account.starting_cash_usd,
        cash_usd=account.cash_usd,
        positions_value_usd=positions_value,
        total_equity_usd=equity,
        net_pnl_usd=net_pnl,
        total_return_pct=total_return,
        realized_pnl_usd=metrics.realized_pnl_usd,
        unrealized_pnl_usd=unrealized,
        max_drawdown_pct=account.max_drawdown_pct,
        fills=metrics.fills,
        buys=metrics.buys,
        sells=metrics.sells,
        winning_sells=metrics.winning_sells,
        losing_sells=metrics.losing_sells,
        win_rate=metrics.win_rate,
    )
