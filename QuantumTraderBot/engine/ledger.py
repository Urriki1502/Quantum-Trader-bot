from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from .models import (
    ExecutionReceipt,
    Quote,
    TradeIntent,
    TradeSide,
    TradeState,
    assert_transition,
    utc_now,
)


class DuplicateIntentError(RuntimeError):
    pass


class TradeNotFoundError(KeyError):
    pass


class ReconciliationError(RuntimeError):
    pass


class InsufficientPositionError(ReconciliationError):
    pass


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    asset: str
    quantity: Decimal
    average_cost_usd: Decimal
    realized_pnl_usd: Decimal

    @property
    def cost_basis_usd(self) -> Decimal:
        return self.quantity * self.average_cost_usd


@dataclass(frozen=True, slots=True)
class PortfolioFillResult:
    execution_id: str
    intent_id: str
    asset: str
    side: TradeSide
    quantity: Decimal
    gross_quote_usd: Decimal
    fee_usd: Decimal
    realized_pnl_usd: Decimal
    existing: bool = False


@dataclass(frozen=True, slots=True)
class PortfolioMetrics:
    fills: int
    buys: int
    sells: int
    winning_sells: int
    losing_sells: int
    realized_pnl_usd: Decimal

    @property
    def win_rate(self) -> Decimal:
        decisive = self.winning_sells + self.losing_sells
        if decisive == 0:
            return Decimal("0")
        return Decimal(self.winning_sells) / Decimal(decisive)


class SQLiteTradeLedger:
    """Durable trade + portfolio ledger for paper/replay and future live adapters.

    Core invariants:
      * intent_id is unique and durable.
      * execution_id is accounted at most once.
      * a confirmed fill and its RECONCILED state transition are committed in
        one SQLite transaction.
      * a restart can recover CONFIRMED trades without re-executing them.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_schema()
        self._migrate_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _create_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    intent_id TEXT PRIMARY KEY,
                    asset TEXT NOT NULL,
                    side TEXT NOT NULL,
                    notional_usd TEXT NOT NULL,
                    max_slippage_bps INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    quote_id TEXT,
                    quote_provider TEXT,
                    quote_price_usd TEXT,
                    execution_id TEXT UNIQUE,
                    execution_mode TEXT,
                    external_ref TEXT,
                    average_price_usd TEXT,
                    filled_base_amount TEXT,
                    filled_quote_usd TEXT,
                    fee_usd TEXT,
                    actual_slippage_bps INTEGER,
                    submitted_at TEXT,
                    confirmed_at TEXT,
                    last_error TEXT
                );

                CREATE TABLE IF NOT EXISTS trade_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    intent_id TEXT NOT NULL,
                    from_state TEXT,
                    to_state TEXT NOT NULL,
                    reason TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(intent_id) REFERENCES trades(intent_id)
                );

                CREATE INDEX IF NOT EXISTS idx_trade_events_intent
                    ON trade_events(intent_id, id);

                CREATE TABLE IF NOT EXISTS positions (
                    asset TEXT PRIMARY KEY,
                    quantity TEXT NOT NULL,
                    average_cost_usd TEXT NOT NULL,
                    realized_pnl_usd TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS portfolio_fills (
                    execution_id TEXT PRIMARY KEY,
                    intent_id TEXT NOT NULL UNIQUE,
                    asset TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity TEXT NOT NULL,
                    gross_quote_usd TEXT NOT NULL,
                    fee_usd TEXT NOT NULL,
                    realized_pnl_usd TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(intent_id) REFERENCES trades(intent_id)
                );

                CREATE INDEX IF NOT EXISTS idx_portfolio_fills_asset
                    ON portfolio_fills(asset, created_at);
                """
            )

    def _migrate_schema(self) -> None:
        required = {
            "actual_slippage_bps": "INTEGER",
            "submitted_at": "TEXT",
            "confirmed_at": "TEXT",
        }
        with self._lock, self._conn:
            columns = {
                row["name"]
                for row in self._conn.execute("PRAGMA table_info(trades)").fetchall()
            }
            for name, sql_type in required.items():
                if name not in columns:
                    self._conn.execute(
                        f"ALTER TABLE trades ADD COLUMN {name} {sql_type}"
                    )

    def create_intent(self, intent: TradeIntent) -> None:
        now = utc_now().isoformat()
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    """
                    INSERT INTO trades (
                        intent_id, asset, side, notional_usd, max_slippage_bps,
                        state, metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        intent.intent_id,
                        intent.asset,
                        intent.side.value,
                        str(intent.notional_usd),
                        intent.max_slippage_bps,
                        TradeState.CREATED.value,
                        json.dumps(dict(intent.metadata), sort_keys=True),
                        intent.created_at.isoformat(),
                        now,
                    ),
                )
                self._conn.execute(
                    """
                    INSERT INTO trade_events(intent_id, from_state, to_state, reason, created_at)
                    VALUES (?, NULL, ?, ?, ?)
                    """,
                    (intent.intent_id, TradeState.CREATED.value, "intent_created", now),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateIntentError(intent.intent_id) from exc

    def get_trade(self, intent_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM trades WHERE intent_id = ?", (intent_id,)
            ).fetchone()
            return dict(row) if row is not None else None

    def load_intent(self, intent_id: str) -> TradeIntent:
        row = self.get_trade(intent_id)
        if row is None:
            raise TradeNotFoundError(intent_id)
        return TradeIntent(
            intent_id=row["intent_id"],
            asset=row["asset"],
            side=TradeSide(row["side"]),
            notional_usd=Decimal(row["notional_usd"]),
            max_slippage_bps=int(row["max_slippage_bps"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            metadata=json.loads(row["metadata_json"]),
        )

    def load_execution_receipt(self, intent_id: str) -> ExecutionReceipt:
        row = self.get_trade(intent_id)
        if row is None:
            raise TradeNotFoundError(intent_id)
        required = (
            "execution_id",
            "quote_id",
            "execution_mode",
            "external_ref",
            "average_price_usd",
            "filled_base_amount",
            "filled_quote_usd",
            "fee_usd",
            "submitted_at",
        )
        if any(row.get(name) is None for name in required):
            raise ReconciliationError(f"trade {intent_id} is missing execution receipt fields")

        state = TradeState.CONFIRMED if row.get("confirmed_at") else TradeState.SUBMITTED
        return ExecutionReceipt(
            execution_id=row["execution_id"],
            quote_id=row["quote_id"],
            mode=row["execution_mode"],
            state=state,
            external_ref=row["external_ref"],
            average_price_usd=Decimal(row["average_price_usd"]),
            filled_base_amount=Decimal(row["filled_base_amount"]),
            filled_quote_usd=Decimal(row["filled_quote_usd"]),
            fee_usd=Decimal(row["fee_usd"]),
            actual_slippage_bps=int(row.get("actual_slippage_bps") or 0),
            submitted_at=datetime.fromisoformat(row["submitted_at"]),
            confirmed_at=(
                datetime.fromisoformat(row["confirmed_at"])
                if row.get("confirmed_at")
                else None
            ),
        )

    def list_trades_by_state(self, state: TradeState) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM trades WHERE state = ? ORDER BY created_at",
                (state.value,),
            ).fetchall()
            return [dict(row) for row in rows]

    def events(self, intent_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, intent_id, from_state, to_state, reason, created_at
                FROM trade_events
                WHERE intent_id = ?
                ORDER BY id
                """,
                (intent_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def transition(self, intent_id: str, target: TradeState, *, reason: str = "") -> None:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT state FROM trades WHERE intent_id = ?", (intent_id,)
            ).fetchone()
            if row is None:
                raise TradeNotFoundError(intent_id)
            current = TradeState(row["state"])
            assert_transition(current, target)
            now = utc_now().isoformat()
            self._conn.execute(
                "UPDATE trades SET state = ?, updated_at = ? WHERE intent_id = ?",
                (target.value, now, intent_id),
            )
            self._conn.execute(
                """
                INSERT INTO trade_events(intent_id, from_state, to_state, reason, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (intent_id, current.value, target.value, reason, now),
            )

    def attach_quote(self, intent_id: str, quote: Quote) -> None:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                UPDATE trades
                SET quote_id = ?, quote_provider = ?, quote_price_usd = ?, updated_at = ?
                WHERE intent_id = ?
                """,
                (
                    quote.quote_id,
                    quote.provider,
                    str(quote.price_usd),
                    utc_now().isoformat(),
                    intent_id,
                ),
            )
            if cur.rowcount != 1:
                raise TradeNotFoundError(intent_id)

    def attach_execution(self, intent_id: str, receipt: ExecutionReceipt) -> None:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                UPDATE trades
                SET execution_id = ?, execution_mode = ?, external_ref = ?,
                    average_price_usd = ?, filled_base_amount = ?,
                    filled_quote_usd = ?, fee_usd = ?,
                    actual_slippage_bps = ?, submitted_at = ?, confirmed_at = ?,
                    updated_at = ?
                WHERE intent_id = ?
                """,
                (
                    receipt.execution_id,
                    receipt.mode,
                    receipt.external_ref,
                    str(receipt.average_price_usd),
                    str(receipt.filled_base_amount),
                    str(receipt.filled_quote_usd),
                    str(receipt.fee_usd),
                    receipt.actual_slippage_bps,
                    receipt.submitted_at.isoformat(),
                    receipt.confirmed_at.isoformat() if receipt.confirmed_at else None,
                    utc_now().isoformat(),
                    intent_id,
                ),
            )
            if cur.rowcount != 1:
                raise TradeNotFoundError(intent_id)

    def set_error(self, intent_id: str, message: str) -> None:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE trades SET last_error = ?, updated_at = ? WHERE intent_id = ?",
                (message, utc_now().isoformat(), intent_id),
            )
            if cur.rowcount != 1:
                raise TradeNotFoundError(intent_id)

    def get_position(self, asset: str) -> PositionSnapshot:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM positions WHERE asset = ?", (asset,)
            ).fetchone()
        if row is None:
            return PositionSnapshot(
                asset=asset,
                quantity=Decimal("0"),
                average_cost_usd=Decimal("0"),
                realized_pnl_usd=Decimal("0"),
            )
        return PositionSnapshot(
            asset=row["asset"],
            quantity=Decimal(row["quantity"]),
            average_cost_usd=Decimal(row["average_cost_usd"]),
            realized_pnl_usd=Decimal(row["realized_pnl_usd"]),
        )

    def reconcile_confirmed_execution(
        self,
        intent: TradeIntent,
        receipt: ExecutionReceipt,
    ) -> PortfolioFillResult:
        if receipt.state is not TradeState.CONFIRMED or receipt.confirmed_at is None:
            raise ReconciliationError("only confirmed executions can be reconciled")

        with self._lock, self._conn:
            trade = self._conn.execute(
                "SELECT state, execution_id FROM trades WHERE intent_id = ?",
                (intent.intent_id,),
            ).fetchone()
            if trade is None:
                raise TradeNotFoundError(intent.intent_id)
            if trade["execution_id"] != receipt.execution_id:
                raise ReconciliationError("receipt execution_id does not match trade")

            existing = self._conn.execute(
                "SELECT * FROM portfolio_fills WHERE execution_id = ? OR intent_id = ?",
                (receipt.execution_id, intent.intent_id),
            ).fetchone()
            if existing is not None:
                if (
                    existing["execution_id"] != receipt.execution_id
                    or existing["intent_id"] != intent.intent_id
                ):
                    raise ReconciliationError("execution/intent accounting collision")
                if TradeState(trade["state"]) is TradeState.CONFIRMED:
                    self._mark_reconciled_locked(
                        intent.intent_id,
                        reason=f"recovered:{receipt.execution_id}",
                    )
                return self._fill_result(existing, existing=True)

            if TradeState(trade["state"]) is not TradeState.CONFIRMED:
                raise ReconciliationError(
                    f"trade must be confirmed before reconciliation, got {trade['state']}"
                )

            current = self._conn.execute(
                "SELECT * FROM positions WHERE asset = ?", (intent.asset,)
            ).fetchone()
            old_qty = Decimal(current["quantity"]) if current else Decimal("0")
            old_avg = Decimal(current["average_cost_usd"]) if current else Decimal("0")
            old_realized = (
                Decimal(current["realized_pnl_usd"]) if current else Decimal("0")
            )

            qty = receipt.filled_base_amount
            gross = receipt.filled_quote_usd
            fee = receipt.fee_usd
            realized = Decimal("0")

            if qty <= 0 or gross < 0 or fee < 0:
                raise ReconciliationError("execution receipt contains invalid fill amounts")

            if intent.side is TradeSide.BUY:
                new_qty = old_qty + qty
                total_cost = (old_qty * old_avg) + gross + fee
                new_avg = total_cost / new_qty
                new_realized = old_realized
            else:
                if qty > old_qty:
                    raise InsufficientPositionError(
                        f"sell quantity {qty} exceeds position {old_qty}"
                    )
                net_proceeds = gross - fee
                cost_basis = old_avg * qty
                realized = net_proceeds - cost_basis
                new_qty = old_qty - qty
                new_avg = old_avg if new_qty > 0 else Decimal("0")
                new_realized = old_realized + realized

            now = utc_now().isoformat()
            self._conn.execute(
                """
                INSERT INTO positions(asset, quantity, average_cost_usd, realized_pnl_usd, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(asset) DO UPDATE SET
                    quantity = excluded.quantity,
                    average_cost_usd = excluded.average_cost_usd,
                    realized_pnl_usd = excluded.realized_pnl_usd,
                    updated_at = excluded.updated_at
                """,
                (
                    intent.asset,
                    str(new_qty),
                    str(new_avg),
                    str(new_realized),
                    now,
                ),
            )
            self._conn.execute(
                """
                INSERT INTO portfolio_fills(
                    execution_id, intent_id, asset, side, quantity,
                    gross_quote_usd, fee_usd, realized_pnl_usd, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt.execution_id,
                    intent.intent_id,
                    intent.asset,
                    intent.side.value,
                    str(qty),
                    str(gross),
                    str(fee),
                    str(realized),
                    now,
                ),
            )
            self._mark_reconciled_locked(
                intent.intent_id,
                reason=f"portfolio_fill:{receipt.execution_id}",
            )

            row = self._conn.execute(
                "SELECT * FROM portfolio_fills WHERE execution_id = ?",
                (receipt.execution_id,),
            ).fetchone()
            return self._fill_result(row, existing=False)

    def _mark_reconciled_locked(self, intent_id: str, *, reason: str) -> None:
        row = self._conn.execute(
            "SELECT state FROM trades WHERE intent_id = ?", (intent_id,)
        ).fetchone()
        if row is None:
            raise TradeNotFoundError(intent_id)
        current = TradeState(row["state"])
        if current is TradeState.RECONCILED:
            return
        assert_transition(current, TradeState.RECONCILED)
        now = utc_now().isoformat()
        self._conn.execute(
            "UPDATE trades SET state = ?, updated_at = ? WHERE intent_id = ?",
            (TradeState.RECONCILED.value, now, intent_id),
        )
        self._conn.execute(
            """
            INSERT INTO trade_events(intent_id, from_state, to_state, reason, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                intent_id,
                current.value,
                TradeState.RECONCILED.value,
                reason,
                now,
            ),
        )

    @staticmethod
    def _fill_result(row: sqlite3.Row, *, existing: bool) -> PortfolioFillResult:
        return PortfolioFillResult(
            execution_id=row["execution_id"],
            intent_id=row["intent_id"],
            asset=row["asset"],
            side=TradeSide(row["side"]),
            quantity=Decimal(row["quantity"]),
            gross_quote_usd=Decimal(row["gross_quote_usd"]),
            fee_usd=Decimal(row["fee_usd"]),
            realized_pnl_usd=Decimal(row["realized_pnl_usd"]),
            existing=existing,
        )

    def portfolio_fill_count(self) -> int:
        with self._lock:
            return int(
                self._conn.execute("SELECT COUNT(*) FROM portfolio_fills").fetchone()[0]
            )

    def open_cost_basis_exposure(self) -> Decimal:
        with self._lock:
            rows = self._conn.execute(
                "SELECT quantity, average_cost_usd FROM positions"
            ).fetchall()
        return sum(
            (
                Decimal(row["quantity"]) * Decimal(row["average_cost_usd"])
                for row in rows
            ),
            Decimal("0"),
        )

    def portfolio_metrics(self) -> PortfolioMetrics:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                    COUNT(*) AS fills,
                    SUM(CASE WHEN side = 'buy' THEN 1 ELSE 0 END) AS buys,
                    SUM(CASE WHEN side = 'sell' THEN 1 ELSE 0 END) AS sells,
                    SUM(CASE WHEN side = 'sell' AND CAST(realized_pnl_usd AS REAL) > 0 THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN side = 'sell' AND CAST(realized_pnl_usd AS REAL) < 0 THEN 1 ELSE 0 END) AS losses
                FROM portfolio_fills
                """
            ).fetchone()
            pnl_rows = self._conn.execute(
                "SELECT realized_pnl_usd FROM portfolio_fills WHERE side = 'sell'"
            ).fetchall()

        realized = sum(
            (Decimal(item["realized_pnl_usd"]) for item in pnl_rows),
            Decimal("0"),
        )
        return PortfolioMetrics(
            fills=int(row["fills"] or 0),
            buys=int(row["buys"] or 0),
            sells=int(row["sells"] or 0),
            winning_sells=int(row["wins"] or 0),
            losing_sells=int(row["losses"] or 0),
            realized_pnl_usd=realized,
        )
