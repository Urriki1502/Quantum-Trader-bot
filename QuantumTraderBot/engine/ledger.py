from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
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


class InsufficientCashError(ReconciliationError):
    pass


@dataclass(frozen=True, slots=True)
class ExecutionAttemptSnapshot:
    attempt_id: str
    intent_id: str
    tx_identity: str
    recent_blockhash: str
    last_valid_block_height: int
    external_ref: str | None
    created_at: datetime
    updated_at: datetime
    existing: bool = False


@dataclass(frozen=True, slots=True)
class PaperAccountSnapshot:
    starting_cash_usd: Decimal
    cash_usd: Decimal
    high_water_equity_usd: Decimal
    max_drawdown_pct: Decimal


@dataclass(frozen=True, slots=True)
class EquitySnapshot:
    cash_usd: Decimal
    positions_value_usd: Decimal
    total_equity_usd: Decimal
    drawdown_pct: Decimal
    max_drawdown_pct: Decimal


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

                CREATE TABLE IF NOT EXISTS paper_account (
                    id INTEGER PRIMARY KEY CHECK(id = 1),
                    starting_cash_usd TEXT NOT NULL,
                    cash_usd TEXT NOT NULL,
                    high_water_equity_usd TEXT NOT NULL,
                    max_drawdown_pct TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS equity_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cash_usd TEXT NOT NULL,
                    positions_value_usd TEXT NOT NULL,
                    total_equity_usd TEXT NOT NULL,
                    drawdown_pct TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS execution_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    intent_id TEXT NOT NULL UNIQUE,
                    tx_identity TEXT NOT NULL UNIQUE,
                    recent_blockhash TEXT NOT NULL,
                    last_valid_block_height INTEGER NOT NULL,
                    external_ref TEXT UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(intent_id) REFERENCES trades(intent_id)
                );
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

    def reserve_execution_attempt(
        self,
        intent_id: str,
        *,
        attempt_id: str,
        tx_identity: str,
        recent_blockhash: str,
        last_valid_block_height: int,
    ) -> ExecutionAttemptSnapshot:
        if not attempt_id or not tx_identity or not recent_blockhash:
            raise ValueError("attempt_id, tx_identity and recent_blockhash are required")
        if last_valid_block_height < 0:
            raise ValueError("last_valid_block_height must be >= 0")

        with self._lock, self._conn:
            trade = self._conn.execute(
                "SELECT state FROM trades WHERE intent_id = ?",
                (intent_id,),
            ).fetchone()
            if trade is None:
                raise TradeNotFoundError(intent_id)

            existing = self._conn.execute(
                """
                SELECT * FROM execution_attempts
                WHERE intent_id = ? OR attempt_id = ? OR tx_identity = ?
                """,
                (intent_id, attempt_id, tx_identity),
            ).fetchone()
            if existing is not None:
                exact = (
                    existing["intent_id"] == intent_id
                    and existing["attempt_id"] == attempt_id
                    and existing["tx_identity"] == tx_identity
                    and existing["recent_blockhash"] == recent_blockhash
                    and int(existing["last_valid_block_height"]) == last_valid_block_height
                )
                if not exact:
                    raise ReconciliationError("execution attempt identity collision")
                return self._attempt_snapshot(existing, existing=True)

            if TradeState(trade["state"]) is not TradeState.EXECUTION_PENDING:
                raise ReconciliationError(
                    "execution attempt may only be reserved from execution_pending"
                )

            now = utc_now().isoformat()
            self._conn.execute(
                """
                INSERT INTO execution_attempts(
                    attempt_id, intent_id, tx_identity, recent_blockhash,
                    last_valid_block_height, external_ref, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    attempt_id,
                    intent_id,
                    tx_identity,
                    recent_blockhash,
                    last_valid_block_height,
                    now,
                    now,
                ),
            )
            row = self._conn.execute(
                "SELECT * FROM execution_attempts WHERE intent_id = ?",
                (intent_id,),
            ).fetchone()
            return self._attempt_snapshot(row, existing=False)

    def attach_attempt_external_ref(
        self,
        intent_id: str,
        external_ref: str,
    ) -> ExecutionAttemptSnapshot:
        if not external_ref:
            raise ValueError("external_ref must be non-empty")
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT * FROM execution_attempts WHERE intent_id = ?",
                (intent_id,),
            ).fetchone()
            if row is None:
                raise ReconciliationError("execution attempt has not been reserved")
            if row["external_ref"] is not None:
                if row["external_ref"] != external_ref:
                    raise ReconciliationError(
                        "execution attempt already has a different external reference"
                    )
                return self._attempt_snapshot(row, existing=True)

            collision = self._conn.execute(
                "SELECT intent_id FROM execution_attempts WHERE external_ref = ?",
                (external_ref,),
            ).fetchone()
            if collision is not None and collision["intent_id"] != intent_id:
                raise ReconciliationError("external execution reference collision")

            now = utc_now().isoformat()
            self._conn.execute(
                """
                UPDATE execution_attempts
                SET external_ref = ?, updated_at = ?
                WHERE intent_id = ?
                """,
                (external_ref, now, intent_id),
            )
            row = self._conn.execute(
                "SELECT * FROM execution_attempts WHERE intent_id = ?",
                (intent_id,),
            ).fetchone()
            return self._attempt_snapshot(row, existing=False)

    def get_execution_attempt(self, intent_id: str) -> ExecutionAttemptSnapshot | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM execution_attempts WHERE intent_id = ?",
                (intent_id,),
            ).fetchone()
        return None if row is None else self._attempt_snapshot(row, existing=True)

    def execution_attempt_count(self) -> int:
        with self._lock:
            return int(
                self._conn.execute("SELECT COUNT(*) FROM execution_attempts").fetchone()[0]
            )

    @staticmethod
    def _attempt_snapshot(
        row: sqlite3.Row,
        *,
        existing: bool,
    ) -> ExecutionAttemptSnapshot:
        return ExecutionAttemptSnapshot(
            attempt_id=row["attempt_id"],
            intent_id=row["intent_id"],
            tx_identity=row["tx_identity"],
            recent_blockhash=row["recent_blockhash"],
            last_valid_block_height=int(row["last_valid_block_height"]),
            external_ref=row["external_ref"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            existing=existing,
        )

    def set_error(self, intent_id: str, message: str) -> None:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE trades SET last_error = ?, updated_at = ? WHERE intent_id = ?",
                (message, utc_now().isoformat(), intent_id),
            )
            if cur.rowcount != 1:
                raise TradeNotFoundError(intent_id)

    def list_positions(self) -> list[PositionSnapshot]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM positions ORDER BY asset"
            ).fetchall()
        return [
            PositionSnapshot(
                asset=row["asset"],
                quantity=Decimal(row["quantity"]),
                average_cost_usd=Decimal(row["average_cost_usd"]),
                realized_pnl_usd=Decimal(row["realized_pnl_usd"]),
            )
            for row in rows
        ]

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

            paper_account = self._conn.execute(
                "SELECT * FROM paper_account WHERE id = 1"
            ).fetchone()
            new_cash: Decimal | None = None
            if paper_account is not None:
                old_cash = Decimal(paper_account["cash_usd"])
                if intent.side is TradeSide.BUY:
                    new_cash = old_cash - gross - fee
                    if new_cash < 0:
                        raise InsufficientCashError(
                            f"paper cash {old_cash} cannot cover confirmed cost {gross + fee}"
                        )
                else:
                    new_cash = old_cash + gross - fee

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
            if paper_account is not None and new_cash is not None:
                self._conn.execute(
                    """
                    UPDATE paper_account
                    SET cash_usd = ?, updated_at = ?
                    WHERE id = 1
                    """,
                    (str(new_cash), now),
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

    def initialize_paper_account(self, starting_cash_usd: Decimal | str | int | float) -> PaperAccountSnapshot:
        starting = Decimal(str(starting_cash_usd))
        if starting <= 0:
            raise ValueError("starting cash must be positive")
        now = utc_now().isoformat()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT * FROM paper_account WHERE id = 1"
            ).fetchone()
            if row is None:
                self._conn.execute(
                    """
                    INSERT INTO paper_account(
                        id, starting_cash_usd, cash_usd,
                        high_water_equity_usd, max_drawdown_pct, updated_at
                    ) VALUES (1, ?, ?, ?, ?, ?)
                    """,
                    (str(starting), str(starting), str(starting), "0", now),
                )
            elif Decimal(row["starting_cash_usd"]) != starting:
                raise ValueError(
                    "paper account already exists with a different starting balance"
                )
        account = self.get_paper_account()
        if account is None:
            raise RuntimeError("failed to initialize paper account")
        return account

    def get_paper_account(self) -> PaperAccountSnapshot | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM paper_account WHERE id = 1"
            ).fetchone()
        if row is None:
            return None
        return PaperAccountSnapshot(
            starting_cash_usd=Decimal(row["starting_cash_usd"]),
            cash_usd=Decimal(row["cash_usd"]),
            high_water_equity_usd=Decimal(row["high_water_equity_usd"]),
            max_drawdown_pct=Decimal(row["max_drawdown_pct"]),
        )

    def mark_to_market(self, prices_usd: dict[str, Decimal | str | int | float]) -> EquitySnapshot:
        prices = {key: Decimal(str(value)) for key, value in prices_usd.items()}
        if any(value <= 0 for value in prices.values()):
            raise ValueError("mark prices must be positive")

        with self._lock, self._conn:
            account = self._conn.execute(
                "SELECT * FROM paper_account WHERE id = 1"
            ).fetchone()
            if account is None:
                raise ReconciliationError("paper account is not initialized")

            rows = self._conn.execute(
                "SELECT asset, quantity FROM positions WHERE CAST(quantity AS REAL) != 0"
            ).fetchall()
            positions_value = Decimal("0")
            for row in rows:
                asset = row["asset"]
                if asset not in prices:
                    raise ReconciliationError(f"missing mark price for {asset}")
                positions_value += Decimal(row["quantity"]) * prices[asset]

            cash = Decimal(account["cash_usd"])
            equity = cash + positions_value
            previous_high = Decimal(account["high_water_equity_usd"])
            high_water = max(previous_high, equity)
            drawdown = (
                Decimal("0")
                if high_water <= 0
                else ((high_water - equity) / high_water) * Decimal("100")
            )
            max_drawdown = max(Decimal(account["max_drawdown_pct"]), drawdown)
            now = utc_now().isoformat()

            self._conn.execute(
                """
                UPDATE paper_account
                SET high_water_equity_usd = ?, max_drawdown_pct = ?, updated_at = ?
                WHERE id = 1
                """,
                (str(high_water), str(max_drawdown), now),
            )
            self._conn.execute(
                """
                INSERT INTO equity_snapshots(
                    cash_usd, positions_value_usd, total_equity_usd,
                    drawdown_pct, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (str(cash), str(positions_value), str(equity), str(drawdown), now),
            )

        return EquitySnapshot(
            cash_usd=cash,
            positions_value_usd=positions_value,
            total_equity_usd=equity,
            drawdown_pct=drawdown,
            max_drawdown_pct=max_drawdown,
        )

    def realized_pnl_today_utc(self) -> Decimal:
        now = datetime.now(timezone.utc)
        start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc).isoformat()
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT realized_pnl_usd
                FROM portfolio_fills
                WHERE side = 'sell' AND created_at >= ?
                """,
                (start,),
            ).fetchall()
        return sum(
            (Decimal(row["realized_pnl_usd"]) for row in rows),
            Decimal("0"),
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
