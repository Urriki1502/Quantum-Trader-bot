from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .models import (
    ExecutionReceipt,
    Quote,
    TradeIntent,
    TradeState,
    assert_transition,
    utc_now,
)


class DuplicateIntentError(RuntimeError):
    pass


class TradeNotFoundError(KeyError):
    pass


class SQLiteTradeLedger:
    """Small durable ledger for paper/replay and the future live engine.

    SQLite is deliberate for the free-first phase. The schema keeps intent_id
    unique so a process restart cannot accidentally execute the same intent
    twice through this engine.
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
                """
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
                    filled_quote_usd = ?, fee_usd = ?, updated_at = ?
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
