from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from .market import MarketTick


class SQLiteMarketJournal:
    """Durable de-duplication journal for market events."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS market_events (
                    event_id TEXT PRIMARY KEY,
                    asset TEXT NOT NULL,
                    price_usd TEXT NOT NULL,
                    liquidity_usd TEXT NOT NULL,
                    price_impact_bps INTEGER NOT NULL,
                    observed_at TEXT NOT NULL,
                    source TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_market_events_asset_time
                ON market_events(asset, observed_at)
                """
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def record(self, tick: MarketTick) -> bool:
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    """
                    INSERT INTO market_events(
                        event_id, asset, price_usd, liquidity_usd,
                        price_impact_bps, observed_at, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tick.event_id,
                        tick.asset,
                        str(tick.price_usd),
                        str(tick.liquidity_usd),
                        tick.price_impact_bps,
                        tick.observed_at.isoformat(),
                        tick.source,
                    ),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def recent(self, asset: str, *, limit: int) -> list[MarketTick]:
        if limit <= 0:
            return []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT *
                FROM market_events
                WHERE asset = ?
                ORDER BY observed_at DESC
                LIMIT ?
                """,
                (asset, limit),
            ).fetchall()
        ticks = [
            MarketTick(
                event_id=row["event_id"],
                asset=row["asset"],
                price_usd=Decimal(row["price_usd"]),
                liquidity_usd=Decimal(row["liquidity_usd"]),
                price_impact_bps=int(row["price_impact_bps"]),
                observed_at=datetime.fromisoformat(row["observed_at"]),
                source=row["source"],
            )
            for row in rows
        ]
        ticks.reverse()
        return ticks

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM market_events").fetchone()[0])
