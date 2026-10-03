import asyncio
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from engine.adapters import PaperExecutionAdapter
from engine.journal import SQLiteMarketJournal
from engine.ledger import SQLiteTradeLedger
from engine.replay import (
    JsonlMarketReplay,
    ReplayFormatError,
    ReplayQuoteProvider,
    run_replay,
)
from engine.risk import RiskEngine, RiskPolicy
from engine.service import TradingEngine
from engine.session import PaperTradingSession
from engine.signals import MovingAverageCrossStrategy


def write_events(path, prices):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with path.open("w", encoding="utf-8") as handle:
        for index, price in enumerate(prices):
            payload = {
                "event_id": f"e{index}",
                "asset": "TOKEN",
                "price_usd": str(price),
                "liquidity_usd": "1000000",
                "price_impact_bps": 0,
                "observed_at": (start + timedelta(minutes=index)).isoformat(),
                "source": "fixture",
            }
            handle.write(json.dumps(payload) + "\n")


def build(tmp_path):
    db = tmp_path / "replay.db"
    ledger = SQLiteTradeLedger(db)
    ledger.initialize_paper_account("1000")
    quotes = ReplayQuoteProvider()
    engine = TradingEngine(
        quote_provider=quotes,
        execution_adapter=PaperExecutionAdapter(slippage_bps=0, fee_bps=25),
        risk_engine=RiskEngine(
            RiskPolicy(
                max_notional_usd=Decimal("100"),
                max_open_exposure_usd=Decimal("1000"),
                min_liquidity_usd=Decimal("10000"),
                max_price_impact_bps=100,
                max_slippage_bps=100,
                max_daily_loss_usd=Decimal("100"),
            )
        ),
        ledger=ledger,
    )
    journal = SQLiteMarketJournal(db)
    session = PaperTradingSession(
        engine=engine,
        journal=journal,
        strategy=MovingAverageCrossStrategy(short_window=2, long_window=3),
        buy_notional_usd="100",
        sell_notional_usd="100",
        max_slippage_bps=100,
    )
    return quotes, session, ledger


def test_replay_produces_deterministic_report_and_equity_curve(tmp_path):
    source_path = tmp_path / "market.jsonl"
    write_events(source_path, [3, 2, 1, 4, 5, 2, 1])
    quotes, session, ledger = build(tmp_path)

    summary = asyncio.run(
        run_replay(
            source=JsonlMarketReplay(source_path),
            session=session,
            quote_provider=quotes,
        )
    )

    assert summary.ticks == 7
    assert summary.buy_signals == 1
    assert summary.sell_signals == 1
    assert summary.executed_trades == 2
    assert summary.report.fills == 2
    assert summary.report.buys == 1
    assert summary.report.sells == 1
    assert '"ticks": 7' in summary.as_json()
    assert ledger.get_position("TOKEN").quantity > 0


def test_replay_rejects_out_of_order_timestamps(tmp_path):
    path = tmp_path / "bad.jsonl"
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        {
            "asset": "TOKEN",
            "price_usd": "1",
            "liquidity_usd": "1000",
            "observed_at": start.isoformat(),
        },
        {
            "asset": "TOKEN",
            "price_usd": "1",
            "liquidity_usd": "1000",
            "observed_at": (start - timedelta(seconds=1)).isoformat(),
        },
    ]
    path.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")

    async def consume():
        result = []
        async for item in JsonlMarketReplay(path).stream():
            result.append(item)
        return result

    with pytest.raises(ReplayFormatError, match="monotonic"):
        asyncio.run(consume())


def test_replay_empty_input_fails_closed(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    quotes, session, _ = build(tmp_path)

    with pytest.raises(ReplayFormatError, match="no market events"):
        asyncio.run(
            run_replay(
                source=JsonlMarketReplay(path),
                session=session,
                quote_provider=quotes,
            )
        )
