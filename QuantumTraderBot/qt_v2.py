#!/usr/bin/env python3
"""QT Engine v2 free-first command line interface.

Available modes are deliberately non-custodial/non-live:
  * replay: deterministic historical JSONL replay
  * paper: continuous paper trading on public Raydium quotes
  * report: inspect a persisted paper ledger

There is intentionally no live-submit command in Phase 1.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.adapters import PaperExecutionAdapter
from engine.assets import AssetRegistry, AssetSpec
from engine.journal import SQLiteMarketJournal
from engine.ledger import SQLiteTradeLedger
from engine.market import RaydiumPollingMarketSource
from engine.raydium_api import RaydiumPoolApiClient, RaydiumTradeApiClient
from engine.raydium_quote_provider import RaydiumUsdcQuoteProvider
from engine.replay import JsonlMarketReplay, ReplayQuoteProvider, run_replay
from engine.report import build_performance_report
from engine.risk import RiskEngine, RiskPolicy
from engine.service import TradingEngine
from engine.session import PaperTradingSession
from engine.signals import MovingAverageCrossStrategy


def risk_policy(
    *,
    starting_cash: Decimal,
    notional: Decimal,
    min_liquidity: Decimal,
    max_slippage_bps: int,
) -> RiskPolicy:
    return RiskPolicy(
        max_notional_usd=max(notional, Decimal("100")),
        max_open_exposure_usd=starting_cash,
        min_liquidity_usd=min_liquidity,
        max_price_impact_bps=max_slippage_bps,
        max_slippage_bps=max_slippage_bps,
        max_daily_loss_usd=starting_cash * Decimal("0.10"),
        max_quote_age_seconds=20,
        buy_fee_reserve_bps=100,
    )


def build_session(
    *,
    db_path: str,
    quote_provider,
    starting_cash: Decimal,
    notional: Decimal,
    min_liquidity: Decimal,
    max_slippage_bps: int,
    short_window: int,
    long_window: int,
):
    ledger = SQLiteTradeLedger(db_path)
    ledger.initialize_paper_account(starting_cash)
    engine = TradingEngine(
        quote_provider=quote_provider,
        execution_adapter=PaperExecutionAdapter(
            slippage_bps=min(10, max_slippage_bps),
            fee_bps=25,
        ),
        risk_engine=RiskEngine(
            risk_policy(
                starting_cash=starting_cash,
                notional=notional,
                min_liquidity=min_liquidity,
                max_slippage_bps=max_slippage_bps,
            )
        ),
        ledger=ledger,
    )
    journal = SQLiteMarketJournal(db_path)
    strategy = MovingAverageCrossStrategy(
        short_window=short_window,
        long_window=long_window,
    )
    session = PaperTradingSession(
        engine=engine,
        journal=journal,
        strategy=strategy,
        buy_notional_usd=notional,
        sell_notional_usd=notional,
        max_slippage_bps=max_slippage_bps,
    )
    session.prime_from_journal("__unused__", limit=0)
    return session, ledger, journal


async def command_replay(args) -> int:
    quote_provider = ReplayQuoteProvider()
    session, ledger, journal = build_session(
        db_path=args.db,
        quote_provider=quote_provider,
        starting_cash=args.initial_cash,
        notional=args.notional,
        min_liquidity=args.min_liquidity_usd,
        max_slippage_bps=args.max_slippage_bps,
        short_window=args.short_window,
        long_window=args.long_window,
    )
    source = JsonlMarketReplay(args.input)
    summary = await run_replay(
        source=source,
        session=session,
        quote_provider=quote_provider,
    )
    print(summary.as_json())
    journal.close()
    ledger.close()
    return 0


async def command_paper(args) -> int:
    registry = AssetRegistry(
        [AssetSpec(symbol=args.symbol, mint=args.mint, decimals=args.decimals)]
    )
    quote_provider = RaydiumUsdcQuoteProvider(
        registry=registry,
        trade_api=RaydiumTradeApiClient(),
        pool_api=RaydiumPoolApiClient(),
    )
    session, ledger, journal = build_session(
        db_path=args.db,
        quote_provider=quote_provider,
        starting_cash=args.initial_cash,
        notional=args.notional,
        min_liquidity=args.min_liquidity_usd,
        max_slippage_bps=args.max_slippage_bps,
        short_window=args.short_window,
        long_window=args.long_window,
    )
    session.prime_from_journal(args.symbol, limit=args.long_window)

    source = RaydiumPollingMarketSource(
        quote_provider=quote_provider,
        asset=args.symbol,
        probe_notional_usd=args.probe_usd,
        max_slippage_bps=args.max_slippage_bps,
        interval_seconds=args.poll_seconds,
    )
    limit = args.max_ticks if args.max_ticks > 0 else None
    latest_price = None

    async for tick in source.stream(limit=limit):
        latest_price = tick.price_usd
        decision = await session.process_tick(tick)
        ledger.mark_to_market({args.symbol: tick.price_usd})
        account = ledger.get_paper_account()
        position = ledger.get_position(args.symbol)
        state = (
            decision.engine_result.state.value
            if decision.engine_result is not None
            else "-"
        )
        print(
            f"{tick.observed_at.isoformat()} "
            f"price={tick.price_usd} liquidity={tick.liquidity_usd} "
            f"signal={decision.signal.action.value} state={state} "
            f"position={position.quantity} cash={account.cash_usd if account else '-'} "
            f"reason={decision.signal.reason}",
            flush=True,
        )

    if latest_price is not None:
        print(
            build_performance_report(
                ledger,
                prices_usd={args.symbol: latest_price},
            ).as_json()
        )
    journal.close()
    ledger.close()
    return 0


def parse_mark(value: str) -> tuple[str, Decimal]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("mark must be SYMBOL=PRICE")
    symbol, raw = value.split("=", 1)
    try:
        price = Decimal(raw)
    except Exception as exc:
        raise argparse.ArgumentTypeError("invalid mark price") from exc
    if not symbol or price <= 0:
        raise argparse.ArgumentTypeError("mark must contain a symbol and positive price")
    return symbol, price


def command_report(args) -> int:
    ledger = SQLiteTradeLedger(args.db)
    report = build_performance_report(
        ledger,
        prices_usd=dict(args.mark),
    )
    print(report.as_json())
    ledger.close()
    return 0


def add_common(parser):
    parser.add_argument("--db", default="qt-paper.db")
    parser.add_argument("--initial-cash", type=Decimal, default=Decimal("1000"))
    parser.add_argument("--notional", type=Decimal, default=Decimal("25"))
    parser.add_argument("--max-slippage-bps", type=int, default=100)
    parser.add_argument("--min-liquidity-usd", type=Decimal, default=Decimal("10000"))
    parser.add_argument("--short-window", type=int, default=3)
    parser.add_argument("--long-window", type=int, default=8)


def build_parser():
    parser = argparse.ArgumentParser(description="QT Engine v2 free-first CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    replay = sub.add_parser("replay", help="Replay JSONL market events")
    add_common(replay)
    replay.add_argument("--input", required=True)

    paper = sub.add_parser("paper", help="Run free/read-only Raydium-backed paper mode")
    add_common(paper)
    paper.add_argument("--symbol", required=True)
    paper.add_argument("--mint", required=True)
    paper.add_argument("--decimals", required=True, type=int)
    paper.add_argument("--poll-seconds", type=float, default=30.0)
    paper.add_argument("--probe-usd", type=Decimal, default=Decimal("10"))
    paper.add_argument("--max-ticks", type=int, default=0)

    report = sub.add_parser("report", help="Report a persisted paper ledger")
    report.add_argument("--db", default="qt-paper.db")
    report.add_argument("--mark", action="append", default=[], type=parse_mark)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "replay":
        return asyncio.run(command_replay(args))
    if args.command == "paper":
        return asyncio.run(command_paper(args))
    return command_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
