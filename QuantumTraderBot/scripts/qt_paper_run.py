#!/usr/bin/env python3
"""Continuous free-first QT paper session using public Raydium data.

No private key is loaded, no transaction is signed, and no transaction is sent.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.adapters import PaperExecutionAdapter
from engine.assets import AssetRegistry, AssetSpec
from engine.journal import SQLiteMarketJournal
from engine.ledger import SQLiteTradeLedger
from engine.market import RaydiumPollingMarketSource
from engine.raydium_api import RaydiumPoolApiClient, RaydiumTradeApiClient
from engine.raydium_quote_provider import RaydiumUsdcQuoteProvider
from engine.risk import RiskEngine, RiskPolicy
from engine.service import TradingEngine
from engine.session import PaperTradingSession
from engine.signals import MovingAverageCrossStrategy


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run QT v2 continuous paper trading")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--mint", required=True)
    parser.add_argument("--decimals", type=int, required=True)
    parser.add_argument("--db", default="qt-paper.db")
    parser.add_argument("--initial-cash", type=Decimal, default=Decimal("1000"))
    parser.add_argument("--notional", type=Decimal, default=Decimal("25"))
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--probe-usd", type=Decimal, default=Decimal("10"))
    parser.add_argument("--max-slippage-bps", type=int, default=100)
    parser.add_argument("--min-liquidity-usd", type=Decimal, default=Decimal("10000"))
    parser.add_argument("--short-window", type=int, default=3)
    parser.add_argument("--long-window", type=int, default=8)
    parser.add_argument("--max-ticks", type=int, default=0)
    args = parser.parse_args()

    registry = AssetRegistry(
        [AssetSpec(symbol=args.symbol, mint=args.mint, decimals=args.decimals)]
    )
    trade_api = RaydiumTradeApiClient()
    pool_api = RaydiumPoolApiClient()
    quote_provider = RaydiumUsdcQuoteProvider(
        registry=registry,
        trade_api=trade_api,
        pool_api=pool_api,
    )

    ledger = SQLiteTradeLedger(args.db)
    ledger.initialize_paper_account(args.initial_cash)
    engine = TradingEngine(
        quote_provider=quote_provider,
        execution_adapter=PaperExecutionAdapter(slippage_bps=10, fee_bps=25),
        risk_engine=RiskEngine(
            RiskPolicy(
                max_notional_usd=max(args.notional, Decimal("100")),
                max_open_exposure_usd=args.initial_cash,
                min_liquidity_usd=args.min_liquidity_usd,
                max_price_impact_bps=args.max_slippage_bps,
                max_slippage_bps=args.max_slippage_bps,
                max_daily_loss_usd=args.initial_cash * Decimal("0.10"),
                max_quote_age_seconds=20,
                buy_fee_reserve_bps=100,
            )
        ),
        ledger=ledger,
    )
    strategy = MovingAverageCrossStrategy(
        short_window=args.short_window,
        long_window=args.long_window,
    )
    journal = SQLiteMarketJournal(args.db)
    session = PaperTradingSession(
        engine=engine,
        journal=journal,
        strategy=strategy,
        buy_notional_usd=args.notional,
        sell_notional_usd=args.notional,
        max_slippage_bps=args.max_slippage_bps,
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

    async for tick in source.stream(limit=limit):
        decision = await session.process_tick(tick)
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
            f"reason={decision.signal.reason}"
        )


if __name__ == "__main__":
    asyncio.run(main())
