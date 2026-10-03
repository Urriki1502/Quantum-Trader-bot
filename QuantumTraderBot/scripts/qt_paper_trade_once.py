#!/usr/bin/env python3
"""Execute one paper trade using live read-only Raydium quotes.

No private key is loaded. No transaction is signed or submitted.
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
from engine.ledger import SQLiteTradeLedger
from engine.models import TradeIntent, TradeSide
from engine.raydium_api import RaydiumPoolApiClient, RaydiumTradeApiClient
from engine.raydium_quote_provider import RaydiumUsdcQuoteProvider
from engine.risk import RiskEngine, RiskPolicy
from engine.service import TradingEngine


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="One free/read-only Raydium-backed QT paper trade"
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--mint", required=True)
    parser.add_argument("--decimals", type=int, required=True)
    parser.add_argument("--side", choices=("buy", "sell"), required=True)
    parser.add_argument("--usd", type=Decimal, required=True)
    parser.add_argument("--max-slippage-bps", type=int, default=100)
    parser.add_argument("--db", default="qt-paper.db")
    parser.add_argument("--initial-cash", type=Decimal, default=Decimal("1000"))
    parser.add_argument("--min-liquidity-usd", type=Decimal, default=Decimal("10000"))
    args = parser.parse_args()

    registry = AssetRegistry(
        [AssetSpec(symbol=args.symbol, mint=args.mint, decimals=args.decimals)]
    )
    ledger = SQLiteTradeLedger(args.db)
    ledger.initialize_paper_account(args.initial_cash)
    quote_provider = RaydiumUsdcQuoteProvider(
        registry=registry,
        trade_api=RaydiumTradeApiClient(),
        pool_api=RaydiumPoolApiClient(),
    )
    engine = TradingEngine(
        quote_provider=quote_provider,
        execution_adapter=PaperExecutionAdapter(
            slippage_bps=min(10, args.max_slippage_bps),
            fee_bps=25,
        ),
        risk_engine=RiskEngine(
            RiskPolicy(
                max_notional_usd=max(args.usd, Decimal("100")),
                max_open_exposure_usd=Decimal("10000"),
                min_liquidity_usd=args.min_liquidity_usd,
                max_price_impact_bps=args.max_slippage_bps,
                max_slippage_bps=args.max_slippage_bps,
                max_daily_loss_usd=Decimal("500"),
                max_quote_age_seconds=20,
            )
        ),
        ledger=ledger,
    )

    intent = TradeIntent.create(
        asset=args.symbol,
        side=TradeSide(args.side),
        notional_usd=args.usd,
        max_slippage_bps=args.max_slippage_bps,
        metadata={"source": "raydium_public_paper"},
    )
    result = await engine.execute_intent(intent, engine.risk_snapshot())
    position = ledger.get_position(args.symbol)
    metrics = ledger.portfolio_metrics()
    account = ledger.get_paper_account()

    print(f"state={result.state.value}")
    print(f"intent_id={intent.intent_id}")
    print(f"provider={result.record.get('quote_provider')}")
    print(f"quote_price_usd={result.record.get('quote_price_usd')}")
    print(f"execution_mode={result.record.get('execution_mode')}")
    print(f"position_quantity={position.quantity}")
    print(f"average_cost_usd={position.average_cost_usd}")
    print(f"realized_pnl_usd={position.realized_pnl_usd}")
    print(f"paper_fills={metrics.fills}")
    print(f"paper_cash_usd={account.cash_usd if account else 'uninitialized'}")
    print("PAPER ONLY: no wallet was loaded and no transaction was submitted.")


if __name__ == "__main__":
    asyncio.run(main())
