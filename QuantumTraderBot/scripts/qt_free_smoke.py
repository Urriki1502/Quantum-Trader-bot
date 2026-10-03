#!/usr/bin/env python3
"""Read-only free-infrastructure smoke test.

No wallet, private key, signing or transaction submission is used.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.raydium_api import RaydiumTradeApiClient
from engine.solana_rpc import SolanaRpcClient

SOL = "So11111111111111111111111111111111111111112"
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


async def main() -> None:
    rpc = SolanaRpcClient()
    raydium = RaydiumTradeApiClient()

    health = await rpc.get_health()
    blockhash = await rpc.get_latest_blockhash()
    quote = await raydium.quote_base_in(
        input_mint=SOL,
        output_mint=USDC,
        amount_atomic=1_000_000,
        slippage_bps=50,
    )

    print(f"Solana health: {health}")
    print(f"Context slot: {blockhash.context_slot}")
    print(f"Last valid block height: {blockhash.last_valid_block_height}")
    print(f"Raydium output atomic: {quote.output_amount_atomic}")
    print(f"Raydium price impact: {quote.price_impact_pct}")
    print(f"Raydium route hops: {len(quote.route_plan)}")
    print("READ-ONLY: no transaction was built, signed or submitted.")


if __name__ == "__main__":
    asyncio.run(main())
