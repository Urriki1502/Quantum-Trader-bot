import asyncio
from datetime import datetime, timezone
from decimal import Decimal

from engine.assets import AssetRegistry, AssetSpec, USDC_MINT
from engine.models import TradeIntent, TradeSide
from engine.raydium_api import RaydiumSwapQuote
from engine.raydium_quote_provider import RaydiumUsdcQuoteProvider


def run(coro):
    return asyncio.run(coro)


class FakeTradeApi:
    def __init__(self, quotes):
        self.quotes = list(quotes)
        self.calls = []

    async def quote_base_in(self, **kwargs):
        self.calls.append(kwargs)
        return self.quotes.pop(0)


class FakePoolApi:
    def __init__(self, liquidity):
        self.liquidity = Decimal(str(liquidity))
        self.calls = []

    async def pair_liquidity_usd(self, mint1, mint2):
        self.calls.append((mint1, mint2))
        return self.liquidity


def raw_quote(
    *,
    input_mint,
    output_mint,
    input_atomic,
    output_atomic,
    slippage_bps=100,
    impact="0.05",
):
    return RaydiumSwapQuote(
        request_id="r",
        input_mint=input_mint,
        output_mint=output_mint,
        input_amount_atomic=input_atomic,
        output_amount_atomic=output_atomic,
        other_amount_threshold_atomic=0,
        slippage_bps=slippage_bps,
        price_impact_pct=Decimal(impact),
        route_plan=(),
        raw_data={},
        created_at=datetime.now(timezone.utc),
    )


def registry():
    return AssetRegistry(
        [
            AssetSpec(
                symbol="MEME",
                mint="Meme111111111111111111111111111111111111111",
                decimals=6,
            )
        ]
    )


def test_buy_normalizes_usdc_route_into_engine_quote():
    asset_mint = registry().resolve("MEME").mint
    trade = FakeTradeApi(
        [
            raw_quote(
                input_mint=USDC_MINT,
                output_mint=asset_mint,
                input_atomic=10_000_000,
                output_atomic=5_000_000,
                impact="0.05",
            )
        ]
    )
    pools = FakePoolApi("250000")
    provider = RaydiumUsdcQuoteProvider(
        registry=registry(),
        trade_api=trade,
        pool_api=pools,
    )
    intent = TradeIntent.create(
        asset="MEME",
        side=TradeSide.BUY,
        notional_usd="10",
        max_slippage_bps=100,
    )

    quote = run(provider.quote(intent))

    assert quote.price_usd == Decimal("2")
    assert quote.estimated_base_amount == Decimal("5")
    assert quote.liquidity_usd == Decimal("250000")
    assert quote.price_impact_bps == 5
    assert trade.calls[0]["input_mint"] == USDC_MINT
    assert trade.calls[0]["amount_atomic"] == 10_000_000


def test_sell_uses_price_probe_then_requested_notional_route():
    asset_mint = registry().resolve("MEME").mint
    trade = FakeTradeApi(
        [
            raw_quote(
                input_mint=asset_mint,
                output_mint=USDC_MINT,
                input_atomic=1_000_000,
                output_atomic=2_000_000,
                impact="0.01",
            ),
            raw_quote(
                input_mint=asset_mint,
                output_mint=USDC_MINT,
                input_atomic=5_000_000,
                output_atomic=10_000_000,
                impact="0.08",
            ),
        ]
    )
    provider = RaydiumUsdcQuoteProvider(
        registry=registry(),
        trade_api=trade,
        pool_api=FakePoolApi("300000"),
    )
    intent = TradeIntent.create(
        asset="MEME",
        side=TradeSide.SELL,
        notional_usd="10",
        max_slippage_bps=100,
    )

    quote = run(provider.quote(intent))

    assert len(trade.calls) == 2
    assert trade.calls[0]["amount_atomic"] == 1_000_000
    assert trade.calls[1]["amount_atomic"] == 5_000_000
    assert quote.estimated_base_amount == Decimal("5")
    assert quote.price_usd == Decimal("2")
    assert quote.price_impact_bps == 8
