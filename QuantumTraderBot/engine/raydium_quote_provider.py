from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal, ROUND_CEILING, ROUND_DOWN
from uuid import uuid4

from .assets import AssetRegistry, USDC_MINT
from .models import Quote, TradeIntent, TradeSide
from .raydium_api import RaydiumPoolApiClient, RaydiumTradeApiClient


USDC_DECIMALS = 6


@dataclass(slots=True)
class RaydiumUsdcQuoteProvider:
    """Normalize live, read-only Raydium routes into the QT quote contract.

    The provider uses USDC as the accounting quote currency. It performs no
    signing and no transaction submission.
    """

    registry: AssetRegistry
    trade_api: RaydiumTradeApiClient
    pool_api: RaydiumPoolApiClient
    quote_ttl_seconds: int = 10
    provider_name: str = "raydium-public-usdc"

    async def quote(self, intent: TradeIntent) -> Quote:
        asset = self.registry.resolve(intent.asset)

        if intent.side is TradeSide.BUY:
            return await self._buy_quote(intent, asset.mint, asset.decimals)
        return await self._sell_quote(intent, asset.mint, asset.decimals)

    async def _buy_quote(
        self,
        intent: TradeIntent,
        asset_mint: str,
        asset_decimals: int,
    ) -> Quote:
        input_atomic = self._usd_to_usdc_atomic(intent.notional_usd)
        raw = await self.trade_api.quote_base_in(
            input_mint=USDC_MINT,
            output_mint=asset_mint,
            amount_atomic=input_atomic,
            slippage_bps=intent.max_slippage_bps,
        )
        base_amount = self._from_atomic(raw.output_amount_atomic, asset_decimals)
        if base_amount <= 0:
            raise ValueError("Raydium returned zero output amount")

        actual_input_usd = self._from_atomic(raw.input_amount_atomic, USDC_DECIMALS)
        price = actual_input_usd / base_amount
        liquidity = await self.pool_api.pair_liquidity_usd(asset_mint, USDC_MINT)

        return self._normalize(
            intent=intent,
            price=price,
            base_amount=base_amount,
            liquidity=liquidity,
            price_impact_pct=raw.price_impact_pct,
            created_at=raw.created_at,
        )

    async def _sell_quote(
        self,
        intent: TradeIntent,
        asset_mint: str,
        asset_decimals: int,
    ) -> Quote:
        one_token_atomic = 10 ** asset_decimals
        price_probe = await self.trade_api.quote_base_in(
            input_mint=asset_mint,
            output_mint=USDC_MINT,
            amount_atomic=one_token_atomic,
            slippage_bps=intent.max_slippage_bps,
        )
        one_token_usdc = self._from_atomic(
            price_probe.output_amount_atomic,
            USDC_DECIMALS,
        )
        if one_token_usdc <= 0:
            raise ValueError("Raydium returned zero price probe output")

        target_base = intent.notional_usd / one_token_usdc
        input_atomic = int(
            (target_base * (Decimal(10) ** asset_decimals)).to_integral_value(
                rounding=ROUND_DOWN
            )
        )
        if input_atomic <= 0:
            raise ValueError("sell notional is below one atomic token unit")

        raw = await self.trade_api.quote_base_in(
            input_mint=asset_mint,
            output_mint=USDC_MINT,
            amount_atomic=input_atomic,
            slippage_bps=intent.max_slippage_bps,
        )
        base_amount = self._from_atomic(raw.input_amount_atomic, asset_decimals)
        output_usd = self._from_atomic(raw.output_amount_atomic, USDC_DECIMALS)
        if base_amount <= 0 or output_usd <= 0:
            raise ValueError("Raydium returned invalid sell quote amounts")
        price = output_usd / base_amount
        liquidity = await self.pool_api.pair_liquidity_usd(asset_mint, USDC_MINT)

        return self._normalize(
            intent=intent,
            price=price,
            base_amount=base_amount,
            liquidity=liquidity,
            price_impact_pct=raw.price_impact_pct,
            created_at=raw.created_at,
        )

    def _normalize(
        self,
        *,
        intent: TradeIntent,
        price: Decimal,
        base_amount: Decimal,
        liquidity: Decimal,
        price_impact_pct: Decimal,
        created_at,
    ) -> Quote:
        impact_bps = int(
            (Decimal(str(price_impact_pct)) * Decimal("100")).to_integral_value(
                rounding=ROUND_CEILING
            )
        )
        return Quote(
            quote_id=str(uuid4()),
            provider=self.provider_name,
            asset=intent.asset,
            side=intent.side,
            price_usd=price,
            estimated_base_amount=base_amount,
            notional_usd=intent.notional_usd,
            liquidity_usd=liquidity,
            price_impact_bps=max(0, impact_bps),
            estimated_fee_usd=Decimal("0"),
            created_at=created_at,
            expires_at=created_at + timedelta(seconds=self.quote_ttl_seconds),
        )

    @staticmethod
    def _usd_to_usdc_atomic(value: Decimal) -> int:
        atomic = int(
            (Decimal(str(value)) * (Decimal(10) ** USDC_DECIMALS)).to_integral_value(
                rounding=ROUND_DOWN
            )
        )
        if atomic <= 0:
            raise ValueError("USD notional is below one USDC atomic unit")
        return atomic

    @staticmethod
    def _from_atomic(value: int, decimals: int) -> Decimal:
        return Decimal(value) / (Decimal(10) ** decimals)
