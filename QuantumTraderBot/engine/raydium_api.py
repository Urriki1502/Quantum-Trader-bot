from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Mapping

from .http import JsonHttpTransport, UrllibJsonTransport


class RaydiumTradeApiError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RaydiumSwapQuote:
    request_id: str
    input_mint: str
    output_mint: str
    input_amount_atomic: int
    output_amount_atomic: int
    other_amount_threshold_atomic: int
    slippage_bps: int
    price_impact_pct: Decimal
    route_plan: tuple[Mapping[str, Any], ...]
    raw_data: Mapping[str, Any]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RaydiumBuiltSwap:
    request_id: str
    transactions_base64: tuple[str, ...]


class RaydiumTradeApiClient:
    """Read/build client for Raydium's public Transaction API.

    This client can quote and build unsigned transaction payloads. It does not
    hold a private key, sign, submit, or claim trade success.
    """

    def __init__(
        self,
        base_url: str = "https://transaction-v1.raydium.io",
        *,
        transport: JsonHttpTransport | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.transport = transport or UrllibJsonTransport()
        self.timeout = timeout

    async def quote_base_in(
        self,
        *,
        input_mint: str,
        output_mint: str,
        amount_atomic: int,
        slippage_bps: int,
        tx_version: str = "V0",
    ) -> RaydiumSwapQuote:
        if amount_atomic <= 0:
            raise ValueError("amount_atomic must be > 0")
        if slippage_bps < 0:
            raise ValueError("slippage_bps must be >= 0")

        envelope = await self.transport.request_json(
            "GET",
            f"{self.base_url}/compute/swap-base-in",
            params={
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": amount_atomic,
                "slippageBps": slippage_bps,
                "txVersion": tx_version,
            },
            timeout=self.timeout,
        )
        data = self._unwrap(envelope)
        route_plan = data.get("routePlan") or []
        if not isinstance(route_plan, list):
            raise RaydiumTradeApiError("Raydium quote routePlan must be a list")

        return RaydiumSwapQuote(
            request_id=str(envelope.get("id", "")),
            input_mint=str(data["inputMint"]),
            output_mint=str(data["outputMint"]),
            input_amount_atomic=int(data["inputAmount"]),
            output_amount_atomic=int(data["outputAmount"]),
            other_amount_threshold_atomic=int(data["otherAmountThreshold"]),
            slippage_bps=int(data["slippageBps"]),
            price_impact_pct=Decimal(str(data.get("priceImpactPct", "0"))),
            route_plan=tuple(route_plan),
            raw_data=dict(data),
            created_at=datetime.now(timezone.utc),
        )

    async def build_swap_base_in(
        self,
        quote: RaydiumSwapQuote,
        *,
        wallet_pubkey: str,
        compute_unit_price_micro_lamports: int,
        tx_version: str = "V0",
        wrap_sol: bool = True,
        unwrap_sol: bool = False,
        input_account: str | None = None,
        output_account: str | None = None,
        max_quote_age_seconds: int = 25,
    ) -> RaydiumBuiltSwap:
        if compute_unit_price_micro_lamports < 0:
            raise ValueError("compute unit price must be >= 0")
        age = (datetime.now(timezone.utc) - quote.created_at).total_seconds()
        if age < 0 or age > max_quote_age_seconds:
            raise RaydiumTradeApiError("refusing to build from a stale Raydium quote")

        body: dict[str, Any] = {
            "computeUnitPriceMicroLamports": str(compute_unit_price_micro_lamports),
            "swapResponse": dict(quote.raw_data),
            "txVersion": tx_version,
            "wallet": wallet_pubkey,
            "wrapSol": wrap_sol,
            "unwrapSol": unwrap_sol,
        }
        if input_account:
            body["inputAccount"] = input_account
        if output_account:
            body["outputAccount"] = output_account

        envelope = await self.transport.request_json(
            "POST",
            f"{self.base_url}/transaction/swap-base-in",
            json_body=body,
            timeout=self.timeout,
        )
        data = self._unwrap(envelope)
        if not isinstance(data, list):
            raise RaydiumTradeApiError("Raydium build response data must be a list")

        txs: list[str] = []
        for entry in data:
            if not isinstance(entry, Mapping) or not entry.get("transaction"):
                raise RaydiumTradeApiError("Raydium build response contains invalid transaction")
            txs.append(str(entry["transaction"]))
        if not txs:
            raise RaydiumTradeApiError("Raydium build response contained no transactions")

        return RaydiumBuiltSwap(
            request_id=str(envelope.get("id", "")),
            transactions_base64=tuple(txs),
        )

    @staticmethod
    def _unwrap(envelope: Mapping[str, Any]) -> Any:
        if envelope.get("success") is not True:
            message = envelope.get("msg") or envelope.get("error") or "unknown Raydium error"
            raise RaydiumTradeApiError(str(message))
        if "data" not in envelope:
            raise RaydiumTradeApiError("Raydium response missing data")
        return envelope["data"]
