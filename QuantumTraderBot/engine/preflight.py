from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .raydium_api import RaydiumBuiltSwap, RaydiumSwapQuote, RaydiumTradeApiClient
from .solana_rpc import SolanaRpcClient


class PreflightSimulationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TransactionSimulation:
    index: int
    ok: bool
    err: Any
    units_consumed: int | None
    logs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreflightResult:
    built_request_id: str
    transactions: tuple[TransactionSimulation, ...]

    @property
    def all_passed(self) -> bool:
        return bool(self.transactions) and all(tx.ok for tx in self.transactions)


class RaydiumPreflightSimulator:
    """Build and simulate Raydium transactions without signing or submitting.

    Simulation deliberately uses sigVerify=false and replaceRecentBlockhash=true.
    This phase answers only: "would the built instructions execute under a fresh
    blockhash with the supplied public wallet address?" It does not prove that a
    later signed live transaction will be accepted.
    """

    def __init__(
        self,
        *,
        trade_api: RaydiumTradeApiClient,
        rpc: SolanaRpcClient,
    ) -> None:
        self.trade_api = trade_api
        self.rpc = rpc

    async def simulate_base_in(
        self,
        quote: RaydiumSwapQuote,
        *,
        wallet_pubkey: str,
        compute_unit_price_micro_lamports: int = 0,
    ) -> PreflightResult:
        built = await self.trade_api.build_swap_base_in(
            quote,
            wallet_pubkey=wallet_pubkey,
            compute_unit_price_micro_lamports=compute_unit_price_micro_lamports,
        )
        if not built.transactions_base64:
            raise PreflightSimulationError("Raydium returned no transaction to simulate")

        simulations: list[TransactionSimulation] = []
        for index, tx_base64 in enumerate(built.transactions_base64):
            value = await self.rpc.simulate_transaction(
                tx_base64,
                replace_recent_blockhash=True,
                sig_verify=False,
            )
            err = value.get("err")
            units = value.get("unitsConsumed")
            logs = value.get("logs") or ()
            if not isinstance(logs, Sequence) or isinstance(logs, (str, bytes)):
                logs = ()
            simulations.append(
                TransactionSimulation(
                    index=index,
                    ok=err is None,
                    err=err,
                    units_consumed=int(units) if units is not None else None,
                    logs=tuple(str(item) for item in logs),
                )
            )

        return PreflightResult(
            built_request_id=built.request_id,
            transactions=tuple(simulations),
        )

    @staticmethod
    def require_pass(result: PreflightResult) -> None:
        if not result.all_passed:
            failed = [tx.index for tx in result.transactions if not tx.ok]
            raise PreflightSimulationError(
                f"preflight failed for transaction indexes: {failed}"
            )
