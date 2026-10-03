from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .http import JsonHttpTransport, UrllibJsonTransport


class SolanaRpcError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LatestBlockhash:
    blockhash: str
    last_valid_block_height: int
    context_slot: int


class SolanaRpcClient:
    """Minimal read/simulate Solana JSON-RPC client.

    Deliberately no sendTransaction method exists in the free-foundation phase.
    """

    def __init__(
        self,
        endpoint: str = "https://api.mainnet-beta.solana.com",
        *,
        transport: JsonHttpTransport | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.endpoint = endpoint
        self.transport = transport or UrllibJsonTransport()
        self.timeout = timeout
        self._request_id = 0

    async def call(self, method: str, params: Sequence[Any] | None = None) -> Any:
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": list(params or []),
        }
        response = await self.transport.request_json(
            "POST",
            self.endpoint,
            json_body=payload,
            timeout=self.timeout,
        )
        if response.get("error") is not None:
            raise SolanaRpcError(f"{method} failed: {response['error']}")
        if "result" not in response:
            raise SolanaRpcError(f"{method} response missing result")
        return response["result"]

    async def get_health(self) -> str:
        return str(await self.call("getHealth"))

    async def get_block_height(self, *, commitment: str = "confirmed") -> int:
        return int(await self.call("getBlockHeight", [{"commitment": commitment}]))

    async def get_latest_blockhash(self, *, commitment: str = "confirmed") -> LatestBlockhash:
        result = await self.call("getLatestBlockhash", [{"commitment": commitment}])
        if not isinstance(result, Mapping):
            raise SolanaRpcError("getLatestBlockhash returned invalid result")
        value = result.get("value")
        context = result.get("context")
        if not isinstance(value, Mapping) or not isinstance(context, Mapping):
            raise SolanaRpcError("getLatestBlockhash missing context/value")
        return LatestBlockhash(
            blockhash=str(value["blockhash"]),
            last_valid_block_height=int(value["lastValidBlockHeight"]),
            context_slot=int(context["slot"]),
        )

    async def is_blockhash_valid(self, blockhash: str, *, commitment: str = "confirmed") -> bool:
        result = await self.call(
            "isBlockhashValid",
            [blockhash, {"commitment": commitment}],
        )
        if not isinstance(result, Mapping) or "value" not in result:
            raise SolanaRpcError("isBlockhashValid returned invalid result")
        return bool(result["value"])

    async def get_signature_statuses(
        self,
        signatures: Sequence[str],
        *,
        search_transaction_history: bool = True,
    ) -> list[Mapping[str, Any] | None]:
        if not signatures:
            return []
        if len(signatures) > 256:
            raise ValueError("Solana getSignatureStatuses supports at most 256 signatures")
        result = await self.call(
            "getSignatureStatuses",
            [
                list(signatures),
                {"searchTransactionHistory": search_transaction_history},
            ],
        )
        if not isinstance(result, Mapping) or not isinstance(result.get("value"), list):
            raise SolanaRpcError("getSignatureStatuses returned invalid result")
        return result["value"]

    async def simulate_transaction(
        self,
        transaction_base64: str,
        *,
        commitment: str = "confirmed",
        replace_recent_blockhash: bool = True,
        sig_verify: bool = False,
    ) -> Mapping[str, Any]:
        result = await self.call(
            "simulateTransaction",
            [
                transaction_base64,
                {
                    "encoding": "base64",
                    "commitment": commitment,
                    "replaceRecentBlockhash": replace_recent_blockhash,
                    "sigVerify": sig_verify,
                },
            ],
        )
        if not isinstance(result, Mapping) or not isinstance(result.get("value"), Mapping):
            raise SolanaRpcError("simulateTransaction returned invalid result")
        return result["value"]
