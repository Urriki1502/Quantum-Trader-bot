from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .solana_rpc import SolanaRpcClient


class OutcomeState(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    EXPIRED = "expired"


class ExpiredBlockhashError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BlockhashCheck:
    blockhash: str
    last_valid_block_height: int
    current_block_height: int
    rpc_reports_valid: bool

    @property
    def valid_for_submission(self) -> bool:
        return (
            self.current_block_height <= self.last_valid_block_height
            and self.rpc_reports_valid
        )


@dataclass(frozen=True, slots=True)
class OutcomeResolution:
    signature: str
    state: OutcomeState
    found: bool
    confirmation_status: str | None
    err: Any
    current_block_height: int | None
    last_valid_block_height: int


class SolanaBlockhashGuard:
    def __init__(self, rpc: SolanaRpcClient) -> None:
        self.rpc = rpc

    async def check(
        self,
        blockhash: str,
        last_valid_block_height: int,
    ) -> BlockhashCheck:
        current_height = await self.rpc.get_block_height(commitment="confirmed")
        rpc_valid = await self.rpc.is_blockhash_valid(
            blockhash,
            commitment="confirmed",
        )
        return BlockhashCheck(
            blockhash=blockhash,
            last_valid_block_height=last_valid_block_height,
            current_block_height=current_height,
            rpc_reports_valid=rpc_valid,
        )

    async def require_valid(
        self,
        blockhash: str,
        last_valid_block_height: int,
    ) -> BlockhashCheck:
        check = await self.check(blockhash, last_valid_block_height)
        if not check.valid_for_submission:
            raise ExpiredBlockhashError(
                f"blockhash is no longer valid at height {check.current_block_height}; "
                f"last valid height was {check.last_valid_block_height}"
            )
        return check


class SolanaOutcomeReconciler:
    """Resolve a submitted signature without ever resubmitting it."""

    def __init__(self, rpc: SolanaRpcClient) -> None:
        self.rpc = rpc

    async def resolve(
        self,
        signature: str,
        *,
        last_valid_block_height: int,
    ) -> OutcomeResolution:
        statuses = await self.rpc.get_signature_statuses(
            [signature],
            search_transaction_history=True,
        )
        status = statuses[0] if statuses else None

        if status is not None:
            err = status.get("err")
            confirmation = status.get("confirmationStatus")
            if err is not None:
                return OutcomeResolution(
                    signature=signature,
                    state=OutcomeState.FAILED,
                    found=True,
                    confirmation_status=str(confirmation) if confirmation else None,
                    err=err,
                    current_block_height=None,
                    last_valid_block_height=last_valid_block_height,
                )
            if confirmation in {"confirmed", "finalized"}:
                return OutcomeResolution(
                    signature=signature,
                    state=OutcomeState.CONFIRMED,
                    found=True,
                    confirmation_status=str(confirmation),
                    err=None,
                    current_block_height=None,
                    last_valid_block_height=last_valid_block_height,
                )
            return OutcomeResolution(
                signature=signature,
                state=OutcomeState.PENDING,
                found=True,
                confirmation_status=str(confirmation) if confirmation else None,
                err=None,
                current_block_height=None,
                last_valid_block_height=last_valid_block_height,
            )

        current_height = await self.rpc.get_block_height(commitment="confirmed")
        state = (
            OutcomeState.EXPIRED
            if current_height > last_valid_block_height
            else OutcomeState.PENDING
        )
        return OutcomeResolution(
            signature=signature,
            state=state,
            found=False,
            confirmation_status=None,
            err=None,
            current_block_height=current_height,
            last_valid_block_height=last_valid_block_height,
        )
