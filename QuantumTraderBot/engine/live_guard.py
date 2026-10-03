from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from .ledger import ExecutionAttemptSnapshot, SQLiteTradeLedger
from .models import Quote, TradeIntent, TradeState
from .outcome import SolanaBlockhashGuard
from .safety import LiveExecutionPolicy, LiveSafetyViolation
from .signer import IsolatedSigner
from .solana_rpc import SolanaRpcClient
from .tx_identity import transaction_identity_from_base64


class LivePreparationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class UnsignedExecutionBundle:
    request_id: str
    transactions_base64: tuple[str, ...]
    recent_blockhash: str
    last_valid_block_height: int


class LiveTransactionBuilder(Protocol):
    async def build(
        self,
        intent: TradeIntent,
        quote: Quote,
        *,
        wallet_pubkey: str,
    ) -> UnsignedExecutionBundle:
        ...


@dataclass(frozen=True, slots=True)
class PreparedLiveExecution:
    intent_id: str
    quote_id: str
    wallet_pubkey: str
    builder_request_id: str
    unsigned_transaction_base64: str
    recent_blockhash: str
    last_valid_block_height: int
    simulation_units_consumed: int | None
    submission_authorized: bool


@dataclass(frozen=True, slots=True)
class SignedLiveExecution:
    plan: PreparedLiveExecution
    signed_transaction_base64: str
    tx_identity: str
    attempt: ExecutionAttemptSnapshot


class GuardedLiveExecutionBoundary:
    """Prepare/sign boundary with no transaction-submission capability.

    Deliberately absent: sendTransaction/sendRawTransaction. Even after a plan
    is signed, this class cannot broadcast it. A future submitter must be a
    separate capability and go through outcome reconciliation.
    """

    def __init__(
        self,
        *,
        ledger: SQLiteTradeLedger,
        policy: LiveExecutionPolicy,
        builder: LiveTransactionBuilder,
        rpc: SolanaRpcClient,
        signer: IsolatedSigner,
    ) -> None:
        self.ledger = ledger
        self.policy = policy
        self.builder = builder
        self.rpc = rpc
        self.signer = signer
        self.blockhash_guard = SolanaBlockhashGuard(rpc)

    async def prepare(
        self,
        intent: TradeIntent,
        quote: Quote,
    ) -> PreparedLiveExecution:
        self.policy.require_intent_allowed(intent)
        self._require_trade_execution_pending(intent.intent_id)
        if quote.quote_id == "" or quote.asset != intent.asset or quote.side != intent.side:
            raise LivePreparationError("quote does not match intent")

        bundle = await self.builder.build(
            intent,
            quote,
            wallet_pubkey=self.signer.public_key,
        )
        if not bundle.transactions_base64:
            raise LivePreparationError("builder returned no transactions")
        if len(bundle.transactions_base64) > self.policy.max_transactions_per_intent:
            raise LivePreparationError(
                "multi-transaction live execution is outside the validated scope"
            )

        await self.blockhash_guard.require_valid(
            bundle.recent_blockhash,
            bundle.last_valid_block_height,
        )

        transaction = bundle.transactions_base64[0]
        simulation = await self.rpc.simulate_transaction(
            transaction,
            commitment="confirmed",
            replace_recent_blockhash=False,
            sig_verify=False,
        )
        if self.policy.require_preflight and simulation.get("err") is not None:
            raise LivePreparationError(
                f"preflight simulation failed: {simulation.get('err')}"
            )

        controls = self.ledger.get_runtime_safety()
        authorized = (
            controls.live_submission_enabled
            and not controls.kill_switch_engaged
        )

        units = simulation.get("unitsConsumed")
        return PreparedLiveExecution(
            intent_id=intent.intent_id,
            quote_id=quote.quote_id,
            wallet_pubkey=self.signer.public_key,
            builder_request_id=bundle.request_id,
            unsigned_transaction_base64=transaction,
            recent_blockhash=bundle.recent_blockhash,
            last_valid_block_height=bundle.last_valid_block_height,
            simulation_units_consumed=int(units) if units is not None else None,
            submission_authorized=authorized,
        )

    async def sign(
        self,
        plan: PreparedLiveExecution,
    ) -> SignedLiveExecution:
        self._require_trade_execution_pending(plan.intent_id)
        controls = self.ledger.get_runtime_safety()
        self.policy.require_signing_authorized(controls)

        await self.blockhash_guard.require_valid(
            plan.recent_blockhash,
            plan.last_valid_block_height,
        )

        signed = await self.signer.sign_transaction(
            plan.unsigned_transaction_base64
        )
        if signed.signer_pubkey != plan.wallet_pubkey:
            raise LivePreparationError(
                "signer returned a transaction for a different public key"
            )

        identity = transaction_identity_from_base64(
            signed.transaction_base64
        ).canonical
        attempt_seed = (
            f"{plan.intent_id}|{plan.quote_id}|{plan.recent_blockhash}"
        ).encode()
        attempt_id = "live-attempt:" + hashlib.sha256(attempt_seed).hexdigest()

        attempt = self.ledger.reserve_execution_attempt(
            plan.intent_id,
            attempt_id=attempt_id,
            tx_identity=identity,
            recent_blockhash=plan.recent_blockhash,
            last_valid_block_height=plan.last_valid_block_height,
        )
        return SignedLiveExecution(
            plan=plan,
            signed_transaction_base64=signed.transaction_base64,
            tx_identity=identity,
            attempt=attempt,
        )

    def _require_trade_execution_pending(self, intent_id: str) -> None:
        row = self.ledger.get_trade(intent_id)
        if row is None:
            raise LivePreparationError("intent is not present in the durable ledger")
        if TradeState(row["state"]) is not TradeState.EXECUTION_PENDING:
            raise LivePreparationError(
                f"live preparation requires execution_pending, got {row['state']}"
            )
