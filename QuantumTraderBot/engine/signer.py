from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SigningDisabledError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SignedTransaction:
    transaction_base64: str
    signer_pubkey: str


class IsolatedSigner(Protocol):
    """Signer capability boundary.

    The trading engine receives only this capability. It never accepts seed
    phrases, private-key strings, keypair arrays, or filesystem key paths.
    """

    @property
    def public_key(self) -> str:
        ...

    async def sign_transaction(
        self,
        unsigned_transaction_base64: str,
    ) -> SignedTransaction:
        ...


@dataclass(slots=True)
class DisabledSigner:
    public_key: str = "disabled"

    async def sign_transaction(
        self,
        unsigned_transaction_base64: str,
    ) -> SignedTransaction:
        raise SigningDisabledError("signing is disabled")
