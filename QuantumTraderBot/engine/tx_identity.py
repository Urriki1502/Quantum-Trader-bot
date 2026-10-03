from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass


class TransactionIdentityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SerializedTransactionIdentity:
    sha256: str
    byte_length: int

    @property
    def canonical(self) -> str:
        return f"sha256:{self.sha256}"


def transaction_identity_from_base64(transaction_base64: str) -> SerializedTransactionIdentity:
    """Hash the exact serialized transaction bytes.

    For a future live path this function must be called *after* signing. The
    resulting identity is stable across RPC retries because it describes the
    exact bytes, not a quote ID or local timestamp.
    """

    if not transaction_base64:
        raise TransactionIdentityError("transaction_base64 must be non-empty")
    try:
        raw = base64.b64decode(transaction_base64, validate=True)
    except Exception as exc:
        raise TransactionIdentityError("transaction is not valid base64") from exc
    if not raw:
        raise TransactionIdentityError("serialized transaction is empty")
    return SerializedTransactionIdentity(
        sha256=hashlib.sha256(raw).hexdigest(),
        byte_length=len(raw),
    )
