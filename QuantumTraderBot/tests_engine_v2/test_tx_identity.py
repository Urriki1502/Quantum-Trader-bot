import base64
import hashlib

import pytest

from engine.tx_identity import (
    TransactionIdentityError,
    transaction_identity_from_base64,
)


def test_transaction_identity_hashes_exact_serialized_bytes():
    raw = b"signed-transaction-bytes"
    encoded = base64.b64encode(raw).decode()

    identity = transaction_identity_from_base64(encoded)

    assert identity.sha256 == hashlib.sha256(raw).hexdigest()
    assert identity.byte_length == len(raw)
    assert identity.canonical == f"sha256:{hashlib.sha256(raw).hexdigest()}"


def test_transaction_identity_is_stable_across_calls():
    encoded = base64.b64encode(b"same bytes").decode()
    first = transaction_identity_from_base64(encoded)
    second = transaction_identity_from_base64(encoded)

    assert first == second


@pytest.mark.parametrize("value", ["", "@@@", "!!!!"])
def test_invalid_serialized_transaction_fails_closed(value):
    with pytest.raises(TransactionIdentityError):
        transaction_identity_from_base64(value)
