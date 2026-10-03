import asyncio

import pytest

from engine.signer import DisabledSigner, SigningDisabledError


def test_disabled_signer_has_no_private_key_path_and_always_fails_closed():
    signer = DisabledSigner()

    assert signer.public_key == "disabled"
    with pytest.raises(SigningDisabledError):
        asyncio.run(signer.sign_transaction("dGVzdA=="))
