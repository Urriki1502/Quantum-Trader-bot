import asyncio

from engine.solana_rpc import SolanaRpcClient


class FakeTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def request_json(self, method, url, *, params=None, json_body=None, timeout=10.0):
        self.calls.append(json_body)
        return self.response


def test_get_transaction_requests_json_parsed_confirmed_v0():
    transport = FakeTransport(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"slot": 123, "meta": {"err": None, "fee": 5000}},
        }
    )
    client = SolanaRpcClient(transport=transport)

    result = asyncio.run(client.get_transaction("signature"))

    assert result["slot"] == 123
    call = transport.calls[0]
    assert call["method"] == "getTransaction"
    assert call["params"][0] == "signature"
    assert call["params"][1]["encoding"] == "jsonParsed"
    assert call["params"][1]["commitment"] == "confirmed"
    assert call["params"][1]["maxSupportedTransactionVersion"] == 0


def test_get_transaction_allows_rpc_null_until_body_is_available():
    transport = FakeTransport(
        {"jsonrpc": "2.0", "id": 1, "result": None}
    )
    client = SolanaRpcClient(transport=transport)

    assert asyncio.run(client.get_transaction("signature")) is None
