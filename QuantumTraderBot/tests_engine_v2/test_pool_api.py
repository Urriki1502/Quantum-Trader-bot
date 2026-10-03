import asyncio
from decimal import Decimal

from engine.raydium_api import RaydiumPoolApiClient


def run(coro):
    return asyncio.run(coro)


class FakeTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def request_json(self, method, url, *, params=None, json_body=None, timeout=10.0):
        self.calls.append((method, url, params))
        return self.response


def test_pool_liquidity_uses_largest_pool_tvl_conservatively():
    transport = FakeTransport(
        {
            "success": True,
            "data": {
                "count": 3,
                "data": [
                    {"id": "a", "tvl": 100000},
                    {"id": "b", "tvl": "25000.5"},
                    {"id": "c", "tvl": 0},
                ],
            },
        }
    )
    client = RaydiumPoolApiClient(transport=transport)

    liquidity = run(client.pair_liquidity_usd("mint-a", "mint-b"))

    assert liquidity == Decimal("100000")
    assert transport.calls[0][2]["poolSortField"] == "liquidity"
