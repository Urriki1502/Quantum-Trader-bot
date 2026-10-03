from __future__ import annotations

from dataclasses import dataclass


USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


@dataclass(frozen=True, slots=True)
class AssetSpec:
    symbol: str
    mint: str
    decimals: int

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if not self.mint.strip():
            raise ValueError("mint must be non-empty")
        if self.decimals < 0 or self.decimals > 18:
            raise ValueError("decimals must be between 0 and 18")


class AssetRegistry:
    def __init__(self, assets: list[AssetSpec] | tuple[AssetSpec, ...] = ()) -> None:
        self._by_symbol: dict[str, AssetSpec] = {}
        self._by_mint: dict[str, AssetSpec] = {}
        for asset in assets:
            self.add(asset)

    def add(self, asset: AssetSpec) -> None:
        symbol = asset.symbol.upper()
        if symbol in self._by_symbol:
            raise ValueError(f"duplicate asset symbol: {asset.symbol}")
        if asset.mint in self._by_mint:
            raise ValueError(f"duplicate asset mint: {asset.mint}")
        self._by_symbol[symbol] = asset
        self._by_mint[asset.mint] = asset

    def resolve(self, key: str) -> AssetSpec:
        if key in self._by_mint:
            return self._by_mint[key]
        spec = self._by_symbol.get(key.upper())
        if spec is None:
            raise KeyError(f"asset not registered: {key}")
        return spec
