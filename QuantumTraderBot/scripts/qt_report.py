#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.ledger import SQLiteTradeLedger
from engine.report import build_performance_report


def parse_mark(value: str) -> tuple[str, Decimal]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("mark must be SYMBOL=PRICE")
    symbol, raw_price = value.split("=", 1)
    symbol = symbol.strip()
    if not symbol:
        raise argparse.ArgumentTypeError("mark symbol is empty")
    try:
        price = Decimal(raw_price)
    except Exception as exc:
        raise argparse.ArgumentTypeError("invalid mark price") from exc
    if price <= 0:
        raise argparse.ArgumentTypeError("mark price must be positive")
    return symbol, price


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QT v2 paper performance report")
    parser.add_argument("--db", default="qt-paper.db")
    parser.add_argument(
        "--mark",
        action="append",
        default=[],
        type=parse_mark,
        help="Current mark price as SYMBOL=PRICE; repeat for multiple open assets",
    )
    args = parser.parse_args()

    ledger = SQLiteTradeLedger(args.db)
    report = build_performance_report(
        ledger,
        prices_usd=dict(args.mark),
    )
    print(report.as_json())


if __name__ == "__main__":
    main()
