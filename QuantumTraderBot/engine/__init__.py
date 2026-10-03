"""QT Engine v2 foundation.

The engine package is intentionally provider-agnostic. It contains no live
credentials, no vendor-specific network calls, and no implicit real-money
execution. Live adapters must implement the interfaces in adapters.py.
"""

from .models import (
    ExecutionReceipt,
    Quote,
    TradeIntent,
    TradeSide,
    TradeState,
)
from .risk import RiskDecision, RiskEngine, RiskPolicy, RiskSnapshot
from .ledger import SQLiteTradeLedger
from .adapters import ExecutionAdapter, PaperExecutionAdapter, QuoteProvider, StaticQuoteProvider
from .service import TradingEngine, EngineResult

__all__ = [
    "ExecutionAdapter",
    "ExecutionReceipt",
    "EngineResult",
    "PaperExecutionAdapter",
    "Quote",
    "QuoteProvider",
    "RiskDecision",
    "RiskEngine",
    "RiskPolicy",
    "RiskSnapshot",
    "SQLiteTradeLedger",
    "StaticQuoteProvider",
    "TradeIntent",
    "TradeSide",
    "TradeState",
    "TradingEngine",
]
