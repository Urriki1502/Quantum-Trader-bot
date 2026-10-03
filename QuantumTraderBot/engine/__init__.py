"""QT Engine v2 foundation.

The engine package is intentionally provider-agnostic. It contains no live
credentials, no vendor-specific signing, and no implicit real-money execution.
Live adapters must implement the interfaces in adapters.py.
"""

from .models import (
    ExecutionReceipt,
    Quote,
    TradeIntent,
    TradeSide,
    TradeState,
)
from .risk import RiskDecision, RiskEngine, RiskPolicy, RiskSnapshot
from .ledger import (
    InsufficientCashError,
    InsufficientPositionError,
    PaperAccountSnapshot,
    EquitySnapshot,
    ExecutionAttemptSnapshot,
    RuntimeSafetySnapshot,
    PortfolioFillResult,
    PortfolioMetrics,
    PositionSnapshot,
    SQLiteTradeLedger,
)
from .adapters import ExecutionAdapter, PaperExecutionAdapter, QuoteProvider, StaticQuoteProvider
from .service import TradingEngine, EngineResult
from .assets import AssetRegistry, AssetSpec, USDC_MINT
from .raydium_quote_provider import RaydiumUsdcQuoteProvider
from .journal import SQLiteMarketJournal
from .market import MarketTick, RaydiumPollingMarketSource
from .session import PaperTradingSession, SessionDecision
from .preflight import (
    PreflightResult,
    PreflightSimulationError,
    RaydiumPreflightSimulator,
    TransactionSimulation,
)
from .report import PerformanceReport, build_performance_report
from .replay import (
    JsonlMarketReplay,
    ReplayFormatError,
    ReplayQuoteProvider,
    ReplaySummary,
    run_replay,
)
from .fill import (
    FillReconciliationError,
    ObservedSwapFill,
    SolanaSwapFillReconciler,
    observed_fill_to_receipt,
)
from .tx_identity import (
    SerializedTransactionIdentity,
    TransactionIdentityError,
    transaction_identity_from_base64,
)
from .outcome import (
    BlockhashCheck,
    ExpiredBlockhashError,
    OutcomeResolution,
    OutcomeState,
    SolanaBlockhashGuard,
    SolanaOutcomeReconciler,
)
from .live_guard import (
    GuardedLiveExecutionBoundary,
    LivePreparationError,
    PreparedLiveExecution,
    SignedLiveExecution,
    UnsignedExecutionBundle,
)
from .safety import LiveExecutionPolicy, LiveSafetyViolation
from .signer import (
    DisabledSigner,
    IsolatedSigner,
    SignedTransaction,
    SigningDisabledError,
)
from .signals import (
    MovingAverageCrossStrategy,
    SignalAction,
    StrategyContext,
    StrategySignal,
)

__all__ = [
    "AssetRegistry",
    "AssetSpec",
    "EngineResult",
    "ExecutionAdapter",
    "ExecutionAttemptSnapshot",
    "ExecutionReceipt",
    "GuardedLiveExecutionBoundary",
    "FillReconciliationError",
    "InsufficientCashError",
    "InsufficientPositionError",
    "IsolatedSigner",
    "PaperAccountSnapshot",
    "EquitySnapshot",
    "BlockhashCheck",
    "ExpiredBlockhashError",
    "ObservedSwapFill",
    "OutcomeResolution",
    "OutcomeState",
    "PaperExecutionAdapter",
    "PortfolioFillResult",
    "PortfolioMetrics",
    "PositionSnapshot",
    "Quote",
    "QuoteProvider",
    "RiskDecision",
    "RiskEngine",
    "RiskPolicy",
    "RiskSnapshot",
    "RuntimeSafetySnapshot",
    "RaydiumUsdcQuoteProvider",
    "ReplayFormatError",
    "ReplayQuoteProvider",
    "ReplaySummary",
    "RaydiumPollingMarketSource",
    "RaydiumPreflightSimulator",
    "SQLiteMarketJournal",
    "SQLiteTradeLedger",
    "StaticQuoteProvider",
    "DisabledSigner",
    "JsonlMarketReplay",
    "LiveExecutionPolicy",
    "LivePreparationError",
    "LiveSafetyViolation",
    "MarketTick",
    "MovingAverageCrossStrategy",
    "PaperTradingSession",
    "PreparedLiveExecution",
    "PerformanceReport",
    "PreflightResult",
    "PreflightSimulationError",
    "SessionDecision",
    "SignedLiveExecution",
    "SignedTransaction",
    "SigningDisabledError",
    "SignalAction",
    "SerializedTransactionIdentity",
    "SolanaBlockhashGuard",
    "SolanaSwapFillReconciler",
    "SolanaOutcomeReconciler",
    "StrategyContext",
    "StrategySignal",
    "TransactionIdentityError",
    "TransactionSimulation",
    "build_performance_report",
    "run_replay",
    "observed_fill_to_receipt",
    "transaction_identity_from_base64",
    "TradeIntent",
    "TradeSide",
    "TradeState",
    "TradingEngine",
    "USDC_MINT",
    "UnsignedExecutionBundle",
]
