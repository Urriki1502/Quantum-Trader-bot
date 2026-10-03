# QT Engine v2 — Free-First Foundation

This branch replaces the old demo execution semantics with a deterministic,
provider-agnostic trading engine. It intentionally does **not** submit
real-money transactions.

## Design rules

1. **Free-first** — SQLite, local replay/paper execution and free/public providers can
   be used before paid infrastructure is considered.
2. **Provider-agnostic** — market-data, quote and execution vendors live behind
   adapters. Strategy/risk/state code must not depend on a specific RPC or DEX vendor.
3. **Fail closed** — stale data, failed risk checks, expired quotes and deterministic
   execution failures do not trade.
4. **Exactly-once intent semantics** — intent_id is durable and unique. Replaying
   the same intent never executes it twice.
5. **Exactly-once portfolio accounting** — execution_id is accounted once; a confirmed
   fill and the RECONCILED transition commit atomically.
6. **Unknown outcome is not failure** — if submission may have happened but the reply
   is lost, state becomes UNKNOWN; automatic resubmission is forbidden until a
   reconciliation adapter proves the chain outcome.
7. **Paper/live lifecycle parity** — paper execution uses the same state machine that a
   future live adapter must implement.

## State machine

    CREATED
      ├─> REJECTED
      └─> RISK_APPROVED
             └─> QUOTED
                    ├─> REJECTED
                    └─> EXECUTION_PENDING
                           ├─> FAILED
                           ├─> UNKNOWN
                           └─> SUBMITTED
                                  ├─> UNKNOWN
                                  └─> CONFIRMED
                                         └─> RECONCILED

FAILED means the engine has a deterministic reason to believe no transaction was
accepted. UNKNOWN means execution may have reached an external system, so retrying
would risk a duplicate order. CONFIRMED but not RECONCILED is a recoverable accounting
boundary; startup recovery completes it without resubmitting.

## Current foundation

- immutable trade intents
- deterministic risk policy and daily-loss breaker
- quote contract
- paper execution adapter
- durable SQLite trade/event ledger
- durable positions and realized PnL
- fee-aware weighted average cost basis
- sell-position validation
- idempotent intent handling
- idempotent portfolio fill accounting
- restart recovery for confirmed-but-unreconciled fills
- unknown-outcome state
- free Solana JSON-RPC read/simulation client
- Raydium public quote + unsigned transaction-build client
- Python 3.11 CI and regression tests

## Next implementation steps

1. Add a mint/decimal registry and convert Raydium USDC routes into normalized engine quotes.
2. Add paper market sessions driven by real read-only quotes.
3. Add transaction simulation and blockhash-expiry evidence to the future live adapter contract.
4. Add persistent equity/drawdown metrics and replay reports.
5. Run continuous paper trading with free/public infrastructure.
6. Only after the above remains green, add a guarded live execution adapter with
   explicit mainnet enablement, tiny hard caps and no implicit retry after ambiguous
   submission.
