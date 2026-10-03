# QT Engine v2 — Free-First Foundation

This branch starts the replacement of the demo execution path with a deterministic,
provider-agnostic trading engine. It intentionally does **not** submit real-money
transactions.

## Design rules

1. **Free-first** — SQLite, local replay/paper execution and free/public providers can
   be used before paid infrastructure is considered.
2. **Provider-agnostic** — market-data, quote and execution vendors live behind
   adapters. Strategy/risk/state code must not depend on a specific RPC or DEX vendor.
3. **Fail closed** — stale data, failed risk checks, expired quotes and deterministic
   execution failures do not trade.
4. **Exactly-once intent semantics** — intent_id is durable and unique. Replaying
   the same intent never executes it twice.
5. **Unknown outcome is not failure** — if submission may have happened but the reply
   is lost, state becomes UNKNOWN; automatic resubmission is forbidden until a
   reconciliation adapter proves the chain outcome.
6. **Paper/live lifecycle parity** — paper execution uses the same state machine that a
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
accepted. UNKNOWN means the opposite: execution may have reached an external
system, so retrying would risk a duplicate order.

## Current foundation

- immutable trade intents
- deterministic risk policy
- quote contract
- paper execution adapter
- durable SQLite trade/event ledger
- idempotent intent handling
- unknown-outcome state
- Python 3.11 CI and regression tests

## Next implementation steps

1. Add a free Solana RPC/WebSocket market-data adapter.
2. Add a Raydium quote adapter without enabling signing.
3. Add transaction simulation and blockhash-expiry handling.
4. Add a paper portfolio/position ledger and PnL reconciliation.
5. Run continuous paper trading with metrics.
6. Only after the above remains green, add a guarded live execution adapter with
   explicit mainnet enablement, tiny hard caps and no implicit retry after ambiguous
   submission.
