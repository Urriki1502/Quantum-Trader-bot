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
- durable paper cash / buying-power accounting
- mark-to-market equity snapshots and max drawdown
- UTC daily realized-PnL circuit-breaker input
- durable market-event journal with replay de-duplication
- deterministic event-derived paper intent IDs
- strategy interface plus a reference moving-average cross strategy
- continuous free/public Raydium paper runner
- mark-to-market JSON performance reports
- unsigned Raydium build + Solana preflight simulation gate
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

1. Run longer paper sessions on free/public Raydium data and collect evidence.
2. Add replay reports, equity curves and strategy attribution.
3. Add blockhash-expiry and signed-transaction identity evidence to the future live adapter contract.
4. Add external-outcome reconciliation for SUBMITTED/UNKNOWN states.
5. Only after the above remains green, add a guarded live execution adapter with
   explicit mainnet enablement, tiny hard caps and no implicit retry after ambiguous
   submission.
