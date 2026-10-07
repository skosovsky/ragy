# Errors, delivery and recovery

Inspect protection before inspecting documents or considering partial success. `access.IsProtectionFailure(err)` recognizes joined failures. Protection, expired/revoked authority and parent cancellation suppress payload and payload-bearing journals. A nonempty error-carried ResultSet never restores a separately returned empty result. Use only the returned result as payload authority; inspect errors and read coverage before accepting incomplete results.

| Condition | Delivery and next action |
|---|---|
| Read protection failure, including joined protection/deadline | Zero payload; stop. Re-establish a valid host authority/binding explicitly. Do not rescue or retry around protection. |
| Parent context canceled/expired | Suppress delivery. A child/local time limit does not override the parent. |
| Ordinary retrieval failure with a returned nonempty set | Explicit partial contract may retain admitted documents with the error. Decide in the host whether they meet the use case. |
| `PartialFailureError` with returned empty set | Empty remains authoritative; inspect failed branches. Do not take documents from its diagnostic Result field. |
| Pure attempt-local/ledger deadline with valid parent and read, no independent callback/settlement failure | Core text recipe may return its documented bounded partial outcome. Extraction and graph recipes retain their zero-output deadline policy. |
| Callback protocol/error/protection or usage overrun plus local expiry | Failure, not bounded success. Already observed causes remain discoverable. No retry/refund or late payload delivery. |
| Invalid configuration/input or unsupported capability | Fix the request/configuration or choose an explicitly capable port. Unsupported scope/publication fails closed. |
| Dispatch/publication/cleanup outcome unknown | Preserve exact operation identity and durable evidence. Inspect/reconcile before another dispatch or destructive action. |
| Read-only inspection returns outcome unknown | Observation is inconclusive; it does not prove that a destructive operation occurred. Repeat explicit inspection under host policy. |
| Optimistic generation/inventory conflict | Refresh captured state and reconsider the intended operation. The library does not sleep, retry or schedule automatically. |

`errors.Is(err, context.DeadlineExceeded)` alone is insufficient to classify a bounded recipe outcome. A host callback deadline is pure local expiry only when it comes solely from the supplied operation context, that limit actually expired and parent/read gates remain valid. An independent joined fault remains failure. `RunObserved` may preserve an owned failed journal only for its documented ordinary-failure path with valid read authority; `Run`/`RunOwn` expose no failed result. Protection suppresses the whole journal.

Every acquired budget lease is settled once, including cancellation after reservation and before dispatch. Actual known usage is retained after a failed call; unknown usage conservatively retains reservations. A denied reservation creates no lease or dispatch. Calls are not refunded and settlement does not renew admission. Inspect [recipe budgets](../recipe/budget/README.md) and [recipe](../recipe/README.md) for precise result fields and units.

Raw administration and scoped reads are different operations. A RawStore or backend administration method does not establish tenant IAM, source admission or publication consistency. Select an explicit Read binding for retrieval; capture/publish/pin through the [lifecycle protocol](../lifecycle/README.md) when snapshot consistency is required. A tenant search filter in the local quickstart is not authorization.

For release unknown/partial/collision states use the [release runbook](release/runbook.md). For lifecycle replay, permanent artifact reservations, pins and cleanup eligibility use the lifecycle guide. Neither successful command exit nor a freshness check establishes future availability, erasure of external sources or a distributed commit.

Callbacks and error methods are cooperative host code. Cancellation gates stop later work and reject returned payload; they cannot preempt arbitrary callbacks. Avoid formatting arbitrary raw provider/host errors into public logs. Use classification and a host-approved diagnostic policy.
