# Attempt-local budget ledger

One `Ledger` belongs to one bounded recipe attempt. All parallel branches must use
that same instance. The host provides limits, an absolute deadline, a pure clock
callback, and pricing in integer cost units. The clock must be concurrency-safe and
must not call back into the ledger. No limit, provider price or retry is implicit.

Reserve one retrieval or model call immediately before dispatch. A failed admission
does not consume any dimension. A successful admission consumes one call and holds
the declared maximum input/output tokens and known cost atomically. The adapter
must enforce its declared maximum usage; accounting cannot stop a remote provider
which ignores that contract. Pass the attempt deadline and any earlier parent
deadline to every actual I/O call.

Settle the returned lease once after completion, including errors and cancellation.
Known actual usage refunds only unused tokens/cost; admitted calls are never
refunded. Unknown usage retains the complete reservation. Actual usage exceeding
the reservation returns an error and retains the reservation; the recipe must
report failure rather than claim budget compliance. Lease copies share settlement
state and cannot refund twice. Settlement has no context requirement so canceled
calls can still be accounted for.

Required cost policy rejects unknown prices before admission. Explicit advisory
policy allows a zero unknown-cost reservation and marks the snapshot UnknownCost;
it cannot claim to enforce a cost cap for that call. Cost settlement for an unknown
price must remain unknown. Snapshot values contain no query, credentials, content
or mutable domain metadata.

This package does not dispatch work, perform pricing lookup, schedule workers,
retry calls, maintain organization quotas or implement billing. Recipe dispatch
integration, typed stage evidence and comparative experiments remain required.

`Deadline()` returns the immutable configured attempt deadline. `Context(parent)`
bounds cooperative callbacks by the remaining time measured with the host clock;
its timer uses elapsed wall time and inherits any earlier parent deadline. Reserve
rechecks the host deadline atomically. This does not stop uncooperative callbacks.
