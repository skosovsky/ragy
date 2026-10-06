# Payload-free execution observation

`observation` is an optional diagnostic boundary. Core imports no OTel or provider
module. A host enables a finite session on an attempt context:

```go
session, err := observation.New(observation.Config{
    MaxEvents: 256,
    Observer: observation.ObserverFunc(func(ctx context.Context, event observation.Event) error {
        // Export the fixed, immutable event. No query, content, metadata, document
        // identity, arbitrary host label, or error message is present.
        return export(ctx, event)
    }),
})
if err != nil {
    return err
}
ctx = observation.WithSession(ctx, session)
result, err := pipeline.Execute(ctx, request)
health := session.Stats()
```

`MaxEvents` must be between 2 and 1,048,576. Begin reserves both start and end
callbacks; a rejected pair produces no fabricated completion. Callback cardinality
and outstanding spans are bounded by this capacity. Dropped pairs and exporter
errors/panics increment local health counters. They never cause operation retries,
change the result, or replace a required evidence sink's error policy.

Callbacks are synchronous, cooperative and serialized for a session. They must
return promptly and honor their context; the library supplies no background queue,
forced timeout, worker, or retries. `Stats` can be called inside a callback. A
callback receives a context with observation disabled, so ordinary nested library
calls cannot recursively export. It must not explicitly re-enable or recursively
use the same session: recursive callbacks would wait for callback serialization.
The host controls exporter latency. Elapsed time is local wall duration including
instrumentation overhead, not provider latency or a remote billing measurement.

Operation ordinals start at one; parent zero denotes a root. Query and branch
ordinals have explicit `Known` flags, so ordinal zero differs from unavailable.
Callbacks follow their actual serialization order. Each accepted operation emits
start before end, but parallel siblings have no deterministic global ordinal or
completion order. End is idempotent and safe for concurrent callers. Correlation
belongs to one session and does not provide deterministic model replay.

Counts and usage distinguish unavailable from observed zero. Unknown numeric
values are normalized to zero and stay `Known=false`. Local cancellation is an
observed local outcome: it does not assert cancellation of a remote request or
refund/zero usage. Generic retrieval and lifecycle boundaries have no remote token
or billing accounting and retain unknown usage. Fixed error classification never
reads raw error messages. A protection failure suppresses payload observations.

With no session, Begin returns the original context and a nil span without
allocation or callbacks; query/branch propagation is also a no-op. Instrumented
boundaries guard diagnostic count extraction when the span is disabled. Enabling
observation does not replay a pipeline, retrieval, model, or lifecycle dispatch.
Do not use operation/query/branch ordinals as metric labels.
