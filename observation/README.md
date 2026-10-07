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

## Session accounting and units

| Counter | Cumulative session unit |
|---|---|
| `Stats.Events` | Attempted start/end callback invocations; incremented before Observe, including callback errors/panics |
| `Stats.Dropped` | Rejected operation pairs due to capacity or invalid stage; no callbacks emitted |
| `Stats.Failures` | Callback attempts returning error or panicking; not operation retries or remote exporter failure acknowledgments |
| `MaxEvents` | Lifetime callback budget, two reserved per accepted operation; odd capacity may leave one unused slot |

For capacity2, one operation emits two callback attempts; a second Begin is one
dropped pair, not two dropped events. Failed callbacks still count toward Events.
There is no reset or implicit session drain. Every accepted span must be ended;
reserving a terminal callback does not fabricate End when a caller abandons it.
Cancellation does not auto-close a span or interrupt a blocked callback. Callback
latency and host methods remain cooperative; bounds limit cardinality, not elapsed
callback execution time.

`Completion.Count` uses the instrumented stage's declared unit: returned retrieval
records, selected evidence elements, encoding records/matrices or stage-specific
work. It is not automatically a document count or token count. Lifecycle or planner
boundaries without observed cardinality retain unknown. `Usage.InputTokens` and
`OutputTokens` are observed tokens; `BilledUnits` is the supplying provider/host
accounting unit. Those units are not interchangeable and imply no price. `Known=false`
normalizes Value to zero; observed zero is `Known=true`. Host-supplied numeric facts
must be truthful. Fixed enums reject or normalize invalid numeric values and accept
no arbitrary labels, query/content/metadata/identities or raw error text.

`Classify` never calls `Error()` to format host errors. It uses `errors.Is` and
protection classification via `errors.As`: custom `Is`, `As`, `Unwrap` may execute,
block or panic. Such host methods must be bounded/cooperative. The library does
not sandbox arbitrary error objects or catch classification panics. This differs
from the explicit isolation of errors/panics in Observer callbacks.

## Optional host offload

The executable [host queue example](example_test.go) enqueues copied Event values
into a finite host-owned channel without waiting. A full queue increments its own
**event** drop counter and returns promptly; the core Dropped counter still counts
**operation pairs** rejected before callbacks. This example has no library worker;
the host owns draining, producer completion, queue closure, downstream errors and
any worker or cancellation policy. Never close the queue while callbacks can send.
Returning nil after host queue drop means core Failures remains zero; only the
host's counter records that drop. A host metry bridge can consume these same fixed
facts without adding metry, OTel or an exporter daemon to core.

The optional [OTel observer](../adapters/observability/otel/README.md) receives both
callbacks but exports only real completions: one accepted ended operation means
two core Events and one diagnostic span. Abandoned spans emit no synthetic
completion. OTel SDK processor/exporter health is host-managed; asynchronous
export failures are not automatically core Failures. Correlation ordinals remain
session-local attributes and are never metric label dimensions.
