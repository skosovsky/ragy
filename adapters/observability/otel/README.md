# OTel adapter

Existing capability wrappers preserve request/result contracts, capability admission, and exactly-once dispatch. Spans carry fixed sanitized outcome/error classification, result counts, and actually observed encoding usage. Error bodies, query/content, metadata, document IDs, host hashes, model identities, provider URLs, and auth data are never exported. An underlying error is returned unchanged to the caller; it is never passed to `RecordError`.

`NewObserver(tracer)` implements core `observation.Observer`. Enable it with `observation.New(Config{MaxEvents: ..., Observer: ...})` and `observation.WithSession`. No second capability wrapper API is introduced. The bounded core session serializes cooperative callbacks and isolates callback errors/panics; exporter failure never retries an operation. Use a host-configured OTel span processor to bound exporter work; the adapter has no queue, retry service, or metrics. OTel processor callbacks are cooperative and must return promptly.

The observer creates a span only for a real completion, with local elapsed start/end timestamps. No started span map is retained, so shared exporters and sessions restarting ordinal 1 cannot collide or leak unfinished operations. Operation/parent/query/branch ordinals are span attributes; the parent ordinal refers to the core session, while the OTel parent is the span from the supplied context. Unknown query/branch/count/usage has an explicit `.known=false` attribute and no fabricated value. Counters over signed OTel integer capacity saturate and carry `.saturated=true`. No ordinal is a metric label. Canceled denotes local observation only, not confirmed remote cancellation or billing reversal.

Semantic conventions use the **OpenTelemetry 1.38.0 GenAI development subset**, with `gen_ai.operation.name=embeddings` and known `gen_ai.usage.input_tokens` only for identified encoding operations; `error.type` follows the general error convention. These generic library spans do not assert complete provider/client GenAI conformance: provider/model are unavailable and intentionally omitted; wrapper span names and all library-specific diagnostics are `ragy.*`. A generic model port does not prove a chat operation, so no `gen_ai.operation.name=chat` is guessed. There is no content export opt-in in this diagnostic adapter; controlled evidence export belongs to the core evidence API.

The current official docs were checked during TASK-17. GenAI has moved out of the current 1.44.0 general semantic conventions to a separate development repository with no release yet. A released, explicit 1.38.0 subset is selected rather than claiming conformance to a moving unreleased branch:

- [Pinned 1.38.0 GenAI span conventions](https://github.com/open-telemetry/semantic-conventions/blob/v1.38.0/docs/gen-ai/gen-ai-spans.md)
- [Current official semantic conventions and GenAI move](https://opentelemetry.io/docs/specs/semconv/)
- [Current GenAI development span conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)
- [GenAI repository releases](https://github.com/open-telemetry/semantic-conventions-genai/releases)
