# External observation contract

This independent `example.com/ragyconsumer` module uses actual in-memory BM25,
public cache, and typed retrieval pipeline with host-owned metadata. Run with
`GOWORK=off go test -race ./observation_contract` from `examples/conformance`.

The tests exercise disabled observation, failing diagnostic exporters, cache hits,
fallback and rescue branches, partial evidence, unsupported execution, canceled
admission and concurrent session use. Exporter errors leave actual retrieval
results and dispatch counts unchanged. Accepted start/completion pairs remain
bounded and model usage remains unavailable for model-free BM25. JSON serialization
of collected events must omit query, content, metadata, identifiers, cache identity
and raw exporter/provider errors.

These diagnostic events contain no arbitrary host strings. Required immutable
evidence recording is a separate contract; diagnostic callback failures do not
establish that a required evidence sink persisted a record or that remote billing
was canceled.
