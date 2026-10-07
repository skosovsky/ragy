# Historical integration guide

Snapshot before T20 at commit 1672c0e, SHA256 of original bytes: `fe825c0cac40b90172af8b70ae9c95a8dd95e3e6a06c7f4b88d9964d3ddff4ba`. This is historical evidence; use [current integration](../integration.md).

# RAGy Integration Semantics

## Retrieval request boundary

Backends and pipeline nodes receive the complete retrieval request envelope:

- `retrieval.Query[TIntent]` for the common no-request-metadata path.
- `retrieval.Request[TIntent, TRequestMeta]` with `retrieval.NewRequestExecutionPipelineBuilder` when callers need typed request metadata and execution metadata.
- `context.Context` is reserved for cancellation, deadlines, tracing, and transport-scoped values. Do not pass business retrieval state through context values.

`Request.EffectiveText()` selects planned expanded text first, planned normalized text second, and raw request text last. Adapters should read query text through that method unless they intentionally need the raw text.
Requests may carry a prebuilt `Plan`; pipelines reuse it and do not call the configured planner again.

## Planning and routing

`ExecutionPipelineBuilder.WithPlanner` / `RequestExecutionPipelineBuilder.WithPlanner` runs `QueryPlanner` before the root graph. The resulting `PlannedQuery` is attached to the request and can carry normalized text, expanded text, typed filters, universal range constraints, diagnostics, and a cache key. `WithPlanBinder` runs after planning and before retrieval, so planned ranges/filters can be bound into request options or typed metadata without an external split.

Use `RouteSwitchNode` for explicit typed routing decisions. Route planners and fallback predicates are generic over `TIntent`, `TRequestMeta`, `TRoute`, `TSignal`, `TMeta`, and `TExecMeta`; domain categories belong to the caller's types, not to `ragy`.

Minimal end-to-end flow:

```go
type ExecMeta struct {
	Route     Route
	QueryText string
}

var localBackend retrieval.RequestExecutionBackend[Intent, RequestMeta, DocMeta, ExecMeta]
var webBackend retrieval.RequestExecutionBackend[Intent, RequestMeta, DocMeta, ExecMeta]

routeSwitch, err := retrieval.NewRequestRouteSwitchBuilder[Intent, RequestMeta, Route, Signal, DocMeta, ExecMeta](
	routePlanner,
).
	RecordDecision(func(exec ExecMeta, decision retrieval.RouteDecision[Route, Signal]) ExecMeta {
		exec.Route = decision.Route
		return exec
	}).
	Case(RouteLocal, retrieval.RequestExecutionRetrieverNode[Intent, RequestMeta, DocMeta, ExecMeta]{
		Backend: localBackend,
	}).
	Case(RouteWeb, retrieval.RequestExecutionRetrieverNode[Intent, RequestMeta, DocMeta, ExecMeta]{
		Backend: webBackend,
	}).
	FallbackOnEmpty(RouteLocal, RouteWeb).
	RescueOnError(RouteLocal, RouteWeb, allowWebRescue).
	Build()
if err != nil {
	// handle error
}

pipeline, err := retrieval.NewRequestExecutionPipelineBuilder[Intent, RequestMeta, DocMeta, ExecMeta]().
	WithExecutionSeed(func(req retrieval.Request[Intent, RequestMeta]) ExecMeta {
		return ExecMeta{QueryText: req.Text}
	}).
	WithPlanner(planner).
	WithPlanBinder(planBinder).
	WithRoot(routeSwitch).
	Build()
if err != nil {
	// handle error
}

result, err := pipeline.Execute(ctx, retrieval.Request[Intent, RequestMeta]{
	Read:    read, // immutable binding obtained from the host authorization decision
	Text:    rawText,
	Intent:  intent,
	Meta:    requestMeta,
	Options: opts,
})
if err != nil {
	// handle error
}

artifact, err := retrieval.DefaultArtifactRenderer[DocMeta]{}.Render(ctx, read, result.ResultSet, retrieval.ArtifactRenderOptions[DocMeta]{
	Budget:        budget,
	CloneMeta:     cloneDocMeta, // deep copy mutable host metadata
	DedupKey:      dedupKey,
	FormatSnippet: formatSnippet,
})
_ = artifact
_ = result.BranchTrace
```

`WithExecutionSeed` initializes `TExecMeta` from the typed request before planning and route execution. Plan binders, route switches, and execution-aware backends receive the current metadata and return the updated value. `RequestExecutionRetrieverNode` preserves incoming metadata when a backend returns a zero-value `Executed`; backends that emit side outputs should return a non-zero updated `Executed`.

## Context artifact handoff

Use `DefaultArtifactRenderer` or a custom `ArtifactRenderer` when the downstream layer needs a structured context payload instead of raw documents. The artifact preserves ordered snippets, provenance, score/rank state, budget accounting, diagnostics, rendered text, dedup policy, source formatting, and the untrusted-data boundary. Renderers receive the original read binding and context, check freshness before consumers and delivery, and require explicit metadata ownership. Supply typed source mapping for exact byte/geometry evidence; label provenance alone has unobserved precision. Renderers must stay independent from prompt, agent, or application runtimes.

## Chunk source projection

Use `SourceDescriptor`, `ChunkIdentityPolicy`, and `ProjectDocuments` to turn splitter output into storage-ready `retrieval.Document[TMeta]` values. Source-level data is projected through typed `MetadataProjector` callbacks; avoid `map[string]any` metadata outside adapter wire boundaries.

## Equal-score tie-break policy

| Layer                                 | Equal-score tie policy                              |
| ------------------------------------- | --------------------------------------------------- |
| ResultSet.Merge (same MergeKey)       | first seen wins                                     |
| Merge / Dedup / RRF output sort       | sorted MergeKey materialization → stable score sort |
| applyTopK                             | input order (stable)                                |
| BM25 rank                             | sorted doc ID materialization → stable score sort   |
| Cohere rerank                         | API index order (stable)                            |
| GroupBy / TopPerGroup group iteration | sorted group keys (ascending)                       |

Source: README tie-breaking section.

## Score semantics

- Native scores retain their finite values, including negative values and values above one. Elasticsearch, Qdrant, BM25 and reranking do not silently clamp/logistic-transform them.
- `ScoreAbsent` is the zero-value rank-only state. Numeric scores require explicit `ScorePresent` or `ScoreNormalized` and nonempty `ScoreSemantics`.
- `ScoreNormalized` requires [0,1] and an explicitly selected policy. Rank normalization declares its scale; RRF declares relative-max rank fusion with its configured k.
- `RetrieveOptions.Threshold` names value, state and semantics. Nil disables thresholding. Zero/negative native thresholds are meaningful. Rank-only/incompatible scales reject thresholding.
- `ScoreMerger`, score sorting and default grouping reject incompatible scales. Use explicitly chosen RRF or a common normalization policy for heterogeneous sources.
- `ScoreHistory` preserves prior numeric observations and contributor IDs through fusion, grouping and reranking. History slices are defensively copied by ResultSet and rendering; domain metadata ownership remains a separate host contract.
- Shadow checks compare document IDs and rank order across different score configurations, not raw score equality.

## partialSuccessRS (Fallback/Rescue gating)

`partialSuccessRS` in `retrieval/orchestrator.go` returns true when:

- `PartialFailureError` carries a non-empty `ResultSet`, or
- any error is paired with a non-empty `ResultSet` (e.g. adapter `PreserveResultOnError`).

When true, Fallback and Rescue **skip secondary** per spec: secondary runs only on primary error **and** empty `ResultSet`.

## ExecutionPipelineBuilder.WithResolver limits

- Built-in execution node types (`BackendNode`, `FallbackNode`, `RescueNode`, `AggregateNode`, `ConditionalNode`, `RequestExecutionRetrieverNode`, `RouteSwitchNode`) receive resolver injection at `Build()`.
- Custom node wrappers are not traversed; set `Resolver` on inner nodes explicitly.
- Evidence: `TestPipelineBuilderDoesNotInjectResolverIntoCustomNode`.

## Allowed map[string]any zones (task9 §3 DoD 1)

- filter.RawAttributes on storage boundary
- Elasticsearch HTTP wire / query DSL inside adapters/elasticsearch
- fake ES test fixtures (elasticsearch_test.go fakeClient Hit.Source)

Domain TMeta must use typed structs + MetadataCodec; map[string]any rejected at codec/graph boundaries.

## Dual ResultSet-on-error contract

| Layer                                      | On validation / hard error                            | On partial failure                        |
| ------------------------------------------ | ----------------------------------------------------- | ----------------------------------------- |
| Backend Retrieve                           | non-nil empty ResultSet + err (RequireErrorResultSet) | N/A                                       |
| Pipeline nodes (Fallback/Rescue/Aggregate) | preserveResultOnError when applicable                 | PartialFailureError + non-empty ResultSet |
| Rescue empty secondary                     | primary error propagated (wrapped)                    | N/A                                       |
| PostProcessorChain                         | preserveResultOnError                                 | partial preserve on ErrProtocol           |

Contract helpers: contracttest/structmeta.go RequireErrorResultSet; retrieval/orchestrator.go preserveResultOnError.
