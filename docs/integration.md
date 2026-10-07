# Typed retrieval integration

Start with the complete [local BM25 example](../examples/local-bm25/main.go). `Query[TIntent]` is the no-request-metadata alias of `Request[TIntent, NoRequestMeta]`; use `NewRequestExecutionPipelineBuilder` for independent typed request metadata. Put business intent/metadata in this envelope, not context values. Context carries cancellation, deadlines and transport/tracing state.

Every request selects an explicit Read binding. `UnrestrictedRead()` chooses current unrestricted retrieval, with no managed snapshot or authorization claim. Scoped/pinned bindings require capable targets; unsupported enforcement rejects before payload I/O. A filter is search selection unless the host independently establishes a mandatory authority-bound scope.

## Planning and composition

`EffectiveText()` chooses planned expanded text, then planned text, then raw request text. A prebuilt Plan is reused; a configured planner is not called again. `WithPlanner` runs before the root and `WithPlanBinder` binds the plan into request options/typed state before retrieval. Planned filters are intersected with request filters and mandatory binding constraints, not substituted for them.

| Node | Dispatch rule |
|---|---|
| BackendNode / RequestBackendNode | Admits the binding and invokes the typed retrieval backend. |
| RequestExecutionRetrieverNode | Passes execution metadata to an execution-aware backend; its returned Executed value is authoritative, including zero. |
| FallbackNode | Secondary on successful empty primary; errors and partial results do not trigger it. |
| RescueNode | Secondary on eligible primary error with empty payload; protection/cancellation and explicit partial failures are not rescued. |
| ConditionalNode | Explicit predicate selects a node. Nil node or predicate is invalid configuration. |
| AggregateNode / RequestExecutionAggregateNode | Independent branches, bounded concurrency, explicit merging and optional typed execution reducer. |
| RouteSwitchNode | Host typed route decisions and explicit cases; routing policy belongs to the host. |

`WithExecutionSeed` initializes execution metadata. Plain BackendNode retains incoming metadata because that backend has no execution output. Execution-aware backend output replaces it; the library does not infer presence from zero values. Aggregate branches receive immutable shared input; supply branch-owned mutable values and an explicit reducer when needed. Build validates configured nodes before dispatch.

The pipeline applies post-processors and terminal threshold/TopK to the returned authoritative set. Non-nil empty sets remain empty even when an error contains earlier documents. `PartialFailureError.Result` is a synchronized diagnostic view; do not use it as a second delivery channel. Read Coverage and error are independent: an error-free partial publication can still have incomplete coverage. See [errors and recovery](errors-and-recovery.md).

The executable [planner](../examples/planner/vector_bm25_aggregate/main.go) and [resilience](../examples/resilience/rescue_search/main.go) examples show typed composition. They are separate development modules, not publishable adapter modules.

## Scores, identity and fusion

`ScoreAbsent` carries rank-only evidence: zero numeric score and empty semantics do not mean a scored zero. `ScorePresent` requires a finite native score and declared semantics; `ScoreNormalized` requires an explicit transformation into [0,1]. Semantics identify the relevant model/configuration scale. Dense matching validates embedding Space and query/document purpose; cosine/BM25/tensor/provider scores are not interchangeable.

Score merging requires comparable states/scales. Combine dense and lexical rankings through an explicit aggregate and rank fusion rather than comparing native scores. RRF contributes once per identity per ranking. Ordinary fusion failure retains observations as diagnostic input and does not silently select ScoreMerger. `DegradingMerger` explicitly invokes a fallback on ordinary failure and returns the result with the error; it never degrades protection or cancellation. Choose ScoreMerger only for host-attested comparable scales.

| Ordering boundary | Determinism |
|---|---|
| Merge/dedup winner for same MergeKey | Best comparable score/rank; equal evidence keeps first seen payload. |
| Merge/dedup materialization | Sorted MergeKeys, then stable score/rank order. |
| BM25 equal native score | Sorted document IDs before stable score sort. |
| Terminal TopK | Retains input order. |
| Group iteration | Sorted group keys. |

Resolver MergeKey, StorageID and source/document identity serve different purposes. Identity must be stable and valid; empty/ambiguous keys are errors. Builder resolver affects known configurable nodes/processors; arbitrary host nodes must implement their own compatible identity contract. Dedup losing payloads with different content/metadata do not add their evidence to the winner. Source support is not invented from an ID match.

A threshold names its ScoreState and ScoreSemantics and is an inclusive minimum in that scale. Nil means absent; negative and zero native thresholds are valid. TopK and FetchLimit count documents, not bytes or tokens. Both zero is invalid; explicit positive FetchLimit must be at least positive TopK. A larger fetch universe can change recall and post-processing quality. See [limits](limits.md).

## Context artifact handoff

`DefaultArtifactRenderer` receives context, the original Read binding, the returned ResultSet and `ArtifactRenderOptions[TMeta]`. Configure `Resource`, `CloneMeta`, formatting and dedup policies explicitly. Resource measurement includes the complete formatted output and names host unit/profile. Measurement/candidate/byte caps bound packing work and delivered serialization, not arbitrary callback CPU or process memory.

Artifact mapping coordinates address snippet Content, not generated labels. RenderedSpan identifies its placement in the rendered output. Contributors identify actual input positions; FullDocument and DeliveryUncertain describe delivery, not semantic sufficiency. UntrustedDataBoundary is a presentation marker, not a model injection defense or factual judge. Keep exact mappings/supports and use [source authority](../source/README.md) for admitted retained text. [Ownership](ownership.md) describes borrowed metadata and explicit captures.
