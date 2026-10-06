# ragy

`ragy` is a capability-first retrieval toolkit with a typed retrieval API.

The core is domain-first and capability-specific:

- `retrieval` for `Document[TMeta]`, `Backend[TIntent, TMeta]`, `Query[TIntent]`, `ExecutionPipeline`, `RetrieveOptions`, planners, and post-processors
- `filter` for schema-bound filter builders and adapter-readable IR
- `embedding` for declared model/revision/config/metric spaces and query/document purpose; `dense`, `lexical`, `tensor`, `graph`, `documents` for capability contracts
- `retrieval` for query-aware reranking and ranked-list merging
- [`chunking`](chunking/README.md) for source-mapped fragments and explicit index text; [`graphingest`](graphingest/README.md) for typed extraction/resolution/materialization with host lifecycle handoff

Provider and storage adapters live under `adapters/...`. Host applications own IAM, prompts, agent/tool loops, model selection, tokenizers, prices and retention policy. The [capability matrix](docs/task19/capabilities.md) separates implemented contracts, local/wire verification and real-service verification.

## Typed retrieval model

The root module exposes shared primitives (`Page`, score clamping, canonical errors). Document payloads live in `retrieval.Document[TMeta]` — there is no untyped document type in the public API. Host applications define `TMeta` as a struct with JSON tags and pick a typed backend:

- `adapters/pgvector`, `adapters/qdrant` — dense vector search via `Retrieve` → `retrieval.ResultSet[TMeta]` (require `RetrieveOptions.Vector`)
- `adapters/elasticsearch` — lexical `multi_match` over `SearchFields` (text query, no vector); always tokenizes queries; optional `Config.Synonyms` for expansion. Include `"content"` in `SearchFields` when you need body-text recall — metadata-only configs do not search `Document.Content`.
- Hybrid dense + lexical fusion is **not** adapter-level: combine backends with `retrieval.AggregateNode` and RRF (see Hybrid fusion below)
- `adapters/neo4j` — graph traversal projected into `retrieval.ResultSet[TMeta]`; pass seeds and depth through `RetrieveOptions.Graph`

Adapters decode stored payloads into `TMeta` at the storage boundary. Filter schemas use `filter.RawAttributes` internally; callers build domain filters only through `filter.Builder`.

### Wire exceptions (`map[string]any`)

Доменная meta — только `TMeta`. Исключения на границе wire/хранения:

| Место                  | Назначение                                                                 |
| ---------------------- | -------------------------------------------------------------------------- |
| `filter.RawAttributes` | encode/decode metadata в адаптерах                                         |
| Elasticsearch HTTP DSL | query/response bodies в `adapters/elasticsearch`                           |
| ES `Hit.Source`        | wire map до decode; undeclared keys вне schema **пропускаются** без ошибки |
| ES / test fakes        | `map[string]any` в тестовых hit payloads                                   |

Всё остальное — struct-based `TMeta` + `filter.Builder`.

## Quick start

```go
package main

import (
	"context"

	"github.com/skosovsky/ragy/dense"
 "github.com/skosovsky/ragy/embedding"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

type DocMeta struct {
	Tenant string `json:"tenant"`
}

func search(
	ctx context.Context,
	embedder dense.Embedder,
	backend retrieval.Backend[struct{}, DocMeta],
	schema filter.Schema,
) (retrieval.RetrievalResult[DocMeta, retrieval.NoExecutionMeta], error) {
	empty := func(err error) (retrieval.RetrievalResult[DocMeta, retrieval.NoExecutionMeta], error) {
		return retrieval.RetrievalResult[DocMeta, retrieval.NoExecutionMeta]{
			ResultSet: retrieval.NewResultSet[DocMeta](nil, retrieval.DocumentIDResolver[DocMeta]{}),
		}, err
	}

	tenant, err := schema.StringField("tenant")
	if err != nil {
		return empty(err)
	}

	builder, err := filter.NewBuilder(schema)
	if err != nil {
		return empty(err)
	}
	cond, err := filter.Eq(builder, tenant, "acme").Build()
	if err != nil {
		return empty(err)
	}

	encoded, err := embedder.Embed(ctx, dense.Request{Inputs: []string{"reset password"}, Purpose: embedding.Query})
	if err != nil {
		return empty(err)
	}

	pipeline, err := retrieval.NewExecutionPipelineBuilder[struct{}, DocMeta, retrieval.NoExecutionMeta]().
		WithRoot(retrieval.BackendNode[struct{}, DocMeta, retrieval.NoExecutionMeta]{Backend: backend}).
		WithPostProcessors(
			retrieval.GroupBy(func(m DocMeta) string { return m.Tenant }, retrieval.DefaultMergeStrategy[DocMeta]()),
		).
		Build()
	if err != nil {
		return empty(err)
	}

	return pipeline.Execute(ctx, retrieval.Query[struct{}]{
	Read: retrieval.UnrestrictedRead(),
		Text: "reset password",
		Options: retrieval.RetrieveOptions{
			TopK:    10,
			Vector: encoded.Embeddings[0].Vector,
 Space: encoded.Embeddings[0].Space,
			Filters: cond,
		},
	})
}
```

### Composition contracts

Backend decorators preserve `ReadCapabilityProvider`, `PublicationAdmission` and request-aware `RequestReadAdmission`. Unsupported guarantees remain unsupported; tracing and caching do not grant scope or publication support. `ProjectedBackend` never invokes its payload `Project` during preflight. For a request-aware backend with a different request type, supply a pure `AdmissionProject`; it must preserve binding, perform no payload/model I/O and not mutate inputs. Same-type admission forwards the original request.

Request intent/metadata, plan intent and incoming execution metadata are immutable host inputs for the duration of execution, including parallel branches. Passing a struct by value does not clone its maps, slices or pointers. Custom nodes needing mutable state must create branch-owned values through host clone policies. Cache and artifact snapshot ports have their separate explicit ownership contracts.

RRF counts each merge key once per input list at its best original position, retaining duplicate supports and score observations. Other lists contribute independent votes; conflicting payloads remain errors. Managed graph host basis IDs cannot be re-registered after release during that adapter's lifetime. Released IDs retain no payload and old readers fail unavailable; this in-process identity guarantee does not provide persistence across restart.

Cross-type scoped/pinned projection requires a pure `AdmissionProject` in every wrapper order. Cache storage applies to current complete reads; pinned and partial reads pass through to the leaf to validate retention.

### Filters

Build domain filters with the typed builder only (`filter.NewBuilder` → `filter.Eq` / `In` / `NotEq` → `Build()`). An empty filter is `builder.Build()` with no predicates; zero-value `filter.Condition` in option structs means no filter. Low-level filter DSL nodes are internal to the `filter` package; adapters translate `filter.Condition.IR()` to native queries:

```go
builder, _ := filter.NewBuilder(schema)
cond, err := filter.Eq(filter.In(builder, category, "docs", "articles"), tenant, "acme").Build()
```

### Post-processing

Standard processors run inside `ExecutionPipeline.Execute` via `WithPostProcessors`:

- `retrieval.GroupBy` with a custom or `DefaultMergeStrategy`
- `retrieval.TopPerGroup`
- `retrieval.Rerank`

Use `NewPostProcessorChain` / `Process` only when post-processing an existing `ResultSet` outside a pipeline.

### Request planning and context artifacts

`ExecutionPipeline.Execute` accepts a typed request envelope and returns `RetrievalResult[TMeta, TExecMeta]`. Use `context.Context` for cancellation, deadlines, and tracing; keep retrieval state in `retrieval.Query[TIntent]` or `retrieval.Request[TIntent, TRequestMeta]`:

```go
type Intent struct {
	AllowExternal bool
}

var intentBackend retrieval.Backend[Intent, DocMeta]

planner := retrieval.QueryPlannerFunc[Intent, retrieval.NoRequestMeta](
	func(ctx context.Context, req retrieval.Request[Intent, retrieval.NoRequestMeta]) (retrieval.PlannedQuery[Intent], error) {
		return retrieval.PlannedQuery[Intent]{
			Text:    req.Text,
			Intent:  req.Intent,
			Filters: req.Options.Filters,
			Diagnostics: []retrieval.PlannerDiagnostic{{
				Key:   "planner",
				Value: "default",
			}},
		}, nil
	},
)

pipeline, err := retrieval.NewExecutionPipelineBuilder[Intent, DocMeta, retrieval.NoExecutionMeta]().
	WithPlanner(planner).
	WithRoot(retrieval.BackendNode[Intent, DocMeta, retrieval.NoExecutionMeta]{Backend: intentBackend}).
	Build()
if err != nil {
	// handle error
}

read := retrieval.UnrestrictedRead()
result, err := pipeline.Execute(ctx, retrieval.Query[Intent]{
	Read: read,
	Text:   "reset password",
	Intent: Intent{AllowExternal: false},
	Options: retrieval.RetrieveOptions{
		TopK:   10,
		Vector: vector,
 Space: space, // host-declared embedding.Space matching the encoder and index
	},
})
if err != nil {
	// handle error
}

artifact, err := retrieval.DefaultArtifactRenderer[DocMeta]{}.Render(ctx, read, result.ResultSet, retrieval.ArtifactRenderOptions[DocMeta]{
	Resource: retrieval.RuneResource(4000),
	CloneMeta: cloneDocMeta, // host-provided deep copy of DocMeta
})
_ = artifact
```

Use `retrieval.NewRequestExecutionPipelineBuilder[TIntent, TRequestMeta, TMeta, TExecMeta]` when request metadata must reach the planner, route predicates, nodes, backend, and execution metadata. If a request already has `Plan`, the pipeline reuses it and skips the configured planner.

`PlannedQuery` carries normalized/expanded text, universal range constraints, typed filters, diagnostics, and a cache key. `RetrievalContextArtifact` carries ordered snippets, provenance, score/rank state, complete serialized resource accounting, diagnostics, rendered text, dedup/source-formatting policy, and an untrusted-data boundary for downstream renderers.

Artifact rendering measures the complete final UTF-8 output: boundary, labels, separators and whole projected snippets. `RuneResource` explicitly selects Unicode code points and finite defaults of 1024 candidates, 1024 measurements and 4 MiB output; tokenization belongs to the host through `ArtifactResource.Measure`, `Unit` and `Profile`. Every accepted trial is measured as a whole, and that exact string is returned. Non-additive or non-monotonic tokenizers require no assumptions about partial snippets. `Resource.Packing` reports whether packing exhausted the candidates or stopped at a resource/measurement limit; it does not prove semantic sufficiency.

Custom `FormatSnippet` returns `FormattedSnippet{Text, ContentSpan}` and an error. The declared byte span must contain the unchanged snippet content. `ContextSnippet.RenderedSpan` addresses that content inside final `RenderedText`, while `Mapping` continues to address the snippet's own content. Derived or callback-projected content sets `DeliveryUncertain`; whole original document delivery is explicit through `FullDocument`. Each snippet carries `Contributors` with positions in the renderer input result set and per-input delivery flags; IDs alone cannot correlate BYOT identities. Dedup only merges supports and contributors when projected text is equal. Input/candidate/output bounds are checked, but host callbacks must cooperate with cancellation and own their allocations. Envelope, byte-bound, formatter and measurement failures return no artifact; `ArtifactError` carries a fixed stage message and an unwrap-able cause.

`RequestPlanBinder` runs after planner and before retrieval execution, including preplanned requests. It is the place to bind planned ranges/filters into `RetrieveOptions` or typed request metadata without splitting the pipeline into `plan -> bind -> retrieve` outside `ragy`.

When callers need executed route, branch trace, diagnostics, or typed side outputs, use execution-aware nodes inside the same pipeline:

```go
type ExecMeta struct {
	Route     string
	QueryText string
}

var localBackend retrieval.ExecutionBackend[Intent, DocMeta, ExecMeta]
var externalBackend retrieval.ExecutionBackend[Intent, DocMeta, ExecMeta]

routeSwitch, err := retrieval.NewRouteSwitchBuilder[Intent, string, struct{}, DocMeta, ExecMeta](
	retrieval.RoutePlannerFunc[Intent, string, struct{}](
		func(ctx context.Context, req retrieval.Query[Intent]) (retrieval.RouteDecision[string, struct{}], error) {
			if req.Intent.AllowExternal {
				return retrieval.RouteDecision[string, struct{}]{Route: "external"}, nil
			}
			return retrieval.RouteDecision[string, struct{}]{Route: "local"}, nil
		},
	),
).
	RecordDecision(func(exec ExecMeta, decision retrieval.RouteDecision[string, struct{}]) ExecMeta {
		exec.Route = decision.Route
		return exec
	}).
	Case("local", retrieval.RequestExecutionRetrieverNode[Intent, retrieval.NoRequestMeta, DocMeta, ExecMeta]{
		Backend: localBackend,
	}).
	Case("external", retrieval.RequestExecutionRetrieverNode[Intent, retrieval.NoRequestMeta, DocMeta, ExecMeta]{
		Backend: externalBackend,
	}).
	FallbackOnEmpty("local", "external").
	Build()
if err != nil {
	// handle error
}

pipeline, err := retrieval.NewExecutionPipelineBuilder[Intent, DocMeta, ExecMeta]().
	WithExecutionSeed(func(req retrieval.Query[Intent]) ExecMeta {
		return ExecMeta{QueryText: req.Text}
	}).
	WithPlanner(planner).
	WithPlanBinder(retrieval.RequestPlanBinderFunc[Intent, retrieval.NoRequestMeta, ExecMeta](
		func(
			ctx context.Context,
			req retrieval.Query[Intent],
			plan *retrieval.PlannedQuery[Intent],
			exec ExecMeta,
		) (retrieval.BoundRequest[Intent, retrieval.NoRequestMeta, ExecMeta], error) {
			if plan != nil {
				req.Options.Filters = plan.Filters
			}
			return retrieval.BoundRequest[Intent, retrieval.NoRequestMeta, ExecMeta]{
				Request:  req,
				Executed: exec,
			}, nil
		},
	)).
	WithRoot(routeSwitch).
	Build()
if err != nil {
	// handle error
}

read := retrieval.UnrestrictedRead()
result, err := pipeline.Execute(ctx, retrieval.Query[Intent]{
	Read: read,
	Text:   "reset password",
	Intent: Intent{AllowExternal: false},
	Options: retrieval.RetrieveOptions{
		TopK:   10,
		Vector: vector,
 Space: space, // host-declared embedding.Space matching the encoder and index
	},
})
if err != nil {
	// handle error
}

docs := result.ResultSet.Documents()
route := result.Executed.Route
trace := result.BranchTrace
_, _, _ = docs, route, trace
```

`WithExecutionSeed` derives the initial `TExecMeta` from the incoming request before planner, binder, route switch, and retrieval nodes run. `RequestExecutionRetrieverNode` passes that metadata into execution-aware backends. Backends that add side outputs should return the updated `Executed`; the returned `Executed` is authoritative, including zero. To retain incoming metadata, return it explicitly.

### Graph retrieval (Neo4j)

Graph backends implement `retrieval.Backend[struct{}, TMeta]` and accept traversal parameters via `RetrieveOptions.Graph`:

```go
rs, err := store.Retrieve(ctx, retrieval.Query[struct{}]{
	Read: retrieval.UnrestrictedRead(),
	Options: retrieval.RetrieveOptions{
		Graph: &retrieval.GraphOptions{
			Seeds:     []string{"project:42"},
			Direction: graph.DirectionOutbound,
			Depth:     2,
		},
	},
})
docs := rs.Documents()
```

The same store also satisfies `graph.Store[TMeta]` for upsert and low-level traversal when needed.

### Other capabilities

- `dense.Index[TMeta]` and `tensor.Index[TMeta]` for vector/tensor writes
- `graph.Store[TMeta]` for traversal and upsert
- `documents.RawStore[TMeta]` for explicit raw lookup and destructive operations; `documents.Hydrator` for scoped exact-revision hydration
- `retrieval.QueryReranker` and `retrieval.ResultMerger` for post-retrieval ranking
- `adapters/cohere/rerank` — Cohere rerank: empty query is validation (empty RS); runtime errors preserve input docs

## Resilience & execution control

`ragy` does **not** run hidden retries, circuit breakers, or backoff inside core or adapters. Policies belong in **your** code: use `context.Context` for deadlines and cancellation, and wrap capability interfaces (`dense.Embedder`, `retrieval.Backend`, `graph.Store`, …) with small **decorators** when you need retries or fallbacks. You may plug in a third-party retry/backoff or executor library around those interfaces if you want; the core stays free of such dependencies.

### Timeouts

Use `context.WithTimeout` (or `context.WithDeadline`) at the scope you care about: one deadline for an entire RAG pipeline, or tighter deadlines per `Embed` / `Retrieve` call. Context cancellation/deadline errors remain discoverable with `errors.Is`. Host callbacks and transports must cooperate with context; local cancellation does not prove remote cancellation or billing rollback.

### Canonical errors (`errors.Is`)

| Sentinel                        | Typical meaning                                                                          | Retry?                                          |
| ------------------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `ragy.ErrInvalidArgument`       | Bad config, bad request, HTTP **4xx** (except 429)                                       | No                                              |
| `ragy.ErrUnavailable`           | Network/transport failure, timeouts, HTTP **429** / **5xx**, DB/RPC failures from stores | Often yes (with backoff)                        |
| `ragy.ErrProtocol`              | Decode/validate/wire-shape failures (`WrapProjectionError`), provider cardinality bugs   | Usually no (bug or provider change)             |
| `ragy.ErrUnsupported`           | Requested capability, publication or provider mode is not implemented                   | No automatic downgrade                         |
| `context.Canceled` / `context.DeadlineExceeded` | Attempt canceled or its explicit deadline expired                       | Host chooses a new attempt; do not hide failure |
| `retrieval.PartialFailureError` | Aggregate child failed while other branches returned docs                                | Partial `ResultSet` is usable; inspect `Errors` |

`context.Canceled` is returned as-is from HTTP transport helpers (caller canceled; not a retry target).

Helpers in the root module:

- `ragy.WrapTransportError` — errors from `http.Client.Do`
- `ragy.ErrorFromHTTPResponse` — map HTTP status + body snippet to the table above
- `ragy.WrapBackendError` — classify errors from `pgvector`, `qdrant`, `elasticsearch`, and `neo4j` store boundaries (`Retrieve`, `Traverse`, and `Upsert`)
- `ragy.WrapProjectionError` — adapter document projection failures (decode, validate, wire shape) as `ErrProtocol`

Provider HTTP clients (OpenAI, Jina, Gemini, Cohere) require finite request/response/time limits, forbid redirects, preserve reported usage as known or unknown, and sanitize provider error payloads. Store adapters (`pgvector`, `qdrant`, `elasticsearch`) classify host bridge errors with the shared sentinels. Any retry wrapper must account for actual calls and unknown outcomes; a required shared recipe ledger does not implicitly refund failed calls.

### Decorator sketch (stdlib only)

Wrap `dense.Embedder` in a struct that implements `Embed` and forwards to the inner embedder after a bounded loop. Retry only when `errors.Is(err, ragy.ErrUnavailable)`; respect `ctx.Done()` between attempts (`time.After` + `select`). A full pattern is in [`examples/resilience/retry_embedder`](examples/resilience/retry_embedder).

### Neo4j and custom runners

[`adapters/neo4j`](adapters/neo4j) implements typed `Retrieve` (graph projection), `Traverse`, and `Upsert`; Cypher execution is delegated to your `Runner`. Transport and RPC failures from `Retrieve`, `Traverse`, and `Upsert` are wrapped with `ragy.WrapBackendError` — classify and retry in your runner layer if needed.

Optional `observation` sessions attach to a context with an explicit finite event capacity.
Events carry fixed stage/outcome/error enums, attempt-local numeric correlation,
counts and observed usage; query, content, metadata, IDs and raw errors are excluded.
Callbacks are serialized, synchronous and cooperative. Diagnostic exporter errors or
panics are counted without retrying or changing retrieval. Disabled sessions invoke
no observer. This diagnostic policy is separate from required evidence recording;
local cancellation does not attest remote cancellation or billing. The independent
[observation consumer](examples/conformance/observation_contract) exercises actual
BM25, typed composition and cache boundaries with `GOWORK=off`.

[`adapters/observability/otel`](adapters/observability/otel) wraps capabilities for tracing; it forwards errors from the inner implementation and does not remap `ragy.Err*`. **Retrieval minimum:** `WrapBackend`, `WrapRequestBackend`, `WrapExecutionPipeline`, `WrapRequestExecutionPipeline`. Other `Wrap*` helpers cover dense/tensor/graph/documents/rerank paths.

### Examples

See [`examples/resilience/`](examples/resilience/) for runnable `retry_embedder` and `rescue_search` patterns. `make test-examples` builds both modules and runs `go test -race` in `examples/resilience` (including rescue semantics).

See [`examples/planner/partial_failure_aggregate`](examples/planner/partial_failure_aggregate) for aggregate `PartialFailureError` handling with `errors.As`. Pass `Options.TopK` (or `FetchLimit`) on every `pipeline.Execute` call.


## Retrieval orchestrator

The retrieval orchestrator builds typed execution graphs from nodes and `retrieval.Query[TIntent]`:

- `BackendNode` wraps any regular `retrieval.Backend`; `RequestExecutionRetrieverNode` wraps execution-aware backends
- `RouteSwitchNode` dispatches one typed route decision into explicit cases, records branch trace, and supports route-aware fallback/rescue edges
- `FallbackNode` — runs secondary only when primary succeeds (`err == nil`) **and** returns an empty `ResultSet`. Use for sparse recall (e.g. catalog miss → web), not for vector outage.
- `RescueNode` — runs secondary when primary returns an error **and** an empty `ResultSet`. On partial success (error + non-empty docs), primary documents are preserved and secondary is skipped — same preserve rule applies to `FallbackNode`.

| Primary outcome        | `FallbackNode`         | `RescueNode`           |
| ---------------------- | ---------------------- | ---------------------- |
| success + empty        | → secondary            | return empty           |
| error + empty          | propagate error        | → secondary            |
| error + docs (partial) | preserve, no secondary | preserve, no secondary |

When rescue succeeds with a **non-empty** secondary, the pipeline returns **`nil` error** even if the primary failed. When secondary returns an empty `ResultSet`, the **primary error is propagated** (wrapped `ErrUnavailable`).

Compose for catalog → vector → web during outage:

```text
Rescue(
  Fallback(Aggregate(catalog, vector), Conditional(AllowWeb, web)),
  Conditional(AllowWeb, web),
)
```

- Inner Fallback: sparse empty recall → web (only if `AllowWeb`).
- Outer Rescue: aggregate/vector hard failure → web (only if `AllowWeb`).
- Apply the same intent gate on **both** web paths; an unguarded Rescue secondary bypasses `AllowWeb`.

- `AggregateNode` merges parallel child nodes with RRF by default (`ReciprocalRankFusion`, `k=60`); set `Merger` to `NewScoreMerger` for homogeneous score scales. Fusion never automatically score-merges on failure. `FusionFailureError.Observations()` retains admitted inputs for ordinary errors. Configure `DegradingMerger{Primary: merger, Fallback: fallback}` explicitly to select degradation; both causes remain observable. Protection/cancellation suppress observations and never dispatch degradation. A successful empty fusion result remains empty. Child errors surface as `PartialFailureError` when admitted observations exist; failures without observations retain ordinary joined causes. The separate returned set is authoritative.
- Post-processors in `ExecutionPipeline` run even when the root returns `PartialFailureError`; on post-processor error the pipeline preserves the separately returned admitted `ResultSet`, including empty. Error-carried payloads never replace it.
- `ConditionalNode` gates execution on query intent (for example `len(opts.Vector) > 0` or `intent.AllowWeb`). A nil `Predicate` is invalid configuration. Use an explicit true or false predicate; false skips the child.

Build a pipeline once with `retrieval.NewExecutionPipelineBuilder`, optionally chain `WithResolver` for custom `MergeKey`, then execute with `pipeline.Execute(ctx, query)` (include `query.Options.TopK` or `FetchLimit`).
See `examples/planner/catalog_vector_fallback`, `examples/planner/partial_failure_aggregate`, `examples/planner/rescue_fallback_aggregate`, `examples/planner/vector_bm25_aggregate`, and `examples/resilience/rescue_search` for planner topologies.

### Hybrid fusion (RRF)

`AggregateNode` uses Reciprocal Rank Fusion by default when merging heterogeneous sources (vector cosine, BM25, web scores). Default RRF constant is `k=60` (`defaultAggregateRRFK` in `retrieval/orchestrator.go`). Override with an explicit merger:

```go
aggregate := retrieval.AggregateNode[Intent, Meta, retrieval.NoExecutionMeta]{
    Nodes:  []retrieval.ExecutionNode[Intent, Meta, retrieval.NoExecutionMeta]{vectorNode, lexicalNode},
    Merger: retrieval.NewScoreMerger[Meta](resolver), // homogeneous scores only
}
```

> Do not use `ScoreMerger` to fuse Elasticsearch or Qdrant with BM25/vector: their native score semantics differ across sources. Use RRF (default) for heterogeneous sources.

Retrieval requests require an explicit `Read` binding. Use
`retrieval.UnrestrictedRead()` for live unrestricted reads. Protected requests use
`access.Scoped` with a mandatory predicate, host policy snapshot/authority and
publication selection. Planner, binder and projector cannot replace this binding.
A capability/authorization failure is fatal to rescue. Local persistent dense/tensor
and managed lexical/graph paths implement pinned lifecycle reads and scope admission.
Managed graph admits nodes and edges before traversal. Capability-preserving cache,
projection and OTel decorators retain these checks; incompatible external adapters
reject scoped or pinned guarantees they cannot provide. See the
[capability matrix](docs/task19/capabilities.md) for the actual per-path limits.

Lifecycle uses the explicit `ragy.lifecycle/v2` format. Local maintenance retains
identity and cleanup fences, protects registered metadata pins and unfinished work,
and refuses capacity without implicit deletion. Value-only publication capture
does not register retention. Target/source availability and authority remain host
contracts; see the [current lifecycle guide](lifecycle/README.md).

Bounded text/graph recipes share an explicit attempt ledger. Context packing counts
the complete formatted artifact, records actual input contributors and reports
delivered provenance uncertainty. Evidence export uses the strict
`ragy.retrieval-evidence/v2` contract and is default-deny; optional diagnostic
observation does not grant evidence export permission.

Numeric documents require explicit `ScoreState` and `ScoreSemantics`. Native scores
are finite and may lie outside [0,1]; only explicitly normalized scores use that
range. `ScoreAbsent` is the zero-value rank-only state. `RetrieveOptions.Threshold`
replaces `MinSimilarity` and declares its numeric value, state and semantics;
negative and zero thresholds are meaningful. Incompatible comparisons return an
error. Prior numeric observations remain in `ScoreHistory` through grouping,
fusion and reranking. Caller-comparator `Rerank` produces explicit rank-only
ordering while preserving those observations.

`PostProcessorChain` applies threshold before its processors and TopK after them.
A pipeline applies terminal threshold and TopK after root and optional chain.
A processor changing the score scale must use an explicit normalization policy;
no native or reranked scale is inferred. Host code can compose separate chains
with different options when input and output thresholds use different scales.
TopK limits delivered count, not backend work or callback memory.

Result-only composition delegates to the execution engine using `NoExecutionMeta`.
`PartialFailureError.Result` is a synchronized diagnostic view; the separately
returned result set is authoritative, even when empty. Joined and wrapped causes
remain inspectable. Protection and cancellation suppress all diagnostic payloads.
`ResolverFor` uses a custom set's optional `ResolverProvider` capability; otherwise
it selects `DocumentIDResolver`. A supplied resolver and nil/typed-nil resolver
(default DocumentID) are explicit alternatives. Required nil/typed-nil ports and
nil chain processors return `ErrInvalidArgument` before invocation.

MergeKey can identify a business group. Winner merge combines supports and score
history only for equal content and metadata; another payload's citations do not
become citations for the winner. RRF still requires equal payloads for a shared
key. Use `GroupBy` to explicitly assemble different content.

`MemoryCache` capacity bounds entry count, not bytes or peak allocation. Its
O(capacity) eviction, metadata cloning outside the lock and owned ragy slices
are retained to preserve ownership. Callback and BYOT values remain stable and
concurrency-safe host inputs. This cleanup makes no cache/copy speedup claim.


### BM25 lexical search

Use `lexical` backends or adapters with `retrieval.BackendNode` inside an execution pipeline. Whitespace-only queries return `ragy.ErrEmptyText`; queries whose tokens are empty after tokenization (for example all stopwords) return an empty `ResultSet` without error.

### Custom MergeKey

```go
type tenantResolver struct{}

func (tenantResolver) Resolve(doc retrieval.Document[Meta]) retrieval.Identity {
    return retrieval.Identity{MergeKey: doc.Meta.Tenant, DocumentID: doc.ID}
}

pipeline, _ := retrieval.NewExecutionPipelineBuilder[Intent, Meta, retrieval.NoExecutionMeta]().
    WithRoot(node).
    WithResolver(tenantResolver{}).
    Build()
```

Empty `MergeKey` is rejected at merge time with `ragy.ErrInvalidArgument` (returns error, does not panic). `Merge` and `Dedup` keep the highest score per key; when scores tie, the first seen document wins.

### Equal-score tie-break policy

Full integration semantics (score semantics, map allowlist, dual RS-on-error): see [INTEGRATION.md](.cursor/docs/INTEGRATION.md).

| Layer                                     | Equal-score tie policy                              |
| ----------------------------------------- | --------------------------------------------------- |
| `ResultSet.Merge` (same MergeKey)         | first seen wins                                     |
| Merge / Dedup / RRF output sort           | sorted MergeKey materialization → stable score sort |
| `applyTopK`                               | input order (stable)                                |
| BM25 rank                                 | sorted doc ID materialization → stable score sort   |
| Cohere rerank                             | API index order (stable)                            |
| `GroupBy` / `TopPerGroup` group iteration | sorted group keys (ascending)                       |

Post-process helpers (`GroupBy`, `TopPerGroup`, `Rerank`) validate input documents and return `ragy.ErrProtocol` (partial preserve) on projection failure. They return `ragy.ErrInvalidArgument` and preserve the input `ResultSet` when required callbacks are nil or invalid — programmer errors surfaced as validation, not panics.

### Public adapter conformance

Custom adapters can import `github.com/skosovsky/ragy/contracttest`.
`RunScopedReadSuite` accepts generic host intent/request/source metadata and a fresh
`ScopedReadFixture` per scenario. Instrument actual payload I/O and materialization
before filtering; final allowed IDs alone do not establish scope enforcement.
The suite checks direct adapter methods as well as capability admission, freshness,
cancellation and deadline propagation. Older internal helper imports were removed.

The [external consumer example](examples/conformance/README.md) runs outside ragy's
module namespace with `GOWORK=off`, including negative adapters and explicit partial
composition. Passing conformance certifies the supplied fixture/profile; hosts
still own authorization policy, I/O instrumentation and target-specific integration.
