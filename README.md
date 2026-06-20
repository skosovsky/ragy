# ragy

`ragy` is a capability-first retrieval toolkit with a typed retrieval API.

The core is domain-first and capability-specific:

- `retrieval` for `Document[TMeta]`, `Backend[TIntent, TMeta]`, `Query[TIntent]`, `Pipeline`, `RetrieveOptions`, planners, and post-processors
- `filter` for schema-bound filter builders and adapter-readable IR
- `dense`, `lexical`, `tensor`, `graph`, `documents` for capability contracts
- `ranking` for query-aware reranking and ranked-list merging
- `chunking` and `graphingest` for ingestion stages

Provider and storage adapters live under `adapters/...`.

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
) (retrieval.ResultSet[DocMeta], error) {
	empty := func(err error) (retrieval.ResultSet[DocMeta], error) {
		return retrieval.NewResultSet[DocMeta](nil, retrieval.DocumentIDResolver[DocMeta]{}), err
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

	vectors, err := embedder.Embed(ctx, []string{"reset password"})
	if err != nil {
		return empty(err)
	}

	pipeline, err := retrieval.NewPipelineBuilder[struct{}, DocMeta]().
		WithRoot(retrieval.RetrieverNode[struct{}, DocMeta]{Backend: backend}).
		WithPostProcessors(
			retrieval.GroupBy(func(m DocMeta) string { return m.Tenant }, retrieval.DefaultMergeStrategy[DocMeta]()),
		).
		Build()
	if err != nil {
		return empty(err)
	}

	return pipeline.Retrieve(ctx, retrieval.Query[struct{}]{
		Text: "reset password",
		Options: retrieval.RetrieveOptions{
			TopK:    10,
			Vector:  vectors[0],
			Filters: cond,
		},
	})
}
```

### Filters

Build domain filters with the typed builder only (`filter.NewBuilder` → `filter.Eq` / `In` / `NotEq` → `Build()`). An empty filter is `builder.Build()` with no predicates; zero-value `filter.Condition` in option structs means no filter. Low-level filter DSL nodes are internal to the `filter` package; adapters translate `filter.Condition.IR()` to native queries:

```go
builder, _ := filter.NewBuilder(schema)
cond, err := filter.Eq(filter.In(builder, category, "docs", "articles"), tenant, "acme").Build()
```

### Post-processing

Standard processors run inside `Pipeline.Retrieve` via `WithPostProcessors`:

- `retrieval.GroupBy` with a custom or `DefaultMergeStrategy`
- `retrieval.TopPerGroup`
- `retrieval.Rerank`

Use `NewPostProcessorChain` / `Process` only when post-processing an existing `ResultSet` outside a pipeline.

### Request planning and context artifacts

`Pipeline.Retrieve` accepts a typed request envelope. Use `context.Context` for cancellation, deadlines, and tracing; keep retrieval state in `retrieval.Query[TIntent]` or `retrieval.Request[TIntent, TRequestMeta]`:

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

pipeline, err := retrieval.NewPipelineBuilder[Intent, DocMeta]().
	WithPlanner(planner).
	WithRoot(retrieval.RetrieverNode[Intent, DocMeta]{Backend: intentBackend}).
	Build()
if err != nil {
	// handle error
}

rs, err := pipeline.Retrieve(ctx, retrieval.Query[Intent]{
	Text:   "reset password",
	Intent: Intent{AllowExternal: false},
	Options: retrieval.RetrieveOptions{
		TopK:   10,
		Vector: vector,
	},
})
if err != nil {
	// handle error
}

artifact, err := retrieval.DefaultArtifactRenderer[DocMeta]{}.Render(rs, retrieval.ArtifactRenderOptions[DocMeta]{
	Budget: 4000,
})
_ = artifact
```

Use `retrieval.NewRequestPipelineBuilder[TIntent, TRequestMeta, TMeta]` when request metadata must reach the planner, route predicates, nodes, and backend. If a request already has `Plan`, the pipeline reuses it and skips the configured planner.

`PlannedQuery` carries normalized/expanded text, universal range constraints, typed filters, diagnostics, and a cache key. `RetrievalContextArtifact` carries ordered snippets, provenance, score/rank state, budget accounting, diagnostics, rendered text, dedup/source-formatting policy, and an untrusted-data boundary for downstream renderers.

### Graph retrieval (Neo4j)

Graph backends implement `retrieval.Backend[struct{}, TMeta]` and accept traversal parameters via `RetrieveOptions.Graph`:

```go
rs, err := store.Retrieve(ctx, retrieval.Query[struct{}]{
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
- `documents.Store[TMeta]` for lookup and destructive document operations
- `ranking.QueryReranker` and `ranking.Merger` for post-retrieval ranking
- `adapters/cohere/rerank` — Cohere rerank: empty query is validation (empty RS); runtime errors preserve input docs

## Resilience & execution control

`ragy` does **not** run hidden retries, circuit breakers, or backoff inside core or adapters. Policies belong in **your** code: use `context.Context` for deadlines and cancellation, and wrap capability interfaces (`dense.Embedder`, `retrieval.Backend`, `graph.Store`, …) with small **decorators** when you need retries or fallbacks. You may plug in a third-party retry/backoff or executor library around those interfaces if you want; the core stays free of such dependencies.

### Timeouts

Use `context.WithTimeout` (or `context.WithDeadline`) at the scope you care about: one deadline for an entire RAG pipeline, or tighter deadlines per `Embed` / `Retrieve` call. Adapter methods respect `ctx`; when the deadline passes, you typically see `context.DeadlineExceeded` wrapped with `ErrUnavailable` (see below).

### Canonical errors (`errors.Is`)

| Sentinel                        | Typical meaning                                                                          | Retry?                                          |
| ------------------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `ragy.ErrInvalidArgument`       | Bad config, bad request, HTTP **4xx** (except 429)                                       | No                                              |
| `ragy.ErrUnavailable`           | Network/transport failure, timeouts, HTTP **429** / **5xx**, DB/RPC failures from stores | Often yes (with backoff)                        |
| `ragy.ErrProtocol`              | Decode/validate/wire-shape failures (`WrapProjectionError`), provider cardinality bugs   | Usually no (bug or provider change)             |
| `retrieval.PartialFailureError` | Aggregate child failed while other branches returned docs                                | Partial `ResultSet` is usable; inspect `Errors` |

`context.Canceled` is returned as-is from HTTP transport helpers (caller canceled; not a retry target).

Helpers in the root module:

- `ragy.WrapTransportError` — errors from `http.Client.Do`
- `ragy.ErrorFromHTTPResponse` — map HTTP status + body snippet to the table above
- `ragy.WrapBackendError` — classify errors from `pgvector`, `qdrant`, `elasticsearch`, and `neo4j` store boundaries (`Retrieve`, `Traverse`, and `Upsert`)
- `ragy.WrapProjectionError` — adapter document projection failures (decode, validate, wire shape) as `ErrProtocol`

HTTP clients for providers (OpenAI, Jina, Gemini, Cohere) and store adapters (`pgvector`, `qdrant`, `elasticsearch`) use these helpers so retry logic can key off `errors.Is(err, ragy.ErrUnavailable)` vs `ErrInvalidArgument`.

### Decorator sketch (stdlib only)

Wrap `dense.Embedder` in a struct that implements `Embed` and forwards to the inner embedder after a bounded loop. Retry only when `errors.Is(err, ragy.ErrUnavailable)`; respect `ctx.Done()` between attempts (`time.After` + `select`). A full pattern is in [`examples/resilience/retry_embedder`](examples/resilience/retry_embedder).

### Neo4j and custom runners

[`adapters/neo4j`](adapters/neo4j) implements typed `Retrieve` (graph projection), `Traverse`, and `Upsert`; Cypher execution is delegated to your `Runner`. Transport and RPC failures from `Retrieve`, `Traverse`, and `Upsert` are wrapped with `ragy.WrapBackendError` — classify and retry in your runner layer if needed.

[`adapters/observability/otel`](adapters/observability/otel) wraps capabilities for tracing; it forwards errors from the inner implementation and does not remap `ragy.Err*`. **Retrieval minimum:** `WrapBackend`, `WrapRequestBackend`, `WrapPipeline`, `WrapRequestPipeline`. Other `Wrap*` helpers cover dense/tensor/graph/documents/rerank paths.

### Examples

See [`examples/resilience/`](examples/resilience/) for runnable `retry_embedder` and `rescue_search` patterns. `make test-examples` builds both modules and runs `go test -race` in `examples/resilience` (including rescue semantics).

See [`examples/planner/partial_failure_aggregate`](examples/planner/partial_failure_aggregate) for aggregate `PartialFailureError` handling with `errors.As`. Pass `Options.TopK` (or `FetchLimit`) on every `pipeline.Retrieve` call.


## Retrieval orchestrator

The retrieval orchestrator builds typed graphs from nodes and `retrieval.Query[TIntent]`:

- `RetrieverNode` wraps any `retrieval.Backend`
- `RouteNode` gates a child behind a typed route decision without hardcoded domain categories
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

- `AggregateNode` merges parallel child nodes with RRF by default (`ReciprocalRankFusion`, `k=60`); set `Merger` to `NewScoreMerger` for homogeneous score scales. When `Merger.Merge` fails, degraded fallback uses sequential `ResultSet.Merge` (highest score per MergeKey) — ordering may differ from RRF. Child errors surface as `PartialFailureError` when other branches succeed.
- Post-processors in `Pipeline` run even when the root returns `PartialFailureError`; on post-processor error the pipeline preserves the last non-empty `ResultSet`.
- `ConditionalNode` gates execution on query intent (for example `len(opts.Vector) > 0` or `intent.AllowWeb`). A **nil `Predicate` runs the child always** (footgun); use an explicit `func(_) bool { return false }` to disable.

Build a pipeline once with `retrieval.NewPipelineBuilder`, optionally chain `WithResolver` for custom `MergeKey`, then execute with `pipeline.Retrieve(ctx, query)` (include `query.Options.TopK` or `FetchLimit`).
See `examples/planner/catalog_vector_fallback`, `examples/planner/partial_failure_aggregate`, `examples/planner/rescue_fallback_aggregate`, `examples/planner/vector_bm25_aggregate`, and `examples/resilience/rescue_search` for planner topologies.

### Hybrid fusion (RRF)

`AggregateNode` uses Reciprocal Rank Fusion by default when merging heterogeneous sources (vector cosine, BM25, web scores). Default RRF constant is `k=60` (`defaultAggregateRRFK` in `retrieval/orchestrator.go`). Override with an explicit merger:

```go
aggregate := retrieval.AggregateNode[Intent, Meta]{
    Nodes:  []retrieval.Node[Intent, Meta]{vectorNode, lexicalNode},
    Merger: retrieval.NewScoreMerger[Meta](resolver), // homogeneous scores only
}
```

> Do not use `ScoreMerger` to fuse Elasticsearch or Qdrant with BM25/vector: their scores are logistic-normalized and not comparable across sources. Use RRF (default) for heterogeneous sources.

### BM25 lexical search

Use `lexical` backends or adapters with `retrieval.RetrieverNode` inside a pipeline. Whitespace-only queries return `ragy.ErrEmptyText`; queries whose tokens are empty after tokenization (for example all stopwords) return an empty `ResultSet` without error.

### Custom MergeKey

```go
type tenantResolver struct{}

func (tenantResolver) Resolve(doc retrieval.Document[Meta]) retrieval.Identity {
    return retrieval.Identity{MergeKey: doc.Meta.Tenant, DocumentID: doc.ID}
}

pipeline, _ := retrieval.NewPipelineBuilder[Intent, Meta]().
    WithRoot(node).
    WithResolver(tenantResolver{}).
    Build()
```

Empty `MergeKey` is rejected at merge time with `ragy.ErrInvalidArgument` (returns error, does not panic). `Merge` and `Dedup` keep the highest score per key; when scores tie, the first seen document wins.

### Equal-score tie-break policy

Full integration semantics (score normalization, map allowlist, dual RS-on-error): see [INTEGRATION.md](.cursor/docs/INTEGRATION.md).

| Layer                                     | Equal-score tie policy                              |
| ----------------------------------------- | --------------------------------------------------- |
| `ResultSet.Merge` (same MergeKey)         | first seen wins                                     |
| Merge / Dedup / RRF output sort           | sorted MergeKey materialization → stable score sort |
| `applyTopK`                               | input order (stable)                                |
| BM25 rank                                 | sorted doc ID materialization → stable score sort   |
| Cohere rerank                             | API index order (stable)                            |
| `GroupBy` / `TopPerGroup` group iteration | sorted group keys (ascending)                       |

Post-process helpers (`GroupBy`, `TopPerGroup`, `Rerank`) validate input documents and return `ragy.ErrProtocol` (partial preserve) on projection failure. They return `ragy.ErrInvalidArgument` and preserve the input `ResultSet` when required callbacks are nil or invalid — programmer errors surfaced as validation, not panics.
