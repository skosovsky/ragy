# ragy

`ragy` is a capability-first retrieval toolkit with a typed retrieval API.

The core is domain-first and capability-specific:

- `retrieval` for `Document[TMeta]`, `Retriever[TMeta]`, `RetrieveOptions`, and post-processors
- `filter` for schema-bound filter builders and adapter-readable IR
- `dense`, `lexical`, `tensor`, `graph`, `documents` for capability contracts
- `ranking` for query-aware reranking and ranked-list merging
- `chunking` and `graphingest` for ingestion stages

Provider and storage adapters live under `adapters/...`.

## Typed retrieval model

The root module exposes shared primitives (`Page`, score clamping, canonical errors). Document payloads live in `retrieval.Document[TMeta]` — there is no untyped document type in the public API. Host applications define `TMeta` (struct or map-like type with JSON tags) and pick a typed backend:

- `adapters/pgvector`, `adapters/qdrant`, `adapters/elasticsearch` — vector / lexical / hybrid search via `Retrieve`
- `adapters/neo4j` — graph traversal projected into `[]retrieval.Document[TMeta]`; pass seeds and depth through `RetrieveOptions.Graph`

Adapters decode stored payloads into `TMeta` at the storage boundary. Filter schemas use `filter.RawAttributes` internally; callers build domain filters only through `filter.Builder`.

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
	backend retrieval.Backend[DocMeta],
) ([]retrieval.Document[DocMeta], error) {
	tenant, err := backend.(interface{ Schema() filter.Schema }).Schema().StringField("tenant")
	if err != nil {
		return nil, err
	}

	builder, err := filter.NewBuilder(backend.(interface{ Schema() filter.Schema }).Schema())
	if err != nil {
		return nil, err
	}
	cond, err := filter.Eq(builder, tenant, "acme").Build()
	if err != nil {
		return nil, err
	}

	vectors, err := embedder.Embed(ctx, []string{"reset password"})
	if err != nil {
		return nil, err
	}

	retriever := retrieval.NewPipeline(backend,
		retrieval.GroupBy(func(m DocMeta) string { return m.Tenant }, retrieval.DefaultMergeStrategy[DocMeta]()),
	)

	return retriever.Retrieve(ctx, "reset password", retrieval.RetrieveOptions{
		TopK:    10,
		Vector:  vectors[0],
		Filters: cond,
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

Standard processors run inside one `Retrieve` call when wrapped with `retrieval.NewPipeline`:

- `retrieval.GroupBy` with a custom or `DefaultMergeStrategy`
- `retrieval.TopPerGroup`
- `retrieval.Rerank`

### Graph retrieval (Neo4j)

Graph backends implement `retrieval.Backend[TMeta]` and accept traversal parameters via `RetrieveOptions.Graph`:

```go
docs, err := store.Retrieve(ctx, "", retrieval.RetrieveOptions{
	Graph: &retrieval.GraphOptions{
		Seeds:     []string{"project:42"},
		Direction: graph.DirectionOutbound,
		Depth:     2,
	},
})
```

The same store also satisfies `graph.Store[TMeta]` for upsert and low-level traversal when needed.

### Other capabilities

- `dense.Index[TMeta]` and `tensor.Index[TMeta]` for vector/tensor writes
- `graph.Store[TMeta]` for traversal and upsert
- `documents.Store[TMeta]` for lookup and destructive document operations
- `ranking.QueryReranker` and `ranking.Merger` for post-retrieval ranking

## Resilience & execution control

`ragy` does **not** run hidden retries, circuit breakers, or backoff inside core or adapters. Policies belong in **your** code: use `context.Context` for deadlines and cancellation, and wrap capability interfaces (`dense.Embedder`, `retrieval.Retriever`, `graph.Store`, …) with small **decorators** when you need retries or fallbacks. You may plug in a third-party retry/backoff or executor library around those interfaces if you want; the core stays free of such dependencies.

### Timeouts

Use `context.WithTimeout` (or `context.WithDeadline`) at the scope you care about: one deadline for an entire RAG pipeline, or tighter deadlines per `Embed` / `Retrieve` call. Adapter methods respect `ctx`; when the deadline passes, you typically see `context.DeadlineExceeded` wrapped with `ErrUnavailable` (see below).

### Canonical errors (`errors.Is`)

| Sentinel                  | Typical meaning                                                                          | Retry?                              |
| ------------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------- |
| `ragy.ErrInvalidArgument` | Bad config, bad request, HTTP **4xx** (except 429)                                       | No                                  |
| `ragy.ErrUnavailable`     | Network/transport failure, timeouts, HTTP **429** / **5xx**, DB/RPC failures from stores | Often yes (with backoff)            |
| `ragy.ErrProtocol`        | Response shape invalid after HTTP **2xx**, cardinality/index bugs                        | Usually no (bug or provider change) |

`context.Canceled` is returned as-is from HTTP transport helpers (caller canceled; not a retry target).

Helpers in the root module:

- `ragy.WrapTransportError` — errors from `http.Client.Do`
- `ragy.ErrorFromHTTPResponse` — map HTTP status + body snippet to the table above
- `ragy.WrapBackendError` — classify errors from `pgvector`, `qdrant`, and `elasticsearch` store boundaries

HTTP clients for providers (OpenAI, Jina, Gemini, Cohere) and store adapters (`pgvector`, `qdrant`, `elasticsearch`) use these helpers so retry logic can key off `errors.Is(err, ragy.ErrUnavailable)` vs `ErrInvalidArgument`.

### Decorator sketch (stdlib only)

Wrap `dense.Embedder` in a struct that implements `Embed` and forwards to the inner embedder after a bounded loop. Retry only when `errors.Is(err, ragy.ErrUnavailable)`; respect `ctx.Done()` between attempts (`time.After` + `select`). A full pattern is in [`examples/resilience/retry_embedder`](examples/resilience/retry_embedder).

### Neo4j and custom runners

[`adapters/neo4j`](adapters/neo4j) implements typed `Retrieve` (graph projection) and delegates Cypher execution to your `Runner`. Classify and retry errors in that layer if needed.

[`adapters/observability/otel`](adapters/observability/otel) wraps capabilities for tracing; it forwards errors from the inner implementation and does not remap `ragy.Err*`.

### Examples

See [`examples/resilience/`](examples/resilience/) for runnable `retry_embedder` and `fallback_search` patterns (`go build ./...` from that module).
