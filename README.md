# ragy

`ragy` is a capability-first retrieval toolkit with a clear-break public API.

The core is domain-first and capability-specific:

- `ragy` for `Document`, `Chunk`, paging, and canonical errors
- `filter` for schema-bound predicate builders and validated IR
- `dense`, `lexical`, `tensor`, `graph`, `documents` for capability contracts
- `ranking` for query-aware reranking and ranked-list merging
- `chunking` and `graphingest` for ingestion stages

Provider and storage adapters live under `adapters/...`.

## Quick start

```go
package main

import (
	"context"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/filter"
)

func search(ctx context.Context, embedder dense.Embedder, searcher dense.Searcher) ([]ragy.Document, error) {
	tenant, err := searcher.Schema().StringField("tenant")
	if err != nil {
		return nil, err
	}

	expr, err := filter.Normalize(filter.Equal(tenant, "acme"))
	if err != nil {
		return nil, err
	}

	vectors, err := embedder.Embed(ctx, []string{"reset password"})
	if err != nil {
		return nil, err
	}

	page, err := ragy.NewPage(10, 0)
	if err != nil {
		return nil, err
	}

	return searcher.Search(ctx, dense.Request{
		Vector: vectors[0],
		Filter: expr,
		Page:   page,
	})
}
```

The same pattern applies to other capabilities:

- `lexical.Searcher` for text-only retrieval
- `tensor.Embedder` and `tensor.Searcher` for late-interaction pipelines
- `graph.Store` for traversal and upsert
- `documents.Store` for lookup and destructive document operations
- `ranking.QueryReranker` and `ranking.Merger` for post-retrieval ranking

## Resilience & execution control

`ragy` does **not** run hidden retries, circuit breakers, or backoff inside core or adapters. Policies belong in **your** code: use `context.Context` for deadlines and cancellation, and wrap capability interfaces (`dense.Embedder`, `dense.Searcher`, `graph.Store`, …) with small **decorators** when you need retries or fallbacks. You may plug in a third-party retry/backoff or executor library around those interfaces if you want; the core stays free of such dependencies.

### Timeouts

Use `context.WithTimeout` (or `context.WithDeadline`) at the scope you care about: one deadline for an entire RAG pipeline, or tighter deadlines per `Embed` / `Search` call. Adapter methods respect `ctx`; when the deadline passes, you typically see `context.DeadlineExceeded` wrapped with `ErrUnavailable` (see below).

### Canonical errors (`errors.Is`)

| Sentinel | Typical meaning | Retry? |
|----------|-----------------|--------|
| `ragy.ErrInvalidArgument` | Bad config, bad request, HTTP **4xx** (except 429) | No |
| `ragy.ErrUnavailable` | Network/transport failure, timeouts, HTTP **429** / **5xx**, DB/RPC failures from stores | Often yes (with backoff) |
| `ragy.ErrProtocol` | Response shape invalid after HTTP **2xx**, cardinality/index bugs | Usually no (bug or provider change) |

`context.Canceled` is returned as-is from HTTP transport helpers (caller canceled; not a retry target).

Helpers in the root module:

- `ragy.WrapTransportError` — errors from `http.Client.Do`
- `ragy.ErrorFromHTTPResponse` — map HTTP status + body snippet to the table above
- `ragy.WrapBackendError` — classify errors from `pgvector`, `qdrant`, and `elasticsearch` store boundaries

HTTP clients for providers (OpenAI, Jina, Gemini, Cohere) and store adapters (`pgvector`, `qdrant`, `elasticsearch`) use these helpers so retry logic can key off `errors.Is(err, ragy.ErrUnavailable)` vs `ErrInvalidArgument`.

### Decorator sketch (stdlib only)

Wrap `dense.Embedder` in a struct that implements `Embed` and forwards to the inner embedder after a bounded loop. Retry only when `errors.Is(err, ragy.ErrUnavailable)`; respect `ctx.Done()` between attempts (`time.After` + `select`). A full pattern is in [`examples/resilience/retry_embedder`](examples/resilience/retry_embedder).

### Neo4j and custom runners

[`adapters/neo4j`](adapters/neo4j) delegates to your `Runner` implementation. Classify and retry errors in that layer if needed.

[`adapters/observability/otel`](adapters/observability/otel) wraps capabilities for tracing; it forwards errors from the inner implementation and does not remap `ragy.Err*`.

### Examples

See [`examples/resilience/`](examples/resilience/) for runnable `retry_embedder` and `fallback_search` patterns (`go build ./...` from that module).
