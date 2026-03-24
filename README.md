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
