// Package cache provides semantic caching for query-response pairs using a vector store and embedder.
// Use VectorCache with any ragy.VectorStore (e.g. pgvector) and ragy.DenseEmbedder.
// For OpenTelemetry, use github.com/skosovsky/ragy/adapters/observability/otel (optional adapter module).
package cache

import "context"

// SemanticCache defines the contract for semantic caching of query-response pairs.
// Implementations use a vector store and embedder to find cached responses for semantically similar queries.
type SemanticCache interface {
	// Get looks up a cached response for the given query. On hit (Document.Score >= threshold; scale is store-specific),
	// returns (response, true, nil). On miss, returns ("", false, nil); err is only set for real failures (e.g. store/embedder error).
	Get(ctx context.Context, query string, threshold float64) (response string, hit bool, err error)

	// Set stores a query-response pair in the cache.
	Set(ctx context.Context, query string, response string) error
}
