package cache

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

// We use ragy.EmbeddingMetadataKey for the dense vector in Document.Metadata (adapter contract).

// VectorCache implements SemanticCache using a VectorStore and DenseEmbedder.
// Query is embedded and used for similarity search; the cached response is stored in Document.Content.
type VectorCache struct {
	store    ragy.VectorStore
	embedder ragy.DenseEmbedder
}

// NewVectorCache creates a semantic cache on top of any ragy.VectorStore (e.g. pgvector).
func NewVectorCache(store ragy.VectorStore, embedder ragy.DenseEmbedder) *VectorCache {
	return &VectorCache{store: store, embedder: embedder}
}

// Get implements SemanticCache.
func (c *VectorCache) Get(ctx context.Context, query string, threshold float64) (string, bool, error) {
	vecs, err := c.embedder.Embed(ctx, []string{query})
	if err != nil {
		return "", false, fmt.Errorf("ragy/cache: embed query: %w", err)
	}
	if len(vecs) == 0 {
		return "", false, fmt.Errorf("ragy/cache: no embedding returned")
	}
	if len(vecs[0]) == 0 {
		return "", false, fmt.Errorf("ragy/cache: empty embedding vector")
	}

	req := ragy.SearchRequest{
		Query:       query,
		DenseVector: vecs[0],
		Limit:       1,
		Filter:      filter.Equal("_cache_type", "semantic"),
	}
	results, err := c.store.Search(ctx, req)
	if err != nil {
		return "", false, fmt.Errorf("ragy/cache: search: %w", err)
	}

	if len(results) == 0 || float64(results[0].Score) < threshold {
		return "", false, nil
	}
	return results[0].Content, true, nil
}

// Set implements SemanticCache.
func (c *VectorCache) Set(ctx context.Context, query string, response string) error {
	vecs, err := c.embedder.Embed(ctx, []string{query})
	if err != nil {
		return fmt.Errorf("ragy/cache: embed query: %w", err)
	}
	if len(vecs) == 0 {
		return fmt.Errorf("ragy/cache: no embedding returned")
	}
	if len(vecs[0]) == 0 {
		return fmt.Errorf("ragy/cache: empty embedding vector")
	}

	hash := sha256.Sum256([]byte(query))
	docID := hex.EncodeToString(hash[:])

	doc := ragy.Document{
		ID:      docID,
		Content: response,
		Metadata: map[string]any{
			"_cache_type":  "semantic",
			"_cache_query": query,
			ragy.EmbeddingMetadataKey: vecs[0],
		},
	}
	if err := c.store.Upsert(ctx, []ragy.Document{doc}); err != nil {
		return fmt.Errorf("ragy/cache: save to store: %w", err)
	}
	return nil
}

var _ SemanticCache = (*VectorCache)(nil)
