package testutil

import (
	"context"
	"testing"

	"github.com/skosovsky/ragy"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestInMemoryVectorStore_UpsertSearch(t *testing.T) {
	ctx := context.Background()
	store := NewInMemoryVectorStore()
	docs := []ragy.Document{
		{ID: "1", Content: "one", Metadata: map[string]any{EmbeddingKey: []float32{1, 0, 0}}},
		{ID: "2", Content: "two", Metadata: map[string]any{EmbeddingKey: []float32{0, 1, 0}}},
	}
	require.NoError(t, store.Upsert(ctx, docs))
	req := ragy.SearchRequest{DenseVector: []float32{1, 0, 0}, Limit: 5}
	out, err := store.Search(ctx, req)
	require.NoError(t, err)
	require.GreaterOrEqual(t, len(out), 1)
	assert.Equal(t, "1", out[0].ID)
}

func TestInMemoryGraphStore_UpsertSearch(t *testing.T) {
	ctx := context.Background()
	store := NewInMemoryGraphStore()
	nodes := []ragy.Node{{ID: "n1", Label: "A", Properties: nil}}
	edges := []ragy.Edge{{SourceID: "n1", TargetID: "n2", Relation: "links"}}
	require.NoError(t, store.UpsertGraph(ctx, nodes, edges))
	ns, es, err := store.SearchGraph(ctx, []string{"n1"}, 1, ragy.SearchRequest{})
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(ns), 1)
	assert.GreaterOrEqual(t, len(es), 1)
}
