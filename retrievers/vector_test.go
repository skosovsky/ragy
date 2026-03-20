package retrievers

import (
	"context"
	"testing"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBaseVectorRetriever_Retrieve(t *testing.T) {
	ctx := context.Background()
	emb := testutil.NewMockDenseEmbedder(4)
	store := testutil.NewInMemoryVectorStore()
	// Upsert one doc with embedding in metadata
	vec, _ := emb.Embed(ctx, []string{"hello"})
	docs := []ragy.Document{{
		ID:       "1",
		Content:  "hello",
		Metadata: map[string]any{testutil.EmbeddingKey: vec[0]},
	}}
	require.NoError(t, store.Upsert(ctx, docs))
	ret := NewBaseVectorRetriever(emb, store)
	res, err := ret.Retrieve(ctx, ragy.SearchRequest{Query: "hello", Limit: 5})
	require.NoError(t, err)
	require.GreaterOrEqual(t, len(res), 1)
	assert.Equal(t, "1", res[0].ID)
}

func TestBaseVectorRetriever_EmptyQuery(t *testing.T) {
	ctx := context.Background()
	ret := NewBaseVectorRetriever(testutil.NewMockDenseEmbedder(4), testutil.NewInMemoryVectorStore())
	_, err := ret.Retrieve(ctx, ragy.SearchRequest{Query: ""})
	require.Error(t, err)
	assert.ErrorIs(t, err, ragy.ErrEmptyQuery)
}
