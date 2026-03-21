package retrievers

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/testutil"
)

func TestRouterRetriever_Retrieve(t *testing.T) {
	ctx := context.Background()
	emb := testutil.NewMockDenseEmbedder(4)
	store := testutil.NewInMemoryVectorStore()
	vec, _ := emb.Embed(ctx, []string{"x"})
	_ = store.Upsert(
		ctx,
		[]ragy.Document{{ID: "1", Content: "x", Metadata: map[string]any{testutil.EmbeddingKey: vec[0]}}},
	)
	base := NewBaseVectorRetriever(emb, store)
	router := NewRouterRetriever(
		func(_ context.Context, _ string) (string, error) { return "vec", nil },
		map[string]ragy.Retriever{"vec": base},
	)
	res, err := router.Retrieve(ctx, ragy.SearchRequest{Query: "x", Limit: 5})
	require.NoError(t, err)
	require.NotEmpty(t, res)
}
