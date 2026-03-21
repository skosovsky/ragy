package retrievers

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/testutil"
)

func TestEnsembleRetriever_Retrieve(t *testing.T) {
	ctx := context.Background()
	emb := testutil.NewMockDenseEmbedder(4)
	store := testutil.NewInMemoryVectorStore()
	vec1, _ := emb.Embed(ctx, []string{"one"})
	vec2, _ := emb.Embed(ctx, []string{"two"})
	_ = store.Upsert(ctx, []ragy.Document{
		{ID: "1", Content: "one", Metadata: map[string]any{testutil.EmbeddingKey: vec1[0]}},
		{ID: "2", Content: "two", Metadata: map[string]any{testutil.EmbeddingKey: vec2[0]}},
	})
	r1 := NewBaseVectorRetriever(emb, store)
	r2 := NewBaseVectorRetriever(emb, store)
	ens := NewEnsembleRetriever([]ragy.Retriever{r1, r2})
	res, err := ens.Retrieve(ctx, ragy.SearchRequest{Query: "one", Limit: 5})
	require.NoError(t, err)
	require.NotEmpty(t, res)
}
