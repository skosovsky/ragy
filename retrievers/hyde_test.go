package retrievers

import (
	"context"
	"testing"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/testutil"
	"github.com/stretchr/testify/require"
)

func TestHyDERetriever_Retrieve(t *testing.T) {
	ctx := context.Background()
	gen := func(_ context.Context, query string) (string, error) {
		return "hypothetical answer: " + query, nil
	}
	emb := testutil.NewMockDenseEmbedder(4)
	store := testutil.NewInMemoryVectorStore()
	vec, _ := emb.Embed(ctx, []string{"hypothetical answer: q"})
	_ = store.Upsert(ctx, []ragy.Document{{ID: "1", Content: "hypothetical answer: q", Metadata: map[string]any{testutil.EmbeddingKey: vec[0]}}})
	hyde := NewHyDERetriever(gen, emb, store)
	res, err := hyde.Retrieve(ctx, ragy.SearchRequest{Query: "q", Limit: 5})
	require.NoError(t, err)
	require.NotNil(t, res.EvalData["hypothesis"])
}
