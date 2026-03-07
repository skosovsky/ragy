package retrievers

import (
	"context"
	"testing"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/testutil"
	"github.com/stretchr/testify/require"
)

type mockTransformer struct{}

func (mockTransformer) Transform(_ context.Context, query string) ([]string, error) {
	return []string{query + " a", query + " b"}, nil
}

func TestMultiQueryRetriever_Retrieve(t *testing.T) {
	ctx := context.Background()
	transformer := mockTransformer{}
	emb := testutil.NewMockDenseEmbedder(4)
	store := testutil.NewInMemoryVectorStore()
	base := NewBaseVectorRetriever(emb, store)
	multi := NewMultiQueryRetriever(transformer, base)
	res, err := multi.Retrieve(ctx, ragy.SearchRequest{Query: "test", Limit: 10})
	require.NoError(t, err)
	require.NotNil(t, res.EvalData["sub_queries"])
	require.Len(t, res.EvalData["sub_queries"], 2)
}
