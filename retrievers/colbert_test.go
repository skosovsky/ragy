package retrievers

import (
	"context"
	"testing"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/testutil"
	"github.com/stretchr/testify/require"
)

func TestColBERTRetriever_Retrieve(t *testing.T) {
	ctx := context.Background()
	emb := testutil.NewMockTensorEmbedder(4, 8)
	store := testutil.NewInMemoryVectorStore()
	tensors, _ := emb.EmbedTensors(ctx, []string{"hello"})
	_ = store.Upsert(ctx, []ragy.Document{{
		ID: "1", Content: "hello",
		Metadata: map[string]any{testutil.TensorKey: tensors[0]},
	}})
	col := NewColBERTRetriever(emb, store)
	res, err := col.Retrieve(ctx, ragy.SearchRequest{Query: "hello", Limit: 5})
	require.NoError(t, err)
	require.NotNil(t, res.EvalData["interaction_scores"])
}
