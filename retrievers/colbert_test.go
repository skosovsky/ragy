package retrievers

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/testutil"
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
	_, err := col.Retrieve(ctx, ragy.SearchRequest{Query: "hello", Limit: 5})
	require.NoError(t, err)
}
