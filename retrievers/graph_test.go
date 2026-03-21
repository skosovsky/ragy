package retrievers

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/testutil"
)

func TestGraphRetriever_Retrieve(t *testing.T) {
	ctx := context.Background()
	graph := testutil.NewInMemoryGraphStore()
	_ = graph.UpsertGraph(ctx, []ragy.Node{
		{ID: "n1", Label: "Entity", Properties: map[string]any{"content": "hello"}},
	}, []ragy.Edge{})
	gr := NewGraphRetriever(graph, WithGraphDepth(1))
	res, err := gr.Retrieve(ctx, ragy.SearchRequest{
		Query:              "ignored",
		Limit:              5,
		GraphSeedEntityIDs: []string{"n1"},
	})
	require.NoError(t, err)
	require.GreaterOrEqual(t, len(res), 1)
}
