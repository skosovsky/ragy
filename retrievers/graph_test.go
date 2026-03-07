package retrievers

import (
	"context"
	"testing"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/testutil"
	"github.com/stretchr/testify/require"
)

func TestGraphRetriever_Retrieve(t *testing.T) {
	ctx := context.Background()
	graph := testutil.NewInMemoryGraphStore()
	_ = graph.UpsertGraph(ctx, []ragy.Node{
		{ID: "n1", Label: "Entity", Properties: map[string]any{"content": "hello"}},
	}, []ragy.Edge{})
	extract := func(_ context.Context, _ string) ([]ragy.Node, []ragy.Edge, error) {
		return []ragy.Node{{ID: "n1", Label: "X", Properties: nil}}, nil, nil
	}
	gr := NewGraphRetrieverWithExtractor(graph, extract, WithGraphDepth(1))
	res, err := gr.Retrieve(ctx, ragy.SearchRequest{Query: "find n1", Limit: 5})
	require.NoError(t, err)
	require.GreaterOrEqual(t, len(res.Documents), 1)
}
