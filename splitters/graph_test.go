package splitters

import (
	"context"
	"testing"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/goleak"
)

func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m)
}

func TestGraphExtractor_NilProvider_Passthrough(t *testing.T) {
	ctx := context.Background()
	inner := NewRecursiveSplitter(WithChunkSize(20))
	graph := testutil.NewInMemoryGraphStore()
	ge := NewGraphExtractor(inner, graph, nil)
	doc := ragy.Document{ID: "np", Content: "one two three four five"}
	var chunks []ragy.Document
	for c, err := range ge.Split(ctx, doc) {
		require.NoError(t, err)
		chunks = append(chunks, c)
	}
	require.GreaterOrEqual(t, len(chunks), 1)
	nodes, _, _ := graph.SearchGraph(ctx, []string{"n1"}, 10, ragy.SearchRequest{})
	assert.Empty(t, nodes, "nil provider must not upsert into graph store")
}

func TestGraphExtractor_Split(t *testing.T) {
	ctx := context.Background()
	inner := NewRecursiveSplitter(WithChunkSize(100))
	graph := testutil.NewInMemoryGraphStore()
	provider := func(_ context.Context, chunk ragy.Document) ([]ragy.Node, []ragy.Edge, error) {
		return []ragy.Node{{ID: "n1", Label: "Chunk", Properties: map[string]any{"text": chunk.Content}}}, nil, nil
	}
	ge := NewGraphExtractor(inner, graph, provider, WithConcurrency(2))
	doc := ragy.Document{ID: "g1", Content: "Hello world. How are you?"}
	var chunks []ragy.Document
	for c, err := range ge.Split(ctx, doc) {
		require.NoError(t, err)
		chunks = append(chunks, c)
	}
	require.GreaterOrEqual(t, len(chunks), 1)
	nodes, edges, _ := graph.SearchGraph(ctx, []string{"n1"}, 1, ragy.SearchRequest{})
	assert.GreaterOrEqual(t, len(nodes), 1)
	_ = edges
}

func TestGraphExtractor_YieldSafety(t *testing.T) {
	ctx := context.Background()
	inner := NewRecursiveSplitter(WithChunkSize(5))
	graph := testutil.NewInMemoryGraphStore()
	provider := func(_ context.Context, _ ragy.Document) ([]ragy.Node, []ragy.Edge, error) {
		return []ragy.Node{{ID: "n", Label: "X", Properties: nil}}, nil, nil
	}
	ge := NewGraphExtractor(inner, graph, provider, WithConcurrency(1))
	doc := ragy.Document{ID: "g2", Content: "a b c d e f g"}
	count := 0
	for c, err := range ge.Split(ctx, doc) {
		require.NoError(t, err)
		count++
		_ = c
		if count >= 1 {
			break
		}
	}
	assert.Equal(t, 1, count)
}
