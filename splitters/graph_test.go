package splitters

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/goleak"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/testutil"
)

func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m)
}

// errGraphStore always fails UpsertGraph for regression tests.
type errGraphStore struct{}

func (errGraphStore) SearchGraph(context.Context, []string, int, ragy.SearchRequest) ([]ragy.Node, []ragy.Edge, error) {
	return nil, nil, nil
}

func (errGraphStore) UpsertGraph(context.Context, []ragy.Node, []ragy.Edge) error {
	return errors.New("upsert failed")
}

func TestGraphExtractor_UpsertGraphError(t *testing.T) {
	ctx := context.Background()
	inner := NewRecursiveSplitter(WithChunkSize(100))
	ge := NewGraphExtractor(
		inner,
		errGraphStore{},
		func(_ context.Context, _ ragy.Document) ([]ragy.Node, []ragy.Edge, error) {
			return []ragy.Node{{ID: "n1", Label: "L"}}, nil, nil
		},
		WithConcurrency(1),
	)
	doc := ragy.Document{ID: "e1", Content: "Hello world. How are you?"}
	var sawErr bool
	for _, err := range ge.Split(ctx, doc) {
		if err != nil {
			require.ErrorContains(t, err, "upsert failed")
			sawErr = true
			break
		}
	}
	require.True(t, sawErr, "expected UpsertGraph error to surface from Split")
}

func TestGraphExtractor_MissingGraphStore_Nil(t *testing.T) {
	ctx := context.Background()
	inner := NewRecursiveSplitter(WithChunkSize(50))
	provider := func(_ context.Context, _ ragy.Document) ([]ragy.Node, []ragy.Edge, error) {
		return []ragy.Node{{ID: "x"}}, nil, nil
	}
	ge := NewGraphExtractor(inner, nil, provider)
	var saw bool
	for _, err := range ge.Split(ctx, ragy.Document{ID: "m1", Content: "a b c d e"}) {
		if err != nil {
			require.ErrorIs(t, err, ragy.ErrMissingGraphStore)
			saw = true
			break
		}
	}
	require.True(t, saw)
}

func TestGraphExtractor_MissingGraphStore_TypedNil(t *testing.T) {
	ctx := context.Background()
	inner := NewRecursiveSplitter(WithChunkSize(50))
	var gs *testutil.InMemoryGraphStore
	var g ragy.GraphStore = gs
	provider := func(_ context.Context, _ ragy.Document) ([]ragy.Node, []ragy.Edge, error) {
		return []ragy.Node{{ID: "x"}}, nil, nil
	}
	ge := NewGraphExtractor(inner, g, provider)
	var saw bool
	for _, err := range ge.Split(ctx, ragy.Document{ID: "m2", Content: "a b c d e"}) {
		if err != nil {
			require.ErrorIs(t, err, ragy.ErrMissingGraphStore)
			saw = true
			break
		}
	}
	require.True(t, saw)
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
