package neo4j

import (
	"context"
	"testing"

	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/internal/contracttest"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/testutil"
)

type fakeRunner struct{}

func (fakeRunner) Traverse(_ context.Context, _ Query) (graph.Snapshot[contracttest.Meta], error) {
	return graph.Snapshot[contracttest.Meta]{}, nil
}

func (fakeRunner) Upsert(_ context.Context, _ graph.Snapshot[contracttest.Meta]) error { return nil }

type brokenRunner struct {
	snapshot graph.Snapshot[contracttest.Meta]
}

func (r brokenRunner) Traverse(_ context.Context, _ Query) (graph.Snapshot[contracttest.Meta], error) {
	return r.snapshot, nil
}

func (brokenRunner) Upsert(_ context.Context, _ graph.Snapshot[contracttest.Meta]) error { return nil }

type memoryRunner struct {
	store *testutil.GraphStore
}

func (r memoryRunner) Traverse(ctx context.Context, query Query) (graph.Snapshot[contracttest.Meta], error) {
	return r.store.Traverse(ctx, graph.TraversalRequest{
		Seeds:      query.Seeds,
		Direction:  query.Direction,
		Depth:      query.Depth,
		NodeFilter: query.NodeFilter,
		EdgeFilter: query.EdgeFilter,
		Page:       query.Page,
	})
}

func (r memoryRunner) Upsert(ctx context.Context, snapshot graph.Snapshot[contracttest.Meta]) error {
	return r.store.Upsert(ctx, snapshot)
}

func TestUpsertRejectsInvalidLabel(t *testing.T) {
	store, err := New[contracttest.Meta](fakeRunner{}, graph.EmptySchema())
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	err = store.Upsert(context.Background(), graph.Snapshot[contracttest.Meta]{
		Nodes: []graph.Node[contracttest.Meta]{{ID: "n1", Labels: []string{"bad-label"}}},
	})
	if err == nil {
		t.Fatal("Upsert() error = nil, want error")
	}
}

func TestTraverseRejectsInvalidRunnerSnapshot(t *testing.T) {
	store, err := New[contracttest.Meta](brokenRunner{
		snapshot: graph.Snapshot[contracttest.Meta]{
			Nodes: []graph.Node[contracttest.Meta]{{
				ID:     "n1",
				Labels: []string{"bad-label"},
			}},
		},
	}, graph.EmptySchema())
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	_, err = store.Traverse(context.Background(), graph.TraversalRequest{
		Seeds:     []string{"n1"},
		Direction: graph.DirectionOutbound,
		Depth:     1,
	})
	if err == nil {
		t.Fatal("Traverse() error = nil, want error")
	}
}

func TestTraverseRejectsDanglingRunnerSnapshot(t *testing.T) {
	store, err := New[contracttest.Meta](brokenRunner{
		snapshot: graph.Snapshot[contracttest.Meta]{
			Nodes: []graph.Node[contracttest.Meta]{{
				ID:     "n1",
				Labels: []string{"Doc"},
			}},
			Edges: []graph.Edge[contracttest.Meta]{{
				ID:       "e1",
				SourceID: "n1",
				TargetID: "missing",
				Type:     "LINKS",
			}},
		},
	}, graph.EmptySchema())
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	_, err = store.Traverse(context.Background(), graph.TraversalRequest{
		Seeds:     []string{"n1"},
		Direction: graph.DirectionOutbound,
		Depth:     1,
	})
	if err == nil {
		t.Fatal("Traverse() error = nil, want error")
	}
}

func TestGraphStoreConformance(t *testing.T) {
	contracttest.RunGraphStoreSuite(
		t,
		func(t *testing.T, snapshot graph.Snapshot[contracttest.Meta], schema graph.Schema) graph.Store[contracttest.Meta] {
			t.Helper()
			backing := &testutil.GraphStore{Snapshot: snapshot, GraphSchema: schema}
			store, err := New[contracttest.Meta](memoryRunner{store: backing}, schema)
			if err != nil {
				t.Fatalf("New(): %v", err)
			}
			return store
		},
	)
}

func TestRetrieveProjectsTraversedNodes(t *testing.T) {
	t.Parallel()

	nodeBuilder := filter.NewSchema()
	if _, err := nodeBuilder.String("tenant"); err != nil {
		t.Fatalf("String(): %v", err)
	}
	nodeSchema, err := nodeBuilder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}
	edgeSchema, err := filter.NewSchema().Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}
	schema, err := graph.NewSchema(nodeSchema, edgeSchema)
	if err != nil {
		t.Fatalf("NewSchema(): %v", err)
	}

	backing := &testutil.GraphStore{
		Snapshot: graph.Snapshot[contracttest.Meta]{
			Nodes: []graph.Node[contracttest.Meta]{
				{ID: "n1", Labels: []string{"Doc"}, Content: "hello", Meta: contracttest.Meta{"tenant": "acme"}},
			},
		},
		GraphSchema: schema,
	}
	store, err := New[contracttest.Meta](memoryRunner{store: backing}, schema)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
		Graph: &retrieval.GraphOptions{
			Seeds:     []string{"n1"},
			Direction: graph.DirectionOutbound,
			Depth:     1,
		},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if len(out) != 1 || out[0].ID != "n1" || out[0].Content != "hello" {
		t.Fatalf("Retrieve() = %#v, want node n1", out)
	}
}
