package neo4j

import (
	"context"
	"testing"

	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/internal/contracttest"
	"github.com/skosovsky/ragy/testutil"
)

type fakeRunner struct{}

func (fakeRunner) Traverse(_ context.Context, _ Query) (graph.Snapshot, error) {
	return graph.Snapshot{}, nil
}

func (fakeRunner) Upsert(_ context.Context, _ graph.Snapshot) error { return nil }

type brokenRunner struct {
	snapshot graph.Snapshot
}

func (r brokenRunner) Traverse(_ context.Context, _ Query) (graph.Snapshot, error) {
	return r.snapshot, nil
}

func (brokenRunner) Upsert(_ context.Context, _ graph.Snapshot) error { return nil }

type memoryRunner struct {
	store *testutil.GraphStore
}

func (r memoryRunner) Traverse(ctx context.Context, query Query) (graph.Snapshot, error) {
	return r.store.Traverse(ctx, graph.TraversalRequest{
		Seeds:      query.Seeds,
		Direction:  query.Direction,
		Depth:      query.Depth,
		NodeFilter: query.NodeFilter,
		EdgeFilter: query.EdgeFilter,
		Page:       query.Page,
	})
}

func (r memoryRunner) Upsert(ctx context.Context, snapshot graph.Snapshot) error {
	return r.store.Upsert(ctx, snapshot)
}

func TestUpsertRejectsInvalidLabel(t *testing.T) {
	store, err := New(fakeRunner{}, graph.EmptySchema())
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	err = store.Upsert(context.Background(), graph.Snapshot{
		Nodes: []graph.Node{{ID: "n1", Labels: []string{"bad-label"}}},
	})
	if err == nil {
		t.Fatal("Upsert() error = nil, want error")
	}
}

func TestTraverseRejectsInvalidRunnerSnapshot(t *testing.T) {
	store, err := New(brokenRunner{
		snapshot: graph.Snapshot{
			Nodes: []graph.Node{{
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
	store, err := New(brokenRunner{
		snapshot: graph.Snapshot{
			Nodes: []graph.Node{{
				ID:     "n1",
				Labels: []string{"Doc"},
			}},
			Edges: []graph.Edge{{
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
		func(t *testing.T, snapshot graph.Snapshot, schema graph.Schema) graph.Store {
			t.Helper()
			backing := &testutil.GraphStore{Snapshot: snapshot, GraphSchema: schema}
			store, err := New(memoryRunner{store: backing}, schema)
			if err != nil {
				t.Fatalf("New(): %v", err)
			}
			return store
		},
	)
}
