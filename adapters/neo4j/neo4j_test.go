package neo4j

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/internal/contracttest"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/testutil"
)

type fakeRunner struct{}

func (fakeRunner) Traverse(_ context.Context, _ Query) (graph.Snapshot[contracttest.StructMeta], error) {
	return graph.Snapshot[contracttest.StructMeta]{}, nil
}

func (fakeRunner) Upsert(_ context.Context, _ graph.Snapshot[contracttest.StructMeta]) error {
	return nil
}

type brokenRunner struct {
	snapshot graph.Snapshot[contracttest.StructMeta]
}

func (r brokenRunner) Traverse(_ context.Context, _ Query) (graph.Snapshot[contracttest.StructMeta], error) {
	return r.snapshot, nil
}

func (brokenRunner) Upsert(_ context.Context, _ graph.Snapshot[contracttest.StructMeta]) error {
	return nil
}

type errRunner struct {
	traverseErr error
	upsertErr   error
}

func (r errRunner) Traverse(_ context.Context, _ Query) (graph.Snapshot[contracttest.StructMeta], error) {
	return graph.Snapshot[contracttest.StructMeta]{}, r.traverseErr
}

func (r errRunner) Upsert(_ context.Context, _ graph.Snapshot[contracttest.StructMeta]) error {
	return r.upsertErr
}

type memoryRunner struct {
	store *testutil.GraphStore
}

func (r memoryRunner) Traverse(ctx context.Context, query Query) (graph.Snapshot[contracttest.StructMeta], error) {
	return r.store.Traverse(ctx, graph.TraversalRequest{
		Seeds:      query.Seeds,
		Direction:  query.Direction,
		Depth:      query.Depth,
		NodeFilter: query.NodeFilter,
		EdgeFilter: query.EdgeFilter,
		Page:       query.Page,
	})
}

func (r memoryRunner) Upsert(ctx context.Context, snapshot graph.Snapshot[contracttest.StructMeta]) error {
	return r.store.Upsert(ctx, snapshot)
}

func TestUpsertRejectsInvalidLabel(t *testing.T) {
	store, err := New[contracttest.StructMeta](fakeRunner{}, graph.EmptySchema(), Config[contracttest.StructMeta]{})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	err = store.Upsert(context.Background(), graph.Snapshot[contracttest.StructMeta]{
		Nodes: []graph.Node[contracttest.StructMeta]{{ID: "n1", Labels: []string{"bad-label"}}},
	})
	if err == nil {
		t.Fatal("Upsert() error = nil, want error")
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Upsert() error = %v, want invalid argument", err)
	}
}

func TestTraverseRejectsInvalidRunnerSnapshot(t *testing.T) {
	store, err := New[contracttest.StructMeta](brokenRunner{
		snapshot: graph.Snapshot[contracttest.StructMeta]{
			Nodes: []graph.Node[contracttest.StructMeta]{{
				ID:     "n1",
				Labels: []string{"bad-label"},
			}},
		},
	}, graph.EmptySchema(), Config[contracttest.StructMeta]{})
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
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Traverse() error = %v, want invalid argument", err)
	}
}

func TestTraverseRejectsDanglingRunnerSnapshot(t *testing.T) {
	store, err := New[contracttest.StructMeta](brokenRunner{
		snapshot: graph.Snapshot[contracttest.StructMeta]{
			Nodes: []graph.Node[contracttest.StructMeta]{{
				ID:     "n1",
				Labels: []string{"Doc"},
			}},
			Edges: []graph.Edge[contracttest.StructMeta]{{
				ID:       "e1",
				SourceID: "n1",
				TargetID: "missing",
				Type:     "LINKS",
			}},
		},
	}, graph.EmptySchema(), Config[contracttest.StructMeta]{})
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
	} else if !errors.Is(err, ragy.ErrInvalidGraph) {
		t.Fatalf("Traverse() error = %v, want invalid graph", err)
	}
}

func TestGraphStoreConformance(t *testing.T) {
	contracttest.RunGraphStoreSuite(
		t,
		func(t *testing.T, snapshot graph.Snapshot[contracttest.StructMeta], schema graph.Schema) graph.Store[contracttest.StructMeta] {
			t.Helper()
			backing := &testutil.GraphStore{Snapshot: snapshot, GraphSchema: schema}
			store, err := New[contracttest.StructMeta](
				memoryRunner{store: backing},
				schema,
				Config[contracttest.StructMeta]{},
			)
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
		Snapshot: graph.Snapshot[contracttest.StructMeta]{
			Nodes: []graph.Node[contracttest.StructMeta]{
				{ID: "n1", Labels: []string{"Doc"}, Content: "hello", Meta: contracttest.StructMeta{Tenant: "acme"}},
			},
		},
		GraphSchema: schema,
	}
	store, err := New[contracttest.StructMeta](memoryRunner{store: backing}, schema, Config[contracttest.StructMeta]{})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
		TopK: 10,
		Graph: &retrieval.GraphOptions{
			Seeds:     []string{"n1"},
			Direction: graph.DirectionOutbound,
			Depth:     1,
		},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	docs := out.Documents()
	if len(docs) != 1 || docs[0].ID != "n1" || docs[0].Content != "hello" {
		t.Fatalf("Retrieve() = %#v, want node n1", docs)
	}
}

func TestRetrieveUsesFetchLimitForBackendSlice(t *testing.T) {
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
		Snapshot: graph.Snapshot[contracttest.StructMeta]{
			Nodes: []graph.Node[contracttest.StructMeta]{
				{ID: "n1", Labels: []string{"Doc"}, Content: "one", Meta: contracttest.StructMeta{Tenant: "acme"}},
				{ID: "n2", Labels: []string{"Doc"}, Content: "two", Meta: contracttest.StructMeta{Tenant: "acme"}},
				{ID: "n3", Labels: []string{"Doc"}, Content: "three", Meta: contracttest.StructMeta{Tenant: "acme"}},
			},
		},
		GraphSchema: schema,
	}
	store, err := New[contracttest.StructMeta](memoryRunner{store: backing}, schema, Config[contracttest.StructMeta]{})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
		FetchLimit: 1,
		TopK:       1,
		Graph: &retrieval.GraphOptions{
			Seeds:     []string{"n1", "n2", "n3"},
			Direction: graph.DirectionOutbound,
			Depth:     1,
		},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if out.Len() != 1 {
		t.Fatalf("out.Len() = %d, want 1", out.Len())
	}
}

func TestRetrieveOptionsInvalidConformance(t *testing.T) {
	t.Parallel()

	contracttest.RunGraphRetrieveOptionsInvalidSuite(t, func(t *testing.T) retrieval.Backend[contracttest.StructMeta] {
		t.Helper()
		store, err := New[contracttest.StructMeta](
			memoryRunner{store: &testutil.GraphStore{}},
			graph.EmptySchema(),
			Config[contracttest.StructMeta]{},
		)
		if err != nil {
			t.Fatalf("New(): %v", err)
		}
		return store
	})
}

type traverseErrorRunner struct {
	err error
}

func (r traverseErrorRunner) Traverse(context.Context, Query) (graph.Snapshot[contracttest.StructMeta], error) {
	return graph.Snapshot[contracttest.StructMeta]{}, r.err
}

func (traverseErrorRunner) Upsert(context.Context, graph.Snapshot[contracttest.StructMeta]) error {
	return nil
}

func TestRetrieveTraverseError(t *testing.T) {
	t.Parallel()

	store, err := New[contracttest.StructMeta](
		traverseErrorRunner{err: ragy.ErrUnavailable},
		graph.EmptySchema(),
		Config[contracttest.StructMeta]{},
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
		TopK: 1,
		Graph: &retrieval.GraphOptions{
			Seeds:     []string{"n1"},
			Direction: graph.DirectionOutbound,
			Depth:     1,
		},
	})
	contracttest.RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want unavailable", err)
	}
}

func TestRetrieveWrapsTraverseError(t *testing.T) {
	t.Parallel()

	raw := errors.New("connection reset")
	store, err := New[contracttest.StructMeta](
		traverseErrorRunner{err: raw},
		graph.EmptySchema(),
		Config[contracttest.StructMeta]{},
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
		TopK: 1,
		Graph: &retrieval.GraphOptions{
			Seeds:     []string{"n1"},
			Direction: graph.DirectionOutbound,
			Depth:     1,
		},
	})
	contracttest.RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want unavailable", err)
	}
	if !errors.Is(err, raw) {
		t.Fatalf("error chain lost upstream: %v", err)
	}
}

func TestRetrievePreservesPartialValidationError(t *testing.T) {
	t.Parallel()

	runner := brokenRunner{
		snapshot: graph.Snapshot[contracttest.StructMeta]{
			Nodes: []graph.Node[contracttest.StructMeta]{
				{ID: "ok", Content: "good"},
				{ID: "", Content: "bad"},
			},
		},
	}
	store, err := New[contracttest.StructMeta](runner, graph.EmptySchema(), Config[contracttest.StructMeta]{})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
		TopK: 5,
		Graph: &retrieval.GraphOptions{
			Seeds:     []string{"ok"},
			Direction: graph.DirectionOutbound,
			Depth:     1,
		},
	})
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Retrieve() error = %v, want protocol", err)
	}
	if out.Len() != 1 || out.Documents()[0].ID != "ok" {
		t.Fatalf("Documents() = %#v, want partial ok node", out.Documents())
	}
}

func TestRetrievePartialProjectionConformance(t *testing.T) {
	contracttest.RunRetrievePartialProjectionSuite(t, func(t *testing.T) retrieval.Backend[contracttest.StructMeta] {
		t.Helper()

		runner := brokenRunner{
			snapshot: graph.Snapshot[contracttest.StructMeta]{
				Nodes: []graph.Node[contracttest.StructMeta]{
					{ID: "ok", Content: "good"},
					{ID: "", Content: "bad"},
				},
			},
		}
		store, err := New[contracttest.StructMeta](runner, graph.EmptySchema(), Config[contracttest.StructMeta]{})
		if err != nil {
			t.Fatalf("New(): %v", err)
		}
		return &neo4jGraphRetrieveBackend{Store: store}
	}, func(t *testing.T) retrieval.Backend[contracttest.StructMeta] {
		t.Helper()

		resolver := contracttest.ContentMergeResolver[contracttest.StructMeta]{}
		runner := brokenRunner{
			snapshot: graph.Snapshot[contracttest.StructMeta]{
				Nodes: []graph.Node[contracttest.StructMeta]{
					{ID: "ok", Content: "merge-key"},
					{ID: "", Content: "bad"},
				},
			},
		}
		store, err := New[contracttest.StructMeta](runner, graph.EmptySchema(), Config[contracttest.StructMeta]{
			Resolver: resolver,
		})
		if err != nil {
			t.Fatalf("New(): %v", err)
		}
		return &neo4jGraphRetrieveBackend{Store: store}
	})
}

type neo4jGraphRetrieveBackend struct {
	*Store[contracttest.StructMeta]
}

func (b *neo4jGraphRetrieveBackend) Retrieve(
	ctx context.Context,
	query string,
	opts retrieval.RetrieveOptions,
) (retrieval.ResultSet[contracttest.StructMeta], error) {
	if opts.Graph == nil {
		opts.Graph = &retrieval.GraphOptions{
			Seeds:     []string{"ok"},
			Direction: graph.DirectionOutbound,
			Depth:     1,
		}
	}
	return b.Store.Retrieve(ctx, query, opts)
}

func TestRetrieveBackendFetchLimitTruncatesTraversalOrder(t *testing.T) {
	t.Parallel()

	runner := brokenRunner{
		snapshot: graph.Snapshot[contracttest.StructMeta]{
			Nodes: []graph.Node[contracttest.StructMeta]{
				{ID: "first", Content: "one"},
				{ID: "second", Content: "two"},
				{ID: "third", Content: "three"},
			},
		},
	}
	store, err := New[contracttest.StructMeta](runner, graph.EmptySchema(), Config[contracttest.StructMeta]{})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
		TopK: 1,
		Graph: &retrieval.GraphOptions{
			Seeds:     []string{"first"},
			Direction: graph.DirectionOutbound,
			Depth:     1,
		},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if out.Len() != 1 || out.Documents()[0].ID != "first" {
		t.Fatalf("Documents() = %#v, want first node by traversal order", out.Documents())
	}
}

func TestTraverseWrapsRunnerError(t *testing.T) {
	t.Parallel()

	raw := errors.New("connection reset")
	store, err := New[contracttest.StructMeta](errRunner{
		traverseErr: raw,
	}, graph.EmptySchema(), Config[contracttest.StructMeta]{})
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
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Traverse() error = %v, want unavailable", err)
	}
	if !errors.Is(err, raw) {
		t.Fatalf("error chain lost upstream: %v", err)
	}
}

func TestUpsertWrapsRunnerError(t *testing.T) {
	t.Parallel()

	raw := errors.New("connection reset")
	store, err := New[contracttest.StructMeta](errRunner{
		upsertErr: raw,
	}, graph.EmptySchema(), Config[contracttest.StructMeta]{})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	err = store.Upsert(context.Background(), graph.Snapshot[contracttest.StructMeta]{
		Nodes: []graph.Node[contracttest.StructMeta]{{ID: "n1", Labels: []string{"Doc"}}},
	})
	if err == nil {
		t.Fatal("Upsert() error = nil, want error")
	}
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Upsert() error = %v, want unavailable", err)
	}
	if !errors.Is(err, raw) {
		t.Fatalf("error chain lost upstream: %v", err)
	}
}
