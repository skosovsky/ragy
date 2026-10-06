package managed_test

import (
	"context"
	"errors"
	"slices"
	"sync/atomic"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/graph/managed"
)

func TestAdmissionCountsSharedFactsBeforeDeduplication(t *testing.T) {
	// Arrange: two publications share identical facts; each write fits its own bound.
	f := newFixtureWithAdmission(t, 5)
	for _, id := range []string{"policy", "faq"} {
		input := payload(id, "r1")
		f.ingest(t, plan(id, "", input, id), input)
	}
	req := request(f.pin(t))
	f.calls = nil
	if err := f.adapter.AdmitTraversal(t.Context(), req); err != nil {
		t.Fatal(err)
	}
	// Act: both traversal and lookup load six selected records before deduplication.
	traversed, traversalErr := f.adapter.Traverse(t.Context(), req)
	found, lookupErr := f.adapter.FindByIDs(t.Context(), req)
	// Assert: no partial result or delivery callback despite three unique facts.
	if !errors.Is(traversalErr, ragy.ErrInvalidArgument) || !errors.Is(lookupErr, ragy.ErrInvalidArgument) ||
		len(traversed.Snapshot.Nodes) != 0 || len(found.Snapshot.Nodes) != 0 || len(f.calls) != 0 {
		t.Fatal("shared selected records bypassed admission bound", traversalErr, lookupErr, f.calls)
	}
}

func TestAdmissionCountsExcludedHostBasisBeforeScopeFiltering(t *testing.T) {
	// Arrange: managed facts fit exactly; a forbidden host basis is still scanned.
	f := newFixtureWithAdmission(t, 3)
	input := payload("policy", "r1")
	f.ingest(t, plan("policy", "", input, "p1"), input)
	basis := hostSnapshot(input)
	for i := range basis.Nodes {
		basis.Nodes[i].Meta.Tenant = "b"
	}
	for i := range basis.Edges {
		basis.Edges[i].Meta.Tenant = "b"
	}
	if err := f.adapter.SetHostBasis(t.Context(), "excluded", basis); err != nil {
		t.Fatal(err)
	}
	req := request(f.pin(t))
	req.HostBasis = "excluded"
	f.calls = nil
	// Act.
	result, err := f.adapter.Traverse(t.Context(), req)
	// Assert: output budgets and highly selective scope cannot hide admission work.
	if !errors.Is(err, ragy.ErrInvalidArgument) || len(result.Supports) != 0 || len(f.calls) != 0 {
		t.Fatal("scope bypassed selected admission bound", err, f.calls)
	}
}

func TestManagedAdjacencyDirectionsCyclesAndSelfLoop(t *testing.T) {
	for _, direction := range []graph.Direction{graph.DirectionOutbound, graph.DirectionInbound, graph.DirectionUndirected} {
		t.Run(string(direction), func(t *testing.T) {
			// Arrange: a cycle and self-loop, plus a private edge joining admitted nodes.
			f := newFixture(t)
			input := payload("policy", "r1")
			input.Edges = append(
				input.Edges,
				managed.Edge[metadata]{
					Reference: ref("policy", "r1", "back", "graph-edge"),
					Value: graph.Edge[metadata]{
						ID:       "back",
						SourceID: "db",
						TargetID: "svc",
						Type:     "depends_on",
						Meta:     metadata{Tenant: "a", Visibility: "public", Name: "back"},
					},
				},
				managed.Edge[metadata]{
					Reference: ref("policy", "r1", "loop", "graph-edge"),
					Value: graph.Edge[metadata]{
						ID:       "loop",
						SourceID: "svc",
						TargetID: "svc",
						Type:     "depends_on",
						Meta:     metadata{Tenant: "a", Visibility: "public", Name: "loop"},
					},
				},
				managed.Edge[metadata]{
					Reference: ref("policy", "r1", "private", "graph-edge"),
					Value: graph.Edge[metadata]{
						ID:       "private",
						SourceID: "svc",
						TargetID: "db",
						Type:     "depends_on",
						Meta:     metadata{Tenant: "b", Visibility: "private", Name: "private"},
					},
				},
			)
			f.ingest(t, plan("policy", "", input, "p1"), input)
			req := request(f.pin(t))
			req.Traversal.Direction = direction
			req.Traversal.Depth = 20
			req.MaxEdges = 3
			f.calls = nil
			// Act.
			result, err := f.adapter.Traverse(t.Context(), req)
			// Assert: all admitted incident edges once, never the private edge.
			ids := make([]string, 0, len(result.Snapshot.Edges))
			for _, edge := range result.Snapshot.Edges {
				ids = append(ids, edge.ID)
			}
			if err != nil || len(result.Snapshot.Nodes) != 2 || !slices.Equal(ids, []string{"back", "e1", "loop"}) ||
				slices.Contains(f.calls, "private") {
				t.Fatal("direction or loop adjacency mismatch", err, ids, f.calls)
			}
		})
	}
}

func TestManagedAdmissionRevocationBarrierFailsClosed(t *testing.T) {
	// Arrange: suspend a freshness gate after request preparation, before delivery.
	f := newFixture(t)
	input := payload("policy", "r1")
	f.ingest(t, plan("policy", "", input, "p1"), input)
	entered, release := make(chan struct{}), make(chan struct{})
	var calls atomic.Int32
	var revoked atomic.Bool
	read := f.pinWithAuthority(t, access.AuthorityFunc(func(context.Context, access.Snapshot) error {
		if calls.Add(1) == 8 {
			close(entered)
			<-release
		}
		if revoked.Load() {
			return ragy.ErrUnavailable
		}
		return nil
	}))
	f.calls = nil
	type outcome struct {
		result managed.Result[metadata]
		err    error
	}
	done := make(chan outcome, 1)
	// Act: revoke while the actual reader is held at a deterministic gate.
	go func() {
		result, err := f.adapter.Traverse(t.Context(), request(read))
		done <- outcome{result: result, err: err}
	}()
	<-entered
	revoked.Store(true)
	close(release)
	got := <-done
	// Assert: admitted maps cannot make revocation stale and callbacks remain absent.
	if !errors.Is(got.err, ragy.ErrUnavailable) || len(got.result.Snapshot.Nodes) != 0 ||
		len(got.result.Supports) != 0 ||
		len(f.calls) != 0 {
		t.Fatal("revoked admission delivered payload", got.err, f.calls)
	}
}

func TestAdmissionExactLimitAndMandatoryConfiguration(t *testing.T) {
	// Arrange: total selected shared inventory fits exactly.
	f := newFixtureWithAdmission(t, 6)
	for _, id := range []string{"policy", "faq"} {
		input := payload(id, "r1")
		f.ingest(t, plan(id, "", input, id), input)
	}
	// Act.
	result, err := f.adapter.Traverse(t.Context(), request(f.pin(t)))
	// Assert: no off-by-one refusal and both independent supports remain attached.
	if err != nil || len(result.Supports) != 3 || len(result.Supports[0].References) != 2 {
		t.Fatal("exact admission limit lost shared supports", err, result.Supports)
	}
	for _, limit := range []int{0, -1} {
		// Arrange: valid independent write capacity but missing/invalid read profile.
		config := managed.Config[metadata]{Namespace: "n", Target: "graph", Store: f.store,
			Schema: graph.Schema{
				NodeAttributes: f.schema,
				EdgeAttributes: f.schema,
			}, MaxRecords: 100, MaxAdmissionRecords: limit,
			CloneMeta: func(meta metadata) (metadata, error) { return meta, nil },
		}
		// Act.
		adapter, configErr := managed.New(config)
		// Assert.
		if adapter != nil || !errors.Is(configErr, ragy.ErrInvalidArgument) {
			t.Fatal("unbounded admission accepted", limit, configErr)
		}
	}
}
