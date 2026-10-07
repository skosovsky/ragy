package neo4j

import (
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/contracttest"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/retrieval"
)

func TestRetrieveRejectsGenericFilterBeforeRunner(t *testing.T) {
	// Arrange: node-versus-edge semantics cannot be inferred from generic metadata.
	fields := filter.NewSchema()
	field, err := fields.String("tenant")
	if err != nil {
		t.Fatal(err)
	}
	schema, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	builder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	condition, err := filter.Eq(builder, field, "a").Build()
	if err != nil {
		t.Fatal(err)
	}
	runner := &deliveryRunner{}
	store, err := New(runner, graph.EmptySchema(), Config[contracttest.StructMeta]{})
	if err != nil {
		t.Fatal(err)
	}
	for _, channel := range []string{"options", "plan-filter", "plan-range"} {
		runner.calls = 0
		request := retrieval.Query[struct{}]{Read: retrieval.UnrestrictedRead(), Options: retrieval.RetrieveOptions{
			TopK: 1, Graph: &retrieval.GraphOptions{Seeds: []string{"n"}, Direction: graph.DirectionOutbound, Depth: 1},
		}}
		switch channel {
		case "options":
			request.Options.Filters = condition
		case "plan-filter":
			request.Plan = &retrieval.PlannedQuery[struct{}]{Filters: condition}
		case "plan-range":
			request.Plan = &retrieval.PlannedQuery[struct{}]{Ranges: []retrieval.RangeConstraint{{Field: "tenant"}}}
		}
		// Act.
		result, err := store.Retrieve(t.Context(), request)
		// Assert: every generic metadata channel fails before host traversal.
		if !errors.Is(err, ragy.ErrUnsupported) || result.Len() != 0 || runner.calls != 0 {
			t.Fatal("generic predicate dispatched", channel, result, err, runner.calls)
		}
	}
}

func TestRetrieveValidatesProjectionNotWholeGraphSnapshot(t *testing.T) {
	// Arrange: valid document fields but invalid graph label; traversal rejects it.
	runner := &deliveryRunner{
		snapshot: graph.Snapshot[contracttest.StructMeta]{
			Nodes: []graph.Node[contracttest.StructMeta]{{ID: "n", Content: "text", Labels: []string{"invalid-label"}}},
		},
	}
	store, err := New(runner, graph.EmptySchema(), Config[contracttest.StructMeta]{})
	if err != nil {
		t.Fatal(err)
	}
	options := &retrieval.GraphOptions{Seeds: []string{"n"}, Direction: graph.DirectionOutbound, Depth: 1}
	// Act.
	documents, projectionErr := retrieveStore(
		t.Context(),
		store,
		"",
		retrieval.RetrieveOptions{TopK: 1, Graph: options},
	)
	snapshot, traversalErr := store.Traverse(
		t.Context(),
		graph.TraversalRequest{Seeds: options.Seeds, Direction: options.Direction, Depth: options.Depth},
	)
	// Assert: explicit projection contract remains distinct from graph administration.
	if projectionErr != nil || documents.Len() != 1 || traversalErr == nil || len(snapshot.Nodes) != 0 ||
		runner.calls != 2 {
		t.Fatal("projection/full graph contract", documents, projectionErr, snapshot, traversalErr, runner.calls)
	}
}
