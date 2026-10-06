package managed

import (
	"testing"
	"time"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/lifecycle"
)

func TestAdjacencyExcludesDanglingEndpoints(t *testing.T) {
	// Arrange: edges remain scoped facts, but forbidden/conflicting endpoints are absent.
	admitted := view[struct{}]{
		nodes: map[string]storedNode[struct{}]{"a": {}, "b": {}},
		edges: map[string]storedEdge[struct{}]{
			"valid":      {record: Edge[struct{}]{Value: graph.Edge[struct{}]{SourceID: "a", TargetID: "b"}}},
			"bridge":     {record: Edge[struct{}]{Value: graph.Edge[struct{}]{SourceID: "a", TargetID: "private"}}},
			"conflicted": {record: Edge[struct{}]{Value: graph.Edge[struct{}]{SourceID: "conflict", TargetID: "b"}}},
		},
		outbound: make(map[string][]string), inbound: make(map[string][]string),
	}
	// Act.
	err := indexAdjacency(t.Context(), access.Unrestricted(), &admitted)
	// Assert: indexes have only the edge whose two endpoints survived admission.
	if err != nil || len(admitted.outbound) != 1 || len(admitted.inbound) != 1 ||
		len(
			admitted.outbound["a"],
		) != 1 || admitted.outbound["a"][0] != "valid" || admitted.inbound["b"][0] != "valid" {
		t.Fatal("excluded endpoints entered adjacency", err, admitted.outbound, admitted.inbound)
	}
}

func TestConfirmedRejectsRetiredEmptyInventory(t *testing.T) {
	// Arrange: previously published empty graph inventory retains its identity after retirement.
	captured := lifecycle.Manifest{
		ID:          "empty",
		PublishedAt: time.Unix(1, 0),
		Targets:     []lifecycle.Target{{Name: "graph", State: lifecycle.TargetReady}},
	}
	retired := captured.Clone()
	retired.Retired = true
	snapshot := lifecycle.Snapshot{Manifests: []lifecycle.Manifest{retired}}
	// Act.
	accepted := confirmed(snapshot, captured, "graph")
	// Assert: equal empty target inventories cannot resurrect a retired handle.
	if accepted {
		t.Fatal("retired empty graph version confirmed")
	}
	// A retained retired candidate also cannot be confirmed against a live inventory.
	if confirmed(lifecycle.Snapshot{Manifests: []lifecycle.Manifest{captured}}, retired, "graph") {
		t.Fatal("retired captured graph version confirmed")
	}
}
