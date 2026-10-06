//go:build darwin || linux

package integration_test

import (
	"testing"

	"github.com/skosovsky/ragy/graph"
	graphmanaged "github.com/skosovsky/ragy/graph/managed"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/source"
)

func TestActualJointSharedGraphSupportCleanup(t *testing.T) {
	for _, basis := range []bool{false, true} {
		t.Run(
			map[bool]string{false: "managed-only", true: "host-basis"}[basis],
			func(t *testing.T) { jointSharedGraphCase(t, basis) },
		)
	}
}
func jointSharedGraphCase(t *testing.T, withBasis bool) {
	t.Helper()
	// Arrange: the declared source chunks support identical service-a -> db-x / e1.
	f := newFixture(t, "graph")
	policy := sharedGraphBatch("policy", []string{"p1", "p2"})
	faq := sharedGraphBatch("faq", []string{"f1"})
	f.ingest(t, sharedGraphPlan("policy-r1", policy), policy)
	f.ingest(t, sharedGraphPlan("faq-r1", faq), faq)
	basis := ""
	if withBasis {
		basis = "foundation"
		snapshot := graph.Snapshot[meta]{
			Nodes: []graph.Node[meta]{policy.Graph.Nodes[0].Value, policy.Graph.Nodes[1].Value},
			Edges: []graph.Edge[meta]{policy.Graph.Edges[0].Value},
		}
		if err := f.secondary.graph.SetHostBasis(t.Context(), basis, snapshot); err != nil {
			t.Fatal(err)
		}
	}
	assertSharedGraphEdge(t, f, basis, []string{"policy:p1", "faq:f1"}, true)
	// Act: remove policy support through actual joint tombstone and both-target cleanup.
	deleteJointSource(t, f, "policy", "policy-r1", policy)
	assertSharedGraphEdge(t, f, basis, []string{"faq:f1"}, true)
	denseDocs, _ := f.read(t, f.pin(t), policy, faq)
	// Graph fact metadata is canonical; dense records still have actual source identity.
	assertRevision(t, denseDocs, "r1", 0)
	// Act: last managed support disappears; a host foundation is independent ownership.
	deleteJointSource(t, f, "faq", "faq-r1", faq)
	assertSharedGraphEdge(t, f, basis, nil, withBasis)
	if withBasis {
		assertSharedGraphEdge(t, f, "", nil, false)
	}
}
func sharedGraphBatch(sourceID string, chunks []string) batch {
	out := sourceBatch(sourceID, "r1", chunks)
	out.Graph.Nodes = nil
	for _, row := range []struct{ id, label string }{{"service-a", "Service"}, {"db-x", "Database"}} {
		ref := out.Dense[0].Reference
		ref.Artifact = row.id
		ref.Representation = "graph-node"
		out.Graph.Nodes = append(
			out.Graph.Nodes,
			graphmanaged.Node[meta]{
				Reference: ref,
				Value: graph.Node[meta]{
					ID:      row.id,
					Labels:  []string{row.label},
					Content: row.id,
					Meta:    meta{Tenant: "a", Visibility: "public", Artifact: row.id},
				},
			},
		)
	}
	ref := out.Dense[0].Reference
	ref.Artifact = "e1"
	ref.Representation = "graph-edge"
	out.Graph.Edges = []graphmanaged.Edge[meta]{
		{
			Reference: ref,
			Value: graph.Edge[meta]{
				ID:       "e1",
				SourceID: "service-a",
				TargetID: "db-x",
				Type:     "depends_on",
				Meta:     meta{Tenant: "a", Visibility: "public", Artifact: "e1"},
			},
		},
	}
	return out
}
func sharedGraphPlan(id string, input batch) lifecycle.Manifest {
	manifest := plan(id, "", "graph", input)
	original := input.Dense[0].Reference
	original.Representation = "utf8"
	for i := range manifest.Targets {
		if manifest.Targets[i].Name == "graph" {
			for j := range manifest.Targets[i].Artifacts {
				manifest.Targets[i].Artifacts[j].Supports = []source.Reference{original}
			}
		}
	}
	return manifest
}
func deleteJointSource(t *testing.T, f *fixture, sourceID, expected string, input batch) {
	t.Helper()
	deleted := plan("delete-"+sourceID, expected, "graph", input)
	deleted.Identity.Revision = "deleted"
	deleted.Tombstone, deleted.Targets = true, nil
	if _, err := f.executor.Prepare(t.Context(), deleted); err != nil {
		t.Fatal(err)
	}
	if _, err := f.executor.Publish(t.Context(), "fixture-a", deleted.ID); err != nil {
		t.Fatal(err)
	}
	cleaner := newCleaner(t, f)
	if _, err := cleaner.Begin(t.Context(), "fixture-a", deleted.ID); err != nil {
		t.Fatal(err)
	}
	for _, target := range []string{"dense", "graph"} {
		if _, err := cleaner.Attempt(t.Context(), "fixture-a", deleted.ID, expected, target, false); err != nil {
			t.Fatal(err)
		}
	}
}
func assertSharedGraphEdge(t *testing.T, f *fixture, basis string, want []string, present bool) {
	t.Helper()
	result, err := f.secondary.graph.Traverse(
		t.Context(),
		graphmanaged.Request{
			Read:      f.pin(t),
			HostBasis: basis,
			Traversal: graph.TraversalRequest{
				Seeds:     []string{"service-a"},
				Direction: graph.DirectionOutbound,
				Depth:     1,
			},
			MaxNodes: 50,
			MaxEdges: 100,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !present {
		if len(result.Snapshot.Edges) != 0 || len(result.Snapshot.Nodes) != 0 {
			t.Fatal("unsupported derived facts retained", result.Snapshot)
		}
		return
	}
	if len(result.Snapshot.Edges) != 1 || result.Snapshot.Edges[0].ID != "e1" || len(result.Snapshot.Nodes) != 2 {
		t.Fatal("shared graph fact disappeared", result.Snapshot)
	}
	var supports map[string]bool
	for _, fact := range result.Supports {
		if fact.Kind == "edge" && fact.ID == "e1" {
			supports = sharedEdgeSupportKeys(t, fact, basis)
			break
		}
	}
	if supports == nil {
		t.Fatal("edge support evidence missing")
	}
	if len(supports) != len(want) {
		t.Fatal("wrong support accounting", supports, want)
	}
	for _, key := range want {
		if !supports[key] {
			t.Fatal("original source support missing", key)
		}
	}
}

func sharedEdgeSupportKeys(t *testing.T, fact graphmanaged.Support, basis string) map[string]bool {
	t.Helper()
	supports := map[string]bool{}
	for _, ref := range fact.References {
		if ref.Revision != "r1" || ref.Representation != "utf8" {
			t.Fatal("invented original support", ref)
		}
		supports[ref.Source+":"+ref.Artifact] = true
	}
	if basis != "" && (len(fact.HostBases) != 1 || fact.HostBases[0] != basis) {
		t.Fatal("host basis ownership changed", fact)
	}
	return supports
}
