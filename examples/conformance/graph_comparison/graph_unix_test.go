//go:build darwin || linux

package main

import (
	"context"

	"slices"
	"testing"

	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/graph/managed"

	"github.com/skosovsky/ragy/graphingest/resolution"
	"github.com/skosovsky/ragy/source"
)

func deterministicSourceExtractions(t *testing.T, f fixture) []sourceExtraction {
	t.Helper()
	// Explicit contract model outputs, independent from the gold entity/edge lists.
	rows := []struct{ id, service, db, team string }{
		{"s1", "Pay", "LedgerDB", "Team A"}, {"s2", billingName, "LedgerDB", ""},
		{"s3", billingName, "", "Team B"}, {"s4", "Search", "IndexDB", "Team B"},
	}
	var batches []sourceExtraction
	for _, values := range rows {
		var row sourceRow
		for _, s := range f.Sources {
			if s.ID == values.id {
				row = s
			}
		}
		mapping, err := mappedSource(row)
		if err != nil {
			t.Fatal(err)
		}
		supports := mapping.Supports()
		entity := func(id, name, kind string) resolution.Entity[string, graphAttributes] {
			return resolution.Entity[string, graphAttributes]{
				ID:        id,
				Namespace: row.Namespace,
				Name:      name,
				Kind:      kind,
				Supports:  supports,
			}
		}
		relation := func(id, to, kind string) resolution.Relation[string, graphAttributes] {
			return resolution.Relation[string, graphAttributes]{
				ID:       id,
				From:     "service",
				To:       to,
				Kind:     kind,
				Supports: supports,
			}
		}
		value := resolution.Extraction[string, string, graphAttributes]{
			Entities: []resolution.Entity[string, graphAttributes]{entity("service", values.service, serviceKind)},
		}
		if values.db != "" {
			value.Entities = append(value.Entities, entity("database", values.db, "Database"))
			value.Relations = append(value.Relations, relation("dependency", "database", "depends_on"))
		}
		if values.team != "" {
			value.Entities = append(value.Entities, entity("team", values.team, "Team"))
			value.Relations = append(value.Relations, relation("ownership", "team", "owned_by"))
		}
		batches = append(
			batches,
			sourceExtraction{SourceID: row.ID, Configuration: digest([]byte("fixed-contract-model")), Value: value},
		)
	}
	return batches
}
func TestActualResolverMaterializerPublishedGraphMatchesGold(t *testing.T) {
	// Arrange: actual durable targets and independent deterministic extractor output.
	var f fixture
	if err := decodeStrict(fixtureJSON, &f); err != nil {
		t.Fatal(err)
	}
	dense, err := buildDenseCorpus(t.Context(), t.TempDir(), f)
	if err != nil {
		t.Fatal(err)
	}
	read, err := dense.bind(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	corpus, err := buildGraphCorpus(t.Context(), t.TempDir(), dense, read, deterministicSourceExtractions(t, f))
	if err != nil {
		t.Fatal(err)
	}
	targets, err := corpus.targets(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	pinned, err := dense.bind(t.Context(), targets)
	if err != nil {
		t.Fatal(err)
	}
	var seeds []string
	for _, e := range corpus.resolved.Entities {
		seeds = append(seeds, e.ID)
	}
	actual, err := corpus.adapter.Traverse(
		context.Background(),
		managed.Request{
			Read:      pinned,
			Traversal: graph.TraversalRequest{Seeds: seeds, Direction: graph.DirectionUndirected, Depth: 1},
			MaxNodes:  localNodeCap,
			MaxEdges:  localEdgeCap,
		},
	)
	// Assert: exact canonical graph and original support inventory, no guessed names.
	if err != nil || len(actual.Conflicts) != 0 || len(corpus.resolved.Unresolved) != 0 {
		t.Fatal(actual, err)
	}
	assertGoldGraph(t, f, actual)
}
func assertGoldGraph(t *testing.T, f fixture, actual managed.Result[graphMetadata]) {
	t.Helper()
	if len(actual.Snapshot.Nodes) != len(f.Entities) || len(actual.Snapshot.Edges) != len(f.Edges) {
		t.Fatal(actual.Snapshot)
	}
	keys := make(map[string]string)
	for _, node := range actual.Snapshot.Nodes {
		keys[node.ID] = node.Meta.SourceID
		found := false
		for _, gold := range f.Entities {
			if gold.Key == node.Meta.SourceID {
				found = true
				assertGoldSupports(t, actual, "node", node.ID, gold.Supports)
			}
		}
		if !found {
			t.Fatal("unexpected canonical entity", node)
		}
	}
	for _, e := range actual.Snapshot.Edges {
		found := false
		for _, gold := range f.Edges {
			if gold.From == keys[e.SourceID] && gold.To == keys[e.TargetID] && gold.Kind == e.Type {
				found = true
				assertGoldSupports(t, actual, "edge", e.ID, gold.Supports)
			}
		}
		if !found {
			t.Fatal("unexpected canonical relation", e)
		}
	}
}
func assertGoldSupports(t *testing.T, actual managed.Result[graphMetadata], kind, id string, sourceIDs []string) {
	t.Helper()
	var expected []source.Reference
	for _, src := range sourceIDs {
		expected = append(expected, originalReference(src))
	}
	for _, support := range actual.Supports {
		if support.Kind != kind || support.ID != id {
			continue
		}
		if len(support.References) != len(expected) || len(support.HostBases) != 0 {
			t.Fatal(support, expected)
		}
		for _, ref := range expected {
			if !slices.Contains(support.References, ref) {
				t.Fatal(support, expected)
			}
		}
		return
	}
	t.Fatal("missing source support", kind, id)
}

func TestGraphProducerRejectsForeignNamespaceAndSourceSupport(t *testing.T) {
	for _, scenario := range []string{"namespace", "foreign-support", "duplicate-source", "missing-source", "configuration"} {
		t.Run(scenario, func(t *testing.T) {
			// Arrange: host source namespace and references cannot come from model decisions.
			var f fixture
			if err := decodeStrict(fixtureJSON, &f); err != nil {
				t.Fatal(err)
			}
			batches := deterministicSourceExtractions(t, f)
			switch scenario {
			case "namespace":
				batches[0].Value.Entities[0].Namespace = "staging"
			case "foreign-support":
				batches[0].Value.Entities[0].Supports = []source.Locator{
					{Reference: originalReference("s2"), Kind: source.DocumentLocation},
				}
			case "duplicate-source":
				batches[1].SourceID = batches[0].SourceID
			case "missing-source":
				batches = batches[:len(batches)-1]
			case "configuration":
				batches[0].Configuration = "unbound-model-config"
			}
			// Act.
			_, err := combineExtractions(f, batches)
			// Assert: no target builder or I/O port is needed to reject this input.
			if err == nil {
				t.Fatal("untrusted source binding accepted")
			}
		})
	}
}
func TestGraphProducerRetainsConflictWithoutSelectingWinner(t *testing.T) {
	// Arrange: contradictory typed owner attributes on the same production identity.
	var f fixture
	if err := decodeStrict(fixtureJSON, &f); err != nil {
		t.Fatal(err)
	}
	dense, err := buildDenseCorpus(t.Context(), t.TempDir(), f)
	if err != nil {
		t.Fatal(err)
	}
	read, err := dense.bind(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	batches := deterministicSourceExtractions(t, f)
	batches[0].Value.Entities[0].Attributes.Owner = "Team A"
	batches[1].Value.Entities[0].Attributes.Owner = "Team B"
	combined, err := combineExtractions(f, batches)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	result, err := resolveGraph(t.Context(), read, f, combined)
	if err != nil {
		t.Fatal(err)
	}
	published, buildErr := buildGraphCorpus(t.Context(), t.TempDir(), dense, read, batches)
	// Assert: retain both variants/supports; a publication does not choose a winner.
	if buildErr != nil {
		t.Fatal(buildErr)
	}
	assertOwnerConflict(t, result)
	targets, err := published.targets(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	pinned, err := dense.bind(t.Context(), targets)
	if err != nil {
		t.Fatal(err)
	}
	var seeds []string
	for _, e := range result.Entities {
		seeds = append(seeds, e.ID)
	}
	actual, err := published.adapter.Traverse(
		t.Context(),
		managed.Request{
			Read:      pinned,
			Traversal: graph.TraversalRequest{Seeds: seeds, Direction: graph.DirectionUndirected, Depth: 1},
			MaxNodes:  localNodeCap,
			MaxEdges:  localEdgeCap,
		},
	)
	if err != nil || len(actual.Conflicts) != 1 || len(actual.Conflicts[0].References) != 2 {
		t.Fatal(actual, err)
	}
	for _, node := range actual.Snapshot.Nodes {
		if node.Meta.SourceID == productionNamespace+"/Service/Billing" {
			t.Fatal("conflicting owner winner delivered", node)
		}
	}
}
func assertOwnerConflict(t *testing.T, result resolution.Result[string, string, graphAttributes]) {
	t.Helper()
	found := false
	for _, group := range result.Entities {
		if group.Identity.Namespace == productionNamespace && group.Identity.Name == billingName {
			found = true
			if len(group.Variants) != 2 || group.Variants[0].Attributes.Owner == group.Variants[1].Attributes.Owner {
				t.Fatal(group)
			}
			var supports []string
			for _, variant := range group.Variants {
				for _, loc := range variant.Supports {
					supports = append(supports, loc.Reference.Source)
				}
			}
			if len(supports) != 2 || !slices.Contains(supports, "s1") || !slices.Contains(supports, "s2") {
				t.Fatal(group)
			}
		}
	}
	if !found {
		t.Fatal("missing canonical conflict")
	}
}
