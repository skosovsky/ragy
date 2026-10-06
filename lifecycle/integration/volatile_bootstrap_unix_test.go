//go:build darwin || linux

package integration_test

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/graph"
	graphmanaged "github.com/skosovsky/ragy/graph/managed"
	"github.com/skosovsky/ragy/lexical"
	lexicalmanaged "github.com/skosovsky/ragy/lexical/managed"
	"github.com/skosovsky/ragy/lifecycle"
)

func TestActualVolatileInventoryCoverageBoundsAndLostData(t *testing.T) {
	for _, target := range []string{"lexical", "graph"} {
		t.Run(target, func(t *testing.T) { volatileInventoryCase(t, target) })
	}
}
func volatileInventoryCase(t *testing.T, target string) {
	t.Helper()
	// Arrange: two actually retained source versions, with durable publication.
	f := newFixture(t, target)
	policy := sourceBatch("policy", "r1", []string{"p1", "p2"})
	faq := sourceBatch("faq", "r1", []string{"f1"})
	f.ingest(t, plan("policy-r1", "", target, policy), policy)
	f.ingest(t, plan("faq-r1", "", target, faq), faq)
	snapshot, err := f.store.Load(t.Context(), "fixture-a")
	if err != nil {
		t.Fatal(err)
	}
	inventory := lifecycle.Inventory{
		Namespace: "fixture-a",
		Kind:      lifecycle.CompleteInventory,
		Watermark: "volatile-1",
		Coverage:  lifecycle.FullInventory,
		Targets:   []string{target, "dense"},
	}
	for _, manifest := range snapshot.Manifests {
		if manifest.Identity.Source == "policy" {
			inventory.Manifests = append(inventory.Manifests, manifest)
		}
	}
	observer := secondaryObserver(t, f, 100, 100)
	callback := func() error { return nil }
	// Act/Assert: incomplete complete coverage fails, while delta may omit faq.
	if err = observer.ObserveInventory(t.Context(), inventory, callback); !errors.Is(err, ragy.ErrProtocol) {
		t.Fatal("retained source omitted", err)
	}
	inventory.Kind = lifecycle.DeltaInventory
	if err = observer.ObserveInventory(t.Context(), inventory, callback); err != nil {
		t.Fatal("delta omitted source", err)
	}
	inventory.Kind = lifecycle.CompleteInventory
	inventory.Unmanaged = []lifecycle.UnmanagedRecord{{Target: target, Key: "manifest:faq-r1"}}
	if err = observer.ObserveInventory(t.Context(), inventory, callback); err != nil {
		t.Fatal("opaque retained revision", err)
	}
	assertVolatileInventoryBounds(t, f, inventory)
	if target == "graph" {
		assertHostBasisInventory(t, f, inventory)
	}
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	if err = observer.ObserveInventory(canceled, inventory, callback); !errors.Is(err, context.Canceled) {
		t.Fatal("canceled observation accepted", err)
	}
	// Arrange/Act: a new volatile adapter has lost records despite the old ready ledger.
	restartVolatileInventoryTarget(t, f)
	fresh := secondaryObserver(t, f, 100, 100)
	if err = fresh.ObserveInventory(t.Context(), inventory, callback); !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatal("ready ledger replaced actual data", err)
	}
}
func assertVolatileInventoryBounds(t *testing.T, f *fixture, inventory lifecycle.Inventory) {
	t.Helper()
	for _, bounds := range [][2]int{{1, 100}, {100, 1}} {
		// Arrange/Act/Assert: enumeration and record verification both have explicit caps.
		observer := secondaryObserver(t, f, bounds[0], bounds[1])
		if err := observer.ObserveInventory(
			t.Context(),
			inventory,
			func() error { return nil },
		); !errors.Is(
			err,
			ragy.ErrUnavailable,
		) {
			t.Fatal("bound ignored", bounds, err)
		}
	}
}
func assertHostBasisInventory(t *testing.T, f *fixture, inventory lifecycle.Inventory) {
	t.Helper()
	// Arrange: a host basis has no managed source identity and cannot become a manifest.
	basis := graph.Snapshot[meta]{Nodes: []graph.Node[meta]{f.sourceHostNode()}, Edges: nil}
	if err := f.secondary.graph.SetHostBasis(t.Context(), "foundation", basis); err != nil {
		t.Fatal(err)
	}
	observer := secondaryObserver(t, f, 100, 100)
	if err := observer.ObserveInventory(
		t.Context(),
		inventory,
		func() error { return nil },
	); !errors.Is(
		err,
		ragy.ErrProtocol,
	) {
		t.Fatal("host basis omitted", err)
	}
	inventory.Unmanaged = append(
		inventory.Unmanaged,
		lifecycle.UnmanagedRecord{Target: f.target, Key: "host:foundation"},
	)
	if err := observer.ObserveInventory(t.Context(), inventory, func() error { return nil }); err != nil {
		t.Fatal("host basis not retained opaque", err)
	}
	// Assert it still exists unchanged: the host may not reuse its identity for other facts.
	if err := f.secondary.graph.SetHostBasis(
		t.Context(),
		"foundation",
		graph.Snapshot[meta]{},
	); !errors.Is(
		err,
		lifecycle.ErrConflict,
	) {
		t.Fatal("observer changed host basis", err)
	}
}
func (f *fixture) sourceHostNode() graph.Node[meta] {
	input := sourceBatch("host", "r1", []string{"h1"})
	return input.Graph.Nodes[0].Value
}
func restartVolatileInventoryTarget(t *testing.T, f *fixture) {
	t.Helper()
	var err error
	if f.target == "lexical" {
		f.secondary.lexical, err = lexicalmanaged.New(
			lexicalmanaged.Config[meta]{
				MaxCachedSnapshots: 32,
				Namespace:          "fixture-a",
				Target:             f.target,
				Store:              f.store,
				Schema:             f.schema,
				BM25:               lexical.Config[meta]{SearchFields: []string{"content"}},
				CloneMeta:          cloneMeta,
			},
		)
	} else {
		f.secondary.graph, err = graphmanaged.New(
			graphmanaged.Config[meta]{
				Namespace:  "fixture-a",
				Target:     f.target,
				Store:      f.store,
				Schema:     graph.Schema{NodeAttributes: f.schema, EdgeAttributes: f.schema},
				CloneMeta:  cloneMeta,
				MaxRecords: 100, MaxAdmissionRecords: 100,
			},
		)
	}
	if err != nil {
		t.Fatal(err)
	}
}
