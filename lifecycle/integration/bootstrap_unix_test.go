//go:build darwin || linux

package integration_test

import (
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
)

func TestActualJointInventoryBootstrap(t *testing.T) {
	for _, target := range []string{"lexical", "tensor", "graph"} {
		t.Run("dense+"+target, func(t *testing.T) { actualJointBootstrap(t, target) })
	}
}

func actualJointBootstrap(t *testing.T, target string) {
	t.Helper()
	// Arrange: both targets have physically staged and published actual source data.
	f := newFixture(t, target)
	input := sourceBatch("policy", "r1", []string{"p1", "p2"})
	f.ingest(t, plan("policy-r1", "", target, input), input)
	snapshot, err := f.store.Load(t.Context(), "fixture-a")
	if err != nil {
		t.Fatal(err)
	}
	inventory := lifecycle.Inventory{
		Namespace: "fixture-a",
		Kind:      lifecycle.CompleteInventory,
		Watermark: "joint-1",
		Coverage:  lifecycle.FullInventory,
		Targets:   []string{target, "dense"},
		Manifests: snapshot.Manifests,
	}
	verifier, err := lifecycle.NewFencedInventoryVerifier(
		map[string]lifecycle.InventoryObserver{"dense": f.dense, target: secondaryObserver(t, f, 100, 100)},
	)
	if err != nil {
		t.Fatal(err)
	}
	destination, err := filestore.New(t.TempDir(), 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	bootstrap, err := lifecycle.NewBootstrapper(destination, verifier)
	if err != nil {
		t.Fatal(err)
	}
	// Act: independently verify both actual catalogs before importing ownership.
	receipt, err := bootstrap.Import(t.Context(), inventory)
	// Assert: one durable publication and both exact target pins.
	if err != nil || len(receipt.Imported) != 1 {
		t.Fatal("joint inventory rejected", receipt, err)
	}
	pub, err := lifecycle.CapturePublication(t.Context(), destination, "fixture-a", []string{"dense", target})
	if err != nil || len(pub.Targets()) != 2 {
		t.Fatal("joint bootstrap publication", err)
	}
	// Each named target independently rejects altered original source supports.
	for _, name := range []string{"dense", target} {
		assertForgedBootstrapTarget(t, bootstrap, inventory, name)
	}
	// Assert: no ownership receipt or publication was replaced on rejection.
	after, err := destination.Load(t.Context(), "fixture-a")
	if err != nil || after.Generation != 1 || len(after.Inventories) != 1 {
		t.Fatal("failed bootstrap mutated ownership", after.Generation, err)
	}
}

func secondaryObserver(t *testing.T, f *fixture, maxEntries, maxRecords int) lifecycle.InventoryObserver {
	t.Helper()
	if f.secondary.tensor != nil {
		return f.secondary.tensor
	}
	var observer lifecycle.InventoryObserver
	var err error
	if f.secondary.graph != nil {
		observer, err = f.secondary.graph.InventoryObserver(maxEntries, maxRecords)
	} else {
		observer, err = f.secondary.lexical.InventoryObserver(maxEntries, maxRecords)
	}
	if err != nil {
		t.Fatal(err)
	}
	return observer
}

func assertForgedBootstrapTarget(
	t *testing.T,
	bootstrap *lifecycle.Bootstrapper,
	inventory lifecycle.Inventory,
	name string,
) {
	t.Helper()
	forged := inventory
	forged.Watermark = "joint-forged-" + name
	forged.Manifests = []lifecycle.Manifest{inventory.Manifests[0].Clone()}
	for i := range forged.Manifests[0].Targets {
		if forged.Manifests[0].Targets[i].Name == name {
			forged.Manifests[0].Targets[i].Artifacts[0].Supports[0].Artifact = "forged-original"
		}
	}
	_, err := bootstrap.Import(t.Context(), forged)
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatal("forged target supports accepted", name, err)
	}
}

func TestActualSingleVolatileTargetBootstrap(t *testing.T) {
	for _, target := range []string{"lexical", "graph"} {
		t.Run(target, func(t *testing.T) { singleVolatileBootstrap(t, target) })
	}
}
func singleVolatileBootstrap(t *testing.T, target string) {
	t.Helper()
	// Arrange: an actual retained target can be imported as a single-target profile.
	f := newFixture(t, target)
	input := sourceBatch("policy", "r1", []string{"p1", "p2"})
	f.ingest(t, plan("policy-r1", "", target, input), input)
	snapshot, err := f.store.Load(t.Context(), "fixture-a")
	if err != nil {
		t.Fatal(err)
	}
	manifest := snapshot.Manifests[0].Clone()
	for _, entry := range manifest.Targets {
		if entry.Name == target {
			manifest.Targets = []lifecycle.Target{entry}
			break
		}
	}
	inventory := lifecycle.Inventory{
		Namespace: "fixture-a",
		Kind:      lifecycle.CompleteInventory,
		Watermark: "single-1",
		Coverage:  lifecycle.FullInventory,
		Targets:   []string{target},
		Manifests: []lifecycle.Manifest{manifest},
	}
	verifier, err := lifecycle.NewFencedInventoryVerifier(
		map[string]lifecycle.InventoryObserver{target: secondaryObserver(t, f, 100, 100)},
	)
	if err != nil {
		t.Fatal(err)
	}
	destination, err := filestore.New(t.TempDir(), 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	bootstrap, err := lifecycle.NewBootstrapper(destination, verifier)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	receipt, err := bootstrap.Import(t.Context(), inventory)
	// Assert: only the selected target is pinned; other target data is not adopted.
	if err != nil || len(receipt.Imported) != 1 {
		t.Fatal("single target import", receipt, err)
	}
	publication, err := lifecycle.CapturePublication(t.Context(), destination, "fixture-a", []string{target})
	if err != nil || len(publication.Targets()) != 1 {
		t.Fatal("single target publication", err)
	}
}
