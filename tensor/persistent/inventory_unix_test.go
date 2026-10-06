//go:build darwin || linux

package persistent_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
	"github.com/skosovsky/ragy/tensor/persistent"
)

func TestActualInventoryBootstrapPreservesUnknownAndFencesMutation(t *testing.T) {
	// Arrange: actual publication plus opaque old data outside managed catalogs.
	config := newConfig(t)
	adapter, err := persistent.New(config)
	if err != nil {
		t.Fatal(err)
	}
	executor := newExecutor(t, config, adapter)
	input := records()
	manifest := plan(input)
	if _, err = executor.Prepare(t.Context(), manifest); err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Stage(t.Context(), "n", manifest.ID, "tensor", input); err != nil {
		t.Fatal(err)
	}
	published, err := executor.Publish(t.Context(), "n", manifest.ID)
	if err != nil {
		t.Fatal(err)
	}
	paths, err := filepath.Glob(filepath.Join(config.Root, "*", "*", "catalog.json"))
	if err != nil || len(paths) != 1 {
		t.Fatal("catalog missing", err)
	}
	root := filepath.Dir(filepath.Dir(paths[0]))
	unknown := filepath.Join(root, "legacy-opaque")
	if err = os.WriteFile(unknown, []byte("unknown host data"), 0o600); err != nil {
		t.Fatal(err)
	}
	inventory := lifecycle.Inventory{
		Namespace: "n",
		Kind:      lifecycle.CompleteInventory,
		Watermark: "actual-1",
		Coverage:  lifecycle.FullInventory,
		Targets:   []string{"tensor"},
		Manifests: []lifecycle.Manifest{published},
	}
	restarted, err := persistent.New(config)
	if err != nil {
		t.Fatal(err)
	}
	verifier, err := lifecycle.NewFencedInventoryVerifier(map[string]lifecycle.InventoryObserver{"tensor": restarted})
	if err != nil {
		t.Fatal(err)
	}
	// Act/Assert: incomplete complete coverage cannot confirm a bootstrap.
	if _, err = verifier.VerifyInventory(t.Context(), inventory); !errors.Is(err, ragy.ErrProtocol) {
		t.Fatal("unknown entry omitted", err)
	}
	inventory.Unmanaged = []lifecycle.UnmanagedRecord{{Target: "tensor", Key: "legacy-opaque"}}
	destination, err := filestore.New(t.TempDir(), 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	bootstrap, err := lifecycle.NewBootstrapper(destination, verifier)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := bootstrap.Import(t.Context(), inventory)
	if err != nil || len(receipt.Imported) != 1 || len(receipt.Unmanaged) != 1 {
		t.Fatal("actual import", receipt, err)
	}
	retained, err := os.ReadFile(unknown)
	if err != nil || string(retained) != "unknown host data" {
		t.Fatal("unknown data touched", err)
	}
	loaded, err := destination.Load(t.Context(), "n")
	if err != nil || len(loaded.Manifests) != 1 || len(loaded.Inventories) != 1 {
		t.Fatal("inventory not durable", err)
	}
	assertInventoryFence(t, restarted, adapter, manifest, input, inventory, verifier)
}

func assertInventoryFence(
	t *testing.T,
	restarted, adapter *persistent.Adapter[metadata],
	manifest lifecycle.Manifest,
	input []persistent.Record[metadata],
	inventory lifecycle.Inventory,
	verifier *lifecycle.FencedInventoryVerifier,
) {
	t.Helper()
	var err error
	// A fresh writer is fenced for the entire observation callback, with no retry.
	callbacks := 0
	err = restarted.ObserveInventory(t.Context(), inventory, func() error {
		callbacks++
		_, stageErr := adapter.Stage(t.Context(), lifecycle.StageRequest{Manifest: manifest, Target: "tensor"}, input)
		if !errors.Is(stageErr, lifecycle.ErrConflict) {
			t.Fatal("writer escaped fence", stageErr)
		}
		return nil
	})
	if err != nil || callbacks != 1 {
		t.Fatal("observation callback", err, callbacks)
	}
	delta := inventory
	delta.Kind = lifecycle.DeltaInventory
	delta.Watermark = "delta-1"
	delta.Unmanaged = nil
	if _, err = verifier.VerifyInventory(t.Context(), delta); err != nil {
		t.Fatal("delta omitted unknown", err)
	}
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err = verifier.VerifyInventory(canceled, inventory); !errors.Is(err, context.Canceled) {
		t.Fatal("cancellation acknowledged", err)
	}
}
