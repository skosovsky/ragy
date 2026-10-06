//go:build darwin || linux

package persistent_test

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense/persistent"
	"github.com/skosovsky/ragy/lifecycle"
)

func TestPersistentSupportInventoryAndOldSchemaCannotAttestReady(t *testing.T) {
	for _, damage := range []string{"supports", "old-schema", "previous-envelope", "missing-inventory"} {
		t.Run(damage, func(t *testing.T) { supportCatalogCase(t, damage) })
	}
}
func supportCatalogCase(t *testing.T, damage string) {
	t.Helper()
	// Arrange: real published files, including original source support inventory.
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
	if _, err = executor.Stage(t.Context(), "n", manifest.ID, "dense", input); err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Publish(t.Context(), "n", manifest.ID); err != nil {
		t.Fatal(err)
	}
	paths, err := filepath.Glob(filepath.Join(config.Root, "*", "*", "catalog.json"))
	if err != nil || len(paths) != 1 {
		t.Fatal("catalog missing", err)
	}
	data, err := os.ReadFile(paths[0])
	if err != nil {
		t.Fatal(err)
	}
	var envelope map[string]json.RawMessage
	if err = json.Unmarshal(data, &envelope); err != nil {
		t.Fatal(err)
	}
	var artifacts []lifecycle.Artifact
	if err = json.Unmarshal(envelope["artifacts"], &artifacts); err != nil || len(artifacts) != len(input) {
		t.Fatal("original inventory not persisted", err)
	}
	// Assert the retained catalog is usable before introducing damage.
	ready, err := adapter.Inspect(t.Context(), lifecycle.StageRequest{Manifest: manifest, Target: "dense"})
	if err != nil || ready.State != lifecycle.TargetReady {
		t.Fatal("intact retained inventory rejected", ready, err)
	}
	switch damage {
	case "supports":
		artifacts[0].Supports[0].Artifact = "different-original"
		envelope["artifacts"], err = json.Marshal(artifacts)
	case "missing-inventory":
		delete(envelope, "artifacts")
	case "previous-envelope":
		envelope["schema"], err = json.Marshal("ragy.dense-index/inventory")
	case "old-schema":
		envelope["schema"], err = json.Marshal("ragy.dense-index")
		delete(envelope, "artifacts")
	}
	if err != nil {
		t.Fatal(err)
	}
	data, err = json.Marshal(envelope)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(paths[0], data, 0o600); err != nil {
		t.Fatal(err)
	}
	restarted, err := persistent.New(config)
	if err != nil {
		t.Fatal(err)
	}
	// Act: payload and artifact reference checksums remain unchanged.
	result, err := restarted.Inspect(t.Context(), lifecycle.StageRequest{Manifest: manifest, Target: "dense"})
	// Assert: changed supports and incompatible old schema never acknowledge readiness.
	if !errors.Is(err, ragy.ErrProtocol) || result.State == lifecycle.TargetReady {
		t.Fatal("unsupported inventory attested ready", result, err)
	}
}
