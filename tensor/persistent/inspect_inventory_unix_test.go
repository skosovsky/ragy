//go:build darwin || linux

package persistent_test

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/tensor/persistent"
)

func TestInspectRejectsChangedArtifactInventory(t *testing.T) {
	// Arrange: actual durable payloads, then a new adapter inspects the same operation.
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
	restarted, err := persistent.New(config)
	if err != nil {
		t.Fatal(err)
	}
	for _, change := range []string{"missing", "replacement", "target-missing", "duplicate"} {
		t.Run(change, func(t *testing.T) {
			request := manifest
			request.Targets = slices.Clone(manifest.Targets)
			request.Targets[0].Artifacts = slices.Clone(manifest.Targets[0].Artifacts)
			switch change {
			case "missing":
				request.Targets[0].Artifacts = request.Targets[0].Artifacts[1:]
			case "replacement":
				request.Targets[0].Artifacts[0].Reference.Artifact = "unwritten"
			case "target-missing":
				request.Targets[0].Name = "other"
			case "duplicate":
				request.Targets[0].Artifacts[1] = request.Targets[0].Artifacts[0]
			}
			// Act.
			result, inspectErr := restarted.Inspect(
				t.Context(),
				lifecycle.StageRequest{Manifest: request, Target: "tensor"},
			)
			// Assert: same identity/payload fingerprint cannot attest another artifact set.
			if inspectErr == nil || result.State == lifecycle.TargetReady {
				t.Fatal("unwritten inventory attested ready", result, inspectErr)
			}
			if !errors.Is(inspectErr, ragy.ErrProtocol) && !errors.Is(inspectErr, ragy.ErrInvalidArgument) {
				t.Fatal("unexpected inventory error", inspectErr)
			}
		})
	}
	// Act: input order changes do not change the artifact set.
	reordered := manifest
	reordered.Targets = slices.Clone(manifest.Targets)
	reordered.Targets[0].Artifacts = slices.Clone(manifest.Targets[0].Artifacts)
	slices.Reverse(reordered.Targets[0].Artifacts)
	result, err := restarted.Inspect(
		t.Context(),
		lifecycle.StageRequest{Manifest: reordered, Target: "tensor"},
	)
	// Assert.
	if err != nil || result.State != lifecycle.TargetReady {
		t.Fatal("set order affected recovery", err)
	}
}

func TestInspectRejectsIncompletePersistedCatalog(t *testing.T) {
	// Arrange: keep all durable payload files, but lose one catalog descriptor.
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
	files, err := filepath.Glob(filepath.Join(config.Root, "*", "*", "catalog.json"))
	if err != nil || len(files) != 1 {
		t.Fatal("catalog missing", err)
	}
	data, err := os.ReadFile(files[0])
	if err != nil {
		t.Fatal(err)
	}
	var catalog map[string]json.RawMessage
	if err = json.Unmarshal(data, &catalog); err != nil {
		t.Fatal(err)
	}
	var descriptors []json.RawMessage
	if err = json.Unmarshal(catalog["records"], &descriptors); err != nil {
		t.Fatal(err)
	}
	catalog["records"], err = json.Marshal(descriptors[1:])
	if err != nil {
		t.Fatal(err)
	}
	data, err = json.Marshal(catalog)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(files[0], data, 0o600); err != nil {
		t.Fatal(err)
	}
	restarted, err := persistent.New(config)
	if err != nil {
		t.Fatal(err)
	}
	// Act: payload checksums remain correct for all remaining descriptors.
	result, err := restarted.Inspect(t.Context(), lifecycle.StageRequest{Manifest: manifest, Target: "tensor"})
	// Assert: a truncated inventory cannot acknowledge the complete planned target.
	if !errors.Is(err, ragy.ErrProtocol) || result.State == lifecycle.TargetReady {
		t.Fatal("incomplete durable catalog attested complete", result, err)
	}
}
