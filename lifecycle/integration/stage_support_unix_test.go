//go:build darwin || linux

package integration_test

import (
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/lifecycle"
)

func TestActualTargetsRejectStageSupportOutsideRegisteredManifest(t *testing.T) {
	for _, target := range []string{"dense", "lexical", "tensor", "graph"} {
		t.Run(target, func(t *testing.T) { registeredSupportCase(t, target) })
	}
}
func registeredSupportCase(t *testing.T, target string) {
	t.Helper()
	// Arrange: durably register dispatch, without writing a target payload.
	secondary := target
	if target == "dense" {
		secondary = "tensor"
	}
	f := newFixture(t, secondary)
	input := sourceBatch("policy", "r1", []string{"p1"})
	manifest := plan("policy-r1", "", secondary, input)
	if _, err := f.executor.Prepare(t.Context(), manifest); err != nil {
		t.Fatal(err)
	}
	snapshot, err := f.store.Load(t.Context(), "fixture-a")
	if err != nil {
		t.Fatal(err)
	}
	snapshot.Manifests[0].State = lifecycle.Unknown
	snapshot.Manifests[0].Checkpoint = lifecycle.Staging
	for i := range snapshot.Manifests[0].Targets {
		if snapshot.Manifests[0].Targets[i].Name == target {
			snapshot.Manifests[0].Targets[i].State = lifecycle.TargetUnknown
		}
	}
	if _, err = f.store.CompareSwap(t.Context(), snapshot.Generation, snapshot); err != nil {
		t.Fatal(err)
	}
	request := lifecycle.StageRequest{Manifest: snapshot.Manifests[0].Clone(), Target: target}
	for i := range request.Manifest.Targets {
		if request.Manifest.Targets[i].Name == target {
			request.Manifest.Targets[i].Artifacts[0].Supports[0].Artifact = "unregistered-original"
		}
	}
	// Act: artifact refs and payload stay identical, but original support is forged.
	var result lifecycle.StageResult
	if target == "dense" {
		result, err = f.dense.Stage(t.Context(), request, input.Dense)
	} else {
		result, err = f.secondary.Stage(t.Context(), request, input)
	}
	// Assert: no target may install provenance different from the durable plan.
	if !errors.Is(err, ragy.ErrProtocol) || result.State == lifecycle.TargetReady {
		t.Fatal("unregistered support staged", result, err)
	}
	request.Manifest = snapshot.Manifests[0].Clone()
	if target == "dense" {
		result, err = f.dense.Inspect(t.Context(), request)
	} else {
		result, err = f.secondary.Inspect(t.Context(), request)
	}
	if err != nil || result.State != lifecycle.TargetPending {
		t.Fatal("rejected stage wrote target data", result, err)
	}
}
