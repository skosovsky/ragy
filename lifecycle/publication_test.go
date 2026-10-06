//go:build darwin || linux

package lifecycle_test

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/lifecycle"
)

func TestCapturePublicationStagingDoesNotChangeLogicalInventory(t *testing.T) {
	// Arrange: empty namespace has a real pinned complete-empty identity.
	executor, store, _, _ := executorFixture(t)
	ctx := context.Background()
	empty, err := lifecycle.CapturePublication(ctx, store, "n", []string{"dense", "tensor"})
	if err != nil || empty.IsCurrent() || len(empty.Targets()) != 0 {
		t.Fatal("empty capture became live", err)
	}
	if _, err = executor.Prepare(ctx, plannedManifest()); err != nil {
		t.Fatal(err)
	}
	for _, target := range []string{"dense", "tensor"} {
		if _, err = executor.Stage(ctx, "n", "pub1", target, "payload"); err != nil {
			t.Fatal(err)
		}
	}
	// Act/Assert: staged checkpoint generation is excluded from logical read identity.
	staged, err := lifecycle.CapturePublication(ctx, store, "n", []string{"tensor", "dense"})
	if err != nil || staged.Reference() != empty.Reference() || len(staged.Targets()) != 0 {
		t.Fatal("staging changed published snapshot", err)
	}
	if _, err = executor.Publish(ctx, "n", "pub1"); err != nil {
		t.Fatal(err)
	}
	published, err := lifecycle.CapturePublication(ctx, store, "n", []string{"dense", "tensor"})
	if err != nil || len(published.Targets()) != 2 || published.Reference() == empty.Reference() {
		t.Fatal("ready joint publication missing", err)
	}
	copyInventory := published.Targets()
	copyInventory[0].Revision = "modified"
	if published.Targets()[0].Revision != "r1" {
		t.Fatal("captured publication aliases caller inventory")
	}
}

func TestCaptureStrictlyRejectsMissingPartialTarget(t *testing.T) {
	// Arrange: explicit partial publication has one target unavailable.
	_, store, _, _ := executorFixture(t)
	manifest := manifestFixture()
	manifest.Partial = true
	manifest.Targets[1].State = lifecycle.TargetFailed
	manifest.Targets[1].Revision = ""
	snapshot := lifecycle.Snapshot{
		Schema: lifecycle.SchemaIdentity, Namespace: "n", Manifests: []lifecycle.Manifest{manifest},
		Publications: []lifecycle.Publication{{Source: "policy", Manifest: manifest.ID}},
	}
	if _, err := store.CompareSwap(context.Background(), 0, snapshot); err != nil {
		t.Fatal(err)
	}
	// Act/Assert: strict capture does not substitute another/older tensor revision.
	if _, err := lifecycle.CapturePublication(
		context.Background(),
		store,
		"n",
		[]string{"tensor"},
	); !errors.Is(
		err,
		ragy.ErrUnsupported,
	) {
		t.Fatal("partial target silently captured", err)
	}
	ready, err := lifecycle.CapturePublication(context.Background(), store, "n", []string{"dense"})
	if err != nil || len(ready.Targets()) != 1 {
		t.Fatal("ready target unavailable", err)
	}
}

func TestPartialCaptureAllExcludedRemainsExplicitlyUnavailable(t *testing.T) {
	// Arrange: a real durable publication has dense ready, tensor pending.
	_, store, _, _ := executorFixture(t)
	manifest := manifestFixture()
	manifest.Partial = true
	manifest.Targets[1].State, manifest.Targets[1].Revision = lifecycle.TargetPending, ""
	snapshot := lifecycle.Snapshot{
		Schema: lifecycle.SchemaIdentity, Namespace: "n", Manifests: []lifecycle.Manifest{manifest},
		Publications: []lifecycle.Publication{{Source: "policy", Manifest: manifest.ID}},
	}
	if _, err := store.CompareSwap(t.Context(), 0, snapshot); err != nil {
		t.Fatal(err)
	}
	// Act: request only the unconfirmed target, with explicit partial permission.
	publication, err := lifecycle.CapturePartialPublication(t.Context(), store, "n", []string{"tensor"})
	// Assert: zero pins mean unavailable partial inventory, never complete-empty.
	if err != nil || !publication.IsPartial() || len(publication.Targets()) != 0 {
		t.Fatal("unavailable target became complete-empty", err)
	}
	if err = publication.AdmitTarget("tensor"); !errors.Is(err, ragy.ErrUnsupported) {
		t.Fatal("missing target admitted", err)
	}
}

func TestPartialCaptureEmptyNamespaceRemainsCompleteEmpty(t *testing.T) {
	// Arrange: no active source exists in this durable namespace.
	_, store, _, _ := executorFixture(t)
	// Act.
	publication, err := lifecycle.CapturePartialPublication(t.Context(), store, "n", []string{"dense", "tensor"})
	// Assert: an actual empty namespace has no invented capability failure.
	if err != nil || publication.IsCurrent() || publication.IsPartial() || len(publication.Targets()) != 0 {
		t.Fatal("empty namespace became unavailable", err)
	}
}
