//go:build darwin || linux

package persistent_test

import (
	"testing"
	"time"

	"github.com/skosovsky/ragy/lifecycle"
)

func TestInspectCleanupRejectsForgedDurableSupports(t *testing.T) {
	// Arrange: actual staged/published dense files, followed by supported cleanup.
	cfg := newConfig(t)
	input := records()
	adapter := published(t, cfg, input)
	exec := newExecutor(t, cfg, adapter)
	deleted := plan(input)
	deleted.ID, deleted.Key, deleted.ExpectedPublication = "deleted", "delete", "operation"
	deleted.Identity.Revision = "r2"
	deleted.Tombstone, deleted.Targets = true, nil
	if _, err := exec.Prepare(t.Context(), deleted); err != nil {
		t.Fatal(err)
	}
	if _, err := exec.Publish(t.Context(), "n", deleted.ID); err != nil {
		t.Fatal(err)
	}
	cleaner, err := lifecycle.NewCleaner(
		lifecycle.CleanerConfig{
			Store:   cfg.Store,
			Now:     time.Now,
			Targets: []lifecycle.CleanupRegistration{{Name: "dense", Port: adapter}},
			Policy:  lifecycle.CleanupPolicy{Deadline: time.Minute, Backoff: []time.Duration{time.Second}},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = cleaner.Begin(t.Context(), "n", deleted.ID); err != nil {
		t.Fatal(err)
	}
	if _, err = cleaner.Attempt(t.Context(), "n", deleted.ID, "operation", "dense", false); err != nil {
		t.Fatal(err)
	}
	stored, err := cfg.Store.Load(t.Context(), "n")
	if err != nil {
		t.Fatal(err)
	}
	var owner, retired lifecycle.Manifest
	for _, m := range stored.Manifests {
		if m.ID == "operation" {
			retired = m.Clone()
		}
		if m.ID == deleted.ID {
			owner = m.Clone()
		}
	}
	retired.Targets[0].Artifacts[0].Supports[0].Artifact = "forged-source"
	// Act: the physical catalog is gone; durable ledger is the authoritative support set.
	state, err := adapter.InspectCleanup(
		t.Context(),
		lifecycle.CleanupRequest{Owner: owner, Retired: retired, Target: "dense", ActivePublication: deleted.ID},
	)
	// Assert.
	if err == nil {
		t.Fatalf("forged original supports accepted after cleanup: state=%s", state)
	}
}
