//go:build darwin || linux

package managed_test

import (
	"errors"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/lifecycle"
)

func TestLexicalCleanupRequiresRegisteredExactInventory(t *testing.T) {
	for _, scenario := range []string{"unregistered", "not-dispatched", "forged-support"} {
		t.Run(scenario, func(t *testing.T) { lexicalCleanupInventoryCase(t, scenario) })
	}
}
func lexicalCleanupInventoryCase(t *testing.T, scenario string) {
	t.Helper()
	// Arrange: a readable retired revision and a published replacement on actual BM25.
	f := newFixture(t)
	first, old := sourcePlan("old", "policy", "r1", "")
	second, newer := sourcePlan("new", "policy", "r2", "old")
	f.ingest(t, first, old, true)
	f.ingest(t, second, newer, true)
	snapshot, err := f.store.Load(t.Context(), "n")
	if err != nil {
		t.Fatal(err)
	}
	var request lifecycle.CleanupRequest
	request.Target = "lexical"
	request.ActivePublication = "new"
	for _, manifest := range snapshot.Manifests {
		switch manifest.ID {
		case "old":
			request.Retired = manifest.Clone()
		case "new":
			request.Owner = manifest.Clone()
		}
	}
	request.Retained = request.Retired.Targets[0].Artifacts
	if scenario != "unregistered" {
		registerLexicalCleanupDispatch(t, f, scenario == "forged-support")
	}
	if scenario == "forged-support" {
		request.Retired.Targets[0].Artifacts[0].Supports[0].Artifact = "forged-original"
	}
	// Act: raw cleanup without a durable dispatch or with changed source supports.
	state, err := f.adapter.Cleanup(t.Context(), request)
	// Assert: no destructive acknowledgement and exact old records still inspect ready.
	if !errors.Is(err, ragy.ErrProtocol) || state != lifecycle.CleanupUnknown {
		t.Fatal("unverified cleanup accepted", scenario, state, err)
	}
	inspected, err := f.adapter.Inspect(t.Context(), lifecycle.StageRequest{Manifest: first, Target: "lexical"})
	if err != nil || inspected.State != lifecycle.TargetReady {
		t.Fatal("old inventory destroyed", inspected, err)
	}
}
func registerLexicalCleanupDispatch(t *testing.T, f *fixture, dispatched bool) {
	t.Helper()
	cleaner, err := lifecycle.NewCleaner(
		lifecycle.CleanerConfig{
			Store:   f.store,
			Now:     func() time.Time { return f.now },
			Targets: []lifecycle.CleanupRegistration{{Name: "lexical", Port: f.adapter}},
			Policy:  lifecycle.CleanupPolicy{Deadline: time.Minute, Backoff: []time.Duration{time.Second}},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = cleaner.Begin(t.Context(), "n", "new"); err != nil {
		t.Fatal(err)
	}
	if !dispatched {
		return
	}
	snapshot, err := f.store.Load(t.Context(), "n")
	if err != nil {
		t.Fatal(err)
	}
	snapshot.Cleanups[0].Items[0].State = lifecycle.CleanupUnknown
	snapshot.Cleanups[0].Items[0].Attempts = 1
	if _, err = f.store.CompareSwap(t.Context(), snapshot.Generation, snapshot); err != nil {
		t.Fatal(err)
	}
}
