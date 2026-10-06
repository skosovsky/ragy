//go:build darwin || linux

package integration_test

import (
	"context"
	"errors"
	"testing"

	"github.com/skosovsky/ragy/lifecycle"
)

func TestActualPrepareAndPublicationCheckpointFailure(t *testing.T) {
	for _, target := range []string{"lexical", "tensor", "graph"} {
		for _, phase := range []string{"prepare", "publish"} {
			for _, after := range []bool{false, true} {
				t.Run(
					target+"/"+phase+"/"+map[bool]string{false: "before", true: "after"}[after],
					func(t *testing.T) { operationCheckpointCase(t, target, phase, after) },
				)
			}
		}
	}
}
func operationCheckpointCase(t *testing.T, target, phase string, after bool) {
	t.Helper()
	// Arrange: immutable old pin and an actual replacement operation.
	f := newFixture(t, target)
	old := sourceBatch("policy", "r1", []string{"p1", "p2"})
	faq := sourceBatch("faq", "r1", []string{"f1"})
	f.ingest(t, plan("policy-r1", "", target, old), old)
	f.ingest(t, plan("faq-r1", "", target, faq), faq)
	captured := f.pin(t)
	newer := sourceBatch("policy", "r2", []string{"p3"})
	manifest := plan("policy-r2", "policy-r1", target, newer)
	if phase == "publish" {
		prepareStages(t, f, manifest, newer)
	}
	port := &countedActualStage{base: densePort{adapter: f.dense}}
	fault := &checkpointFaultStore{base: f.store, failAt: 1, afterCommit: after}
	executor := checkpointExecutor(t, f, fault, "dense", port)
	// Act: lose one durable CAS result at the selected boundary.
	var err error
	if phase == "prepare" {
		_, err = executor.Prepare(t.Context(), manifest)
	} else {
		_, err = executor.Publish(t.Context(), "fixture-a", manifest.ID)
	}
	if !errors.Is(err, lifecycle.ErrOutcomeUnknown) || !errors.Is(err, context.DeadlineExceeded) || fault.calls != 1 ||
		!fault.fired {
		t.Fatal("checkpoint failure hidden", err)
	}
	assertOperationCheckpointTruth(t, f, manifest.ID, phase, after)
	wantRevision, wantCount := "r1", 2
	if phase == "publish" && after {
		wantRevision, wantCount = "r2", 1
	}
	denseDocs, secondaryDocs := f.read(t, f.pin(t), old, newer, faq)
	assertRevision(t, denseDocs, wantRevision, wantCount)
	assertRevision(t, secondaryDocs, wantRevision, wantCount)
	// A fresh executor reconciles Prepare/Publish via Load, never target staging.
	restarted := checkpointExecutor(t, f, fault, "dense", port)
	if phase == "prepare" {
		_, err = restarted.Prepare(t.Context(), manifest)
	} else {
		_, err = restarted.Publish(t.Context(), "fixture-a", manifest.ID)
	}
	expectedCAS := 1
	if !after {
		expectedCAS = 2
	}
	if err != nil || fault.calls != expectedCAS || port.stages != 0 || port.inspections != 0 {
		t.Fatal("operation recovery repeated committed work", err, fault.calls, port)
	}
	if phase == "prepare" {
		for _, name := range []string{"dense", target} {
			if _, err = restarted.Stage(t.Context(), "fixture-a", manifest.ID, name, newer); err != nil {
				t.Fatal(err)
			}
		}
		if _, err = restarted.Publish(t.Context(), "fixture-a", manifest.ID); err != nil {
			t.Fatal(err)
		}
	}
	denseDocs, secondaryDocs = f.read(t, f.pin(t), old, newer, faq)
	assertRevision(t, denseDocs, "r2", 1)
	assertRevision(t, secondaryDocs, "r2", 1)
	denseDocs, secondaryDocs = f.read(t, captured, old, newer, faq)
	assertRevision(t, denseDocs, "r1", 2)
	assertRevision(t, secondaryDocs, "r1", 2)
}
func assertOperationCheckpointTruth(t *testing.T, f *fixture, id, phase string, after bool) {
	t.Helper()
	snapshot, err := f.store.Load(t.Context(), "fixture-a")
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, manifest := range snapshot.Manifests {
		if manifest.ID != id {
			continue
		}
		found = true
		switch phase {
		case "prepare":
			if !after || manifest.State != lifecycle.Planned {
				t.Fatal("prepare inferred durable ownership", manifest.State, after)
			}
		case "publish":
			want := lifecycle.Ready
			if after {
				want = lifecycle.Published
			}
			if manifest.State != want {
				t.Fatal("publication inferred durable commit", manifest.State, want)
			}
		}
	}
	if found != (phase == "publish" || after) {
		t.Fatal("durable operation presence", found, phase, after)
	}
}
