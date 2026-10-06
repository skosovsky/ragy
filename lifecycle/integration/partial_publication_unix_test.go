//go:build darwin || linux

package integration_test

import (
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
	densefs "github.com/skosovsky/ragy/dense/persistent"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/retrieval"
)

func TestActualJointExplicitPartialPublicationFreezesMissingTarget(t *testing.T) {
	for _, target := range []string{"lexical", "tensor", "graph"} {
		t.Run("dense+"+target, func(t *testing.T) {
			t.Run("pending", func(t *testing.T) { partialPublicationCase(t, target, false) })
			t.Run("lost-response", func(t *testing.T) { partialPublicationCase(t, target, true) })
		})
	}
}

func partialPublicationCase(t *testing.T, target string, lostResponse bool) {
	t.Helper()
	// Arrange: complete r1 remains readable while only dense r2 is staged.
	f := newFixture(t, target)
	old := sourceBatch("policy", "r1", []string{"p1", "p2"})
	faq := sourceBatch("faq", "r1", []string{"f1"})
	f.ingest(t, plan("policy-r1", "", target, old), old)
	f.ingest(t, plan("faq-r1", "", target, faq), faq)
	oldRead := f.pin(t)
	newer := sourceBatch("policy", "r2", []string{"p3"})
	pending := plan("policy-r2", "policy-r1", target, newer)
	pending.Partial = true
	if _, err := f.executor.Prepare(t.Context(), pending); err != nil {
		t.Fatal(err)
	}
	if _, err := f.executor.Stage(t.Context(), "fixture-a", pending.ID, "dense", newer); err != nil {
		t.Fatal(err)
	}
	missingState := lifecycle.TargetPending
	if lostResponse {
		f.secondary.failAfter = true
		if _, stageErr := f.executor.Stage(
			t.Context(), "fixture-a", pending.ID, target, newer,
		); !errors.Is(stageErr, lifecycle.ErrOutcomeUnknown) {
			t.Fatal("lost response did not retain uncertainty", stageErr)
		}
		missingState = lifecycle.TargetUnknown
	}
	dispatched := f.secondary.calls
	before, err := lifecycle.CapturePublication(t.Context(), f.store, "fixture-a", []string{"dense", target})
	if err != nil || before.Reference() != oldRead.Publication().Reference() {
		t.Fatal("partial staging changed active publication", err)
	}
	// Act: explicitly publish only confirmed ready data with a retained missing target.
	published, err := f.executor.Publish(t.Context(), "fixture-a", pending.ID)
	// Assert: the outcome retains partial state and never invents the absent revision.
	if err != nil || !published.Partial || published.State != lifecycle.Published ||
		published.Targets[0].State != lifecycle.TargetReady || published.Targets[0].Revision != "r2" ||
		published.Targets[1].State != missingState || published.Targets[1].Revision != "" {
		t.Fatal("partial target evidence lost", published, err)
	}
	if _, err = lifecycle.CapturePublication(
		t.Context(), f.store, "fixture-a", []string{"dense", target},
	); !errors.Is(err, ragy.ErrUnsupported) {
		t.Fatal("strict joint read accepted partial publication", err)
	}
	densePublication, err := lifecycle.CapturePublication(t.Context(), f.store, "fixture-a", []string{"dense"})
	if err != nil {
		t.Fatal(err)
	}
	denseResult, err := f.dense.Retrieve(t.Context(), retrieval.Query[densefs.Intent]{
		Read:    f.bind(t, densePublication),
		Intent:  densefs.Intent{Embedding: dense.Embedding{Space: denseSpace(), Vector: []float32{1, 0}}},
		Options: retrieval.RetrieveOptions{TopK: 10},
	})
	if err != nil {
		t.Fatal(err)
	}
	assertRevision(t, denseResult.Documents(), "r2", 1)
	assertPartialCaptureFanout(t, f, newer)
	oldDense, oldSecondary := f.read(t, oldRead, old, faq)
	assertRevision(t, oldDense, "r1", 2)
	assertRevision(t, oldSecondary, "r1", 2)
	replay, err := f.executor.Publish(t.Context(), "fixture-a", pending.ID)
	if err != nil || !replay.Partial || replay.Targets[1].State != missingState || f.secondary.calls != dispatched {
		t.Fatal("publication replay promoted missing target", err)
	}
	loaded, err := f.store.Load(t.Context(), "fixture-a")
	if err != nil {
		t.Fatal(err)
	}
	for _, manifest := range loaded.Manifests {
		if manifest.ID == pending.ID && (!manifest.Partial || manifest.Targets[1].Revision != "") {
			t.Fatal("durable partial state was promoted")
		}
	}
}
