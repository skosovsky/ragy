//go:build darwin || linux

package integration_test

import (
	"context"
	"errors"
	"testing"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/lifecycle"
)

type cleanupCheckpointFixture struct {
	f        *fixture
	old      batch
	faq      batch
	captured access.Binding
}

func prepareCleanupCheckpoint(t *testing.T, target string) cleanupCheckpointFixture {
	t.Helper()
	f := newFixture(t, target)
	old := sourceBatch("policy", "r1", []string{"p1", "p2"})
	faq := sourceBatch("faq", "r1", []string{"f1"})
	f.ingest(t, plan("policy-r1", "", target, old), old)
	f.ingest(t, plan("faq-r1", "", target, faq), faq)
	captured := f.pin(t)
	deleted := plan("deleted", "policy-r1", target, old)
	deleted.Identity.Revision = "r2"
	deleted.Tombstone, deleted.Targets = true, nil
	if _, err := f.executor.Prepare(t.Context(), deleted); err != nil {
		t.Fatal(err)
	}
	if _, err := f.executor.Publish(t.Context(), "fixture-a", deleted.ID); err != nil {
		t.Fatal(err)
	}
	return cleanupCheckpointFixture{f: f, old: old, faq: faq, captured: captured}
}
func TestActualCleanupCheckpointFailureBeforeAndAfterCommit(t *testing.T) {
	for _, target := range []string{"lexical", "tensor", "graph"} {
		for _, selected := range []string{"dense", target} {
			for _, phase := range []string{"begin", "dispatch", "complete", "complete-job"} {
				for _, after := range []bool{false, true} {
					t.Run(
						target+"/"+selected+"/"+phase+"/"+map[bool]string{false: "before", true: "after"}[after],
						func(t *testing.T) { cleanupCheckpointCase(t, target, selected, phase, after) },
					)
				}
			}
		}
	}
}
func cleanupCheckpointCase(t *testing.T, target, selected, phase string, after bool) {
	t.Helper()
	// Arrange: real old files/records remain behind a durable tombstone read barrier.
	setup := prepareCleanupCheckpoint(t, target)
	f := setup.f
	initializeCleanupCheckpoint(t, f, selected, phase)
	completion := phase == "complete" || phase == "complete-job"
	var base lifecycle.CleanupPort = f.dense
	if selected != "dense" {
		base = actualSecondaryCleanup(f)
	}
	port := &publicationInterleavedCleanup{base: base}
	failAt := 1
	if completion {
		failAt = 2
	}
	fault := &checkpointFaultStore{base: f.store, failAt: failAt, afterCommit: after}
	cleaner := checkpointCleaner(t, f, fault, selected, port)
	// Act: lose the selected job/dispatch/completion checkpoint acknowledgement.
	var err error
	if phase == "begin" {
		_, err = cleaner.Begin(t.Context(), "fixture-a", "deleted")
	} else {
		_, err = cleaner.Attempt(t.Context(), "fixture-a", "deleted", "policy-r1", selected, false)
	}
	if !errors.Is(err, lifecycle.ErrOutcomeUnknown) || !errors.Is(err, context.DeadlineExceeded) || !fault.fired ||
		fault.calls != failAt {
		t.Fatal("cleanup checkpoint fault hidden", fault, err)
	}
	want := lifecycle.CleanupWaiting
	if phase == "dispatch" && after || completion && !after {
		want = lifecycle.CleanupUnknown
	}
	if completion && after {
		want = lifecycle.CleanupDone
	}
	assertCleanupDurableTruth(t, f, selected, phase != "begin" || after, want)
	writes := 0
	if completion {
		writes = 1
	}
	assertCleanupCheckpointEffect(t, setup, port, selected, writes)
	resumeCleanupCheckpoint(t, setup, fault, port, selected, phase, after, want, writes)
}

func assertCleanupCheckpointEffect(
	t *testing.T,
	setup cleanupCheckpointFixture,
	port *publicationInterleavedCleanup,
	selected string,
	writes int,
) {
	t.Helper()
	if port.calls != writes || port.inspections != 0 {
		t.Fatal("destructive dispatch despite checkpoint failure", port)
	}
	if writes != 0 {
		assertRetiredTargetUnavailable(t, setup.f, setup.captured, setup.old, selected)
		return
	}
	denseDocs, secondaryDocs := setup.f.read(t, setup.captured, setup.old, setup.faq)
	assertRevision(t, denseDocs, "r1", 2)
	assertRevision(t, secondaryDocs, "r1", 2)
}

func resumeCleanupCheckpoint(
	t *testing.T,
	setup cleanupCheckpointFixture,
	fault *checkpointFaultStore,
	port *publicationInterleavedCleanup,
	selected, phase string,
	after bool,
	want lifecycle.CleanupState,
	writes int,
) {
	t.Helper()
	f := setup.f
	restarted := checkpointCleaner(t, f, fault, selected, port)
	inspections := 0
	if phase == "begin" {
		replayCleanupBegin(t, restarted, fault, after)
	} else {
		inspections = reconcileCleanupCheckpoint(t, f, restarted, port, selected, want, writes)
	}
	if _, err := restarted.Attempt(t.Context(), "fixture-a", "deleted", "policy-r1", selected, false); err != nil {
		t.Fatal("explicit cleanup continuation", err)
	}
	if port.calls != 1 || port.inspections != inspections {
		t.Fatal("confirmed deletion repeated", port)
	}
	assertRetiredTargetUnavailable(t, f, setup.captured, setup.old, selected)
	finishCleanupCheckpoint(t, setup, restarted, selected)
}
func replayCleanupBegin(t *testing.T, cleaner *lifecycle.Cleaner, fault *checkpointFaultStore, after bool) {
	t.Helper()
	if _, err := cleaner.Begin(t.Context(), "fixture-a", "deleted"); err != nil {
		t.Fatal(err)
	}
	expectedCAS := 1
	if !after {
		expectedCAS = 2
	}
	if fault.calls != expectedCAS {
		t.Fatal("begin repeated committed ownership", fault.calls)
	}
}

func reconcileCleanupCheckpoint(
	t *testing.T,
	f *fixture,
	cleaner *lifecycle.Cleaner,
	port *publicationInterleavedCleanup,
	selected string,
	want lifecycle.CleanupState,
	writes int,
) int {
	t.Helper()
	job, err := cleaner.Reconcile(t.Context(), "fixture-a", "deleted", "policy-r1", selected)
	if err != nil {
		t.Fatal("cleanup checkpoint reconcile", err)
	}
	inspections := 0
	if want == lifecycle.CleanupUnknown {
		inspections = 1
	}
	if port.inspections != inspections || port.calls != writes {
		t.Fatal("inspection repeated deletion", port)
	}
	item := retiredCleanupItem(t, job, selected)
	if item.NextAt.After(f.now) {
		f.now = item.NextAt
	}
	return inspections
}

func assertCleanupDurableTruth(t *testing.T, f *fixture, target string, present bool, want lifecycle.CleanupState) {
	t.Helper()
	snapshot, err := f.store.Load(t.Context(), "fixture-a")
	if err != nil {
		t.Fatal(err)
	}
	if !present {
		if len(snapshot.Cleanups) != 0 {
			t.Fatal("uncommitted cleanup inferred durable")
		}
		return
	}
	if len(snapshot.Cleanups) != 1 {
		t.Fatal("durable cleanup missing")
	}
	item := retiredCleanupItem(t, snapshot.Cleanups[0], target)
	if item.State != want {
		t.Fatal("candidate checkpoint differs from durable truth", item, want)
	}
}
func finishCleanupCheckpoint(t *testing.T, setup cleanupCheckpointFixture, cleaner *lifecycle.Cleaner, first string) {
	t.Helper()
	f := setup.f
	other := f.target
	if first == f.target {
		other = "dense"
	}
	job, err := cleaner.Attempt(t.Context(), "fixture-a", "deleted", "policy-r1", other, false)
	if err != nil || !job.Complete {
		t.Fatal("joint cleanup checkpoint completion", job, err)
	}
	denseDocs, secondaryDocs := f.read(t, f.pin(t), setup.old, setup.faq)
	assertRevision(t, denseDocs, "r1", 0)
	assertRevision(t, secondaryDocs, "r1", 0)
}

func initializeCleanupCheckpoint(t *testing.T, f *fixture, selected, phase string) {
	t.Helper()
	if phase == "begin" {
		return
	}
	cleaner := newCleaner(t, f)
	if _, err := cleaner.Begin(t.Context(), "fixture-a", "deleted"); err != nil {
		t.Fatal(err)
	}
	if phase != "complete-job" {
		return
	}
	other := f.target
	if selected == f.target {
		other = "dense"
	}
	if _, err := cleaner.Attempt(t.Context(), "fixture-a", "deleted", "policy-r1", other, false); err != nil {
		t.Fatal(err)
	}
}
