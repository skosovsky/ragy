//go:build darwin || linux

package integration_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/skosovsky/ragy/lifecycle"
)

type publicationInterleavedCleanup struct {
	base        lifecycle.CleanupPort
	publish     func()
	calls       int
	inspections int
}

func (p *publicationInterleavedCleanup) Cleanup(
	ctx context.Context,
	request lifecycle.CleanupRequest,
) (lifecycle.CleanupState, error) {
	p.calls++
	if p.publish != nil {
		publish := p.publish
		p.publish = nil
		publish()
	}
	return p.base.Cleanup(ctx, request)
}

func (p *publicationInterleavedCleanup) InspectCleanup(
	ctx context.Context,
	request lifecycle.CleanupRequest,
) (lifecycle.CleanupState, error) {
	p.inspections++
	return p.base.InspectCleanup(ctx, request)
}

func TestActualCleanupPreservesNewerPublishedReuseOfArtifactIDs(t *testing.T) {
	for _, target := range []string{"lexical", "tensor", "graph"} {
		for _, interleave := range []string{"dense", target} {
			t.Run(
				"dense+"+target+"/interleave-"+interleave,
				func(t *testing.T) { actualNewerCleanupCase(t, target, interleave) },
			)
		}
	}
}
func actualNewerCleanupCase(t *testing.T, target, interleave string) {
	t.Helper()
	// Arrange: old physical data remains behind an acknowledged tombstone and cleanup job.
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
	var base lifecycle.CleanupPort = f.dense
	if interleave != "dense" {
		base = actualSecondaryCleanup(f)
	}
	fault := &publicationInterleavedCleanup{base: base}
	cleaner := cleanupWithFault(t, f, interleave, fault)
	job, err := cleaner.Begin(t.Context(), "fixture-a", deleted.ID)
	if err != nil || len(job.Items) != 2 {
		t.Fatal("captured retirement", job, err)
	}
	newer := sourceBatch("policy", "r3", []string{"p1"})
	prepareStages(t, f, plan("policy-r3", deleted.ID, target, newer), newer)
	fault.publish = func() {
		if _, publishErr := f.executor.Publish(t.Context(), "fixture-a", "policy-r3"); publishErr != nil {
			t.Fatal(publishErr)
		}
	}
	// Act: publish a real replacement after the cleanup request captures its source fence.
	_, err = cleaner.Attempt(t.Context(), "fixture-a", deleted.ID, "policy-r1", interleave, false)
	// Assert: stale expected publication stops the actual target before deletion.
	if !errors.Is(err, lifecycle.ErrConflict) || !errors.Is(err, lifecycle.ErrOutcomeUnknown) {
		t.Fatal("stale cleanup crossed source fence", err)
	}
	denseDocs, secondaryDocs := f.read(t, captured, old, faq)
	assertRevision(t, denseDocs, "r1", 2)
	assertRevision(t, secondaryDocs, "r1", 2)
	restarted := cleanupWithFault(t, f, interleave, fault)
	job, err = restarted.Reconcile(t.Context(), "fixture-a", deleted.ID, "policy-r1", interleave)
	if err != nil || fault.calls != 1 || fault.inspections != 1 {
		t.Fatal("unknown stale cleanup recovery", err)
	}
	assertCleanupItem(t, job, interleave, lifecycle.CleanupWaiting, 1)
	f.now = f.now.Add(time.Second)
	if _, err = restarted.Attempt(t.Context(), "fixture-a", deleted.ID, "policy-r1", interleave, false); err != nil {
		t.Fatal("current source fence cleanup", err)
	}
	assertRetiredTargetUnavailable(t, f, captured, old, interleave)
	completeNewerCleanup(t, f, restarted, deleted.ID, interleave, old, newer, faq)
	if fault.calls != 2 || fault.inspections != 1 {
		t.Fatal("hidden cleanup retry", fault.calls, fault.inspections)
	}
}

func completeNewerCleanup(
	t *testing.T,
	f *fixture,
	cleaner *lifecycle.Cleaner,
	owner, first string,
	old, newer, faq batch,
) {
	t.Helper()
	other := f.target
	if first == f.target {
		other = "dense"
	}
	job, err := cleaner.Attempt(t.Context(), "fixture-a", owner, "policy-r1", other, false)
	if err != nil || !job.Complete {
		t.Fatal("retirement did not complete", job, err)
	}
	// Assert exact revision ownership: reused p1 belongs to r3, unrelated FAQ is intact.
	denseDocs, secondaryDocs := f.read(t, f.pin(t), old, newer, faq)
	assertRevision(t, denseDocs, "r3", 1)
	assertRevision(t, secondaryDocs, "r3", 1)
	snapshot, err := f.store.Load(t.Context(), "fixture-a")
	if err != nil {
		t.Fatal(err)
	}
	for _, publication := range snapshot.Publications {
		if publication.Source == "policy" && publication.Manifest != "policy-r3" {
			t.Fatal("cleanup replaced newer publication", publication)
		}
	}
	for _, item := range job.Items {
		if item.Manifest != "policy-r1" {
			t.Fatal("newer plan added to retirement", item)
		}
	}
}
