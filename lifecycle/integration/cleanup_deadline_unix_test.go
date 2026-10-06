//go:build darwin || linux

package integration_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/skosovsky/ragy/lifecycle"
)

type unavailableCleanup struct {
	base     lifecycle.CleanupPort
	restored bool
	calls    int
}

func (p *unavailableCleanup) Cleanup(
	ctx context.Context,
	request lifecycle.CleanupRequest,
) (lifecycle.CleanupState, error) {
	p.calls++
	if !p.restored {
		return lifecycle.CleanupWaiting, nil
	}
	return p.base.Cleanup(ctx, request)
}

func (p *unavailableCleanup) InspectCleanup(
	ctx context.Context,
	request lifecycle.CleanupRequest,
) (lifecycle.CleanupState, error) {
	return p.base.InspectCleanup(ctx, request)
}

func TestActualCleanupFakeClockDeadlineAndExplicitRecovery(t *testing.T) {
	for _, target := range []string{"lexical", "tensor", "graph"} {
		for _, waiting := range []string{"dense", target} {
			t.Run("dense+"+target+"/waiting-"+waiting, func(t *testing.T) { actualCleanupDeadline(t, target, waiting) })
		}
	}
}
func actualCleanupDeadline(t *testing.T, target, waiting string) {
	t.Helper()
	// Arrange: acknowledgement hides the source while physical inventory stays intact.
	f := newFixture(t, target)
	old := sourceBatch("policy", "r1", []string{"p1", "p2"})
	faq := sourceBatch("faq", "r1", []string{"f1"})
	f.ingest(t, plan("policy-r1", "", target, old), old)
	f.ingest(t, plan("faq-r1", "", target, faq), faq)
	captured := f.pin(t)
	tombstone := plan("deleted", "policy-r1", target, old)
	tombstone.Identity.Revision = "r2"
	tombstone.Tombstone, tombstone.Targets = true, nil
	if _, err := f.executor.Prepare(t.Context(), tombstone); err != nil {
		t.Fatal(err)
	}
	if _, err := f.executor.Publish(t.Context(), "fixture-a", tombstone.ID); err != nil {
		t.Fatal(err)
	}
	var base lifecycle.CleanupPort = f.dense
	if waiting != "dense" {
		base = actualSecondaryCleanup(f)
	}
	outage := &unavailableCleanup{base: base}
	cleaner := cleanupWithFault(t, f, waiting, outage)
	initial, err := cleaner.Begin(t.Context(), "fixture-a", tombstone.ID)
	if err != nil {
		t.Fatal(err)
	}
	if initial.Deadline.Sub(initial.StartedAt) != time.Minute {
		t.Fatal("deadline not based on acknowledgment", initial)
	}
	// Act: each new Cleaner reloads the real durable queue; host advances fake time.
	job := driveActualCleanupBackoff(t, f, waiting, outage, initial.Deadline)
	// Assert: overdue is durable and ordinary calls never dispatch beyond the deadline.
	if !job.Overdue || job.Complete {
		t.Fatal("deadline inferred completion", job)
	}
	denseDocs, secondaryDocs := f.read(t, f.pin(t), old, faq)
	assertRevision(t, denseDocs, "r1", 0)
	assertRevision(t, secondaryDocs, "r1", 0)
	// The original access token expired during the fake-clock outage. The host
	// explicitly authorizes a fresh gate for this retained publication snapshot.
	captured = f.bind(t, captured.Publication())
	denseDocs, secondaryDocs = f.read(t, captured, old, faq)
	assertRevision(t, denseDocs, "r1", 2)
	assertRevision(t, secondaryDocs, "r1", 2)
	// Recovery keeps the confirmed backoff; the host advances to its due time.
	if nextAt := retiredCleanupItem(t, job, waiting).NextAt; nextAt.After(f.now) {
		f.now = nextAt
	}
	outage.restored = true
	recovery := cleanupWithFault(t, f, waiting, outage)
	if _, err = recovery.Attempt(t.Context(), "fixture-a", tombstone.ID, "policy-r1", waiting, true); err != nil {
		t.Fatal("explicit recovery failed", err)
	}
	assertRetiredTargetUnavailable(t, f, captured, old, waiting)
	other := target
	if waiting == target {
		other = "dense"
	}
	job, err = recovery.Attempt(t.Context(), "fixture-a", tombstone.ID, "policy-r1", other, true)
	if err != nil || !job.Complete {
		t.Fatal("recovery did not finish joint job", err)
	}
	denseDocs, secondaryDocs = f.read(t, f.pin(t), old, faq)
	assertRevision(t, denseDocs, "r1", 0)
	assertRevision(t, secondaryDocs, "r1", 0)
}

func driveActualCleanupBackoff(
	t *testing.T,
	f *fixture,
	target string,
	outage *unavailableCleanup,
	deadline time.Time,
) lifecycle.CleanupJob {
	t.Helper()
	backoff := []time.Duration{time.Second, 2 * time.Second, 4 * time.Second}
	attempt := 0
	for f.now.Before(deadline) {
		cleaner := cleanupWithFault(t, f, target, outage)
		job, err := cleaner.Attempt(t.Context(), "fixture-a", "deleted", "policy-r1", target, false)
		if err != nil {
			t.Fatal("due attempt", err)
		}
		attempt++
		item := retiredCleanupItem(t, job, target)
		delay := backoff[min(attempt-1, len(backoff)-1)]
		if item.State != lifecycle.CleanupWaiting || item.Attempts != uint64(attempt) || outage.calls != attempt ||
			item.NextAt.Sub(f.now) != delay {
			t.Fatal("backoff checkpoint", item, outage.calls)
		}
		// A second call at the same clock cannot dispatch or increment attempts.
		_, err = cleaner.Attempt(t.Context(), "fixture-a", "deleted", "policy-r1", target, false)
		if !errors.Is(err, lifecycle.ErrCleanupNotDue) || outage.calls != attempt {
			t.Fatal("hidden retry", err)
		}
		f.now = item.NextAt
		if f.now.After(deadline) {
			f.now = deadline
		}
	}
	restarted := cleanupWithFault(t, f, target, outage)
	job, err := restarted.Attempt(t.Context(), "fixture-a", "deleted", "policy-r1", target, false)
	if !errors.Is(err, lifecycle.ErrCleanupOverdue) || outage.calls != attempt {
		t.Fatal("overdue target dispatched", err)
	}
	return job
}
func retiredCleanupItem(t *testing.T, job lifecycle.CleanupJob, target string) lifecycle.RetiredTarget {
	t.Helper()
	for _, item := range job.Items {
		if item.Target == target && item.Manifest == "policy-r1" {
			return item
		}
	}
	t.Fatal("retired target missing")
	return lifecycle.RetiredTarget{}
}
