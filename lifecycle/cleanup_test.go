//go:build darwin || linux

package lifecycle_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
)

type cleanupHost struct {
	calls        int
	inspected    int
	waiting      bool
	loseResponse bool
	done         bool
}

func (h *cleanupHost) Cleanup(_ context.Context, request lifecycle.CleanupRequest) (lifecycle.CleanupState, error) {
	h.calls++
	if request.Retired.ID != "pub1" || request.Owner.ID != "deleted" || request.ActivePublication != "deleted" {
		return lifecycle.CleanupUnknown, lifecycle.ErrConflict
	}
	if h.waiting {
		return lifecycle.CleanupWaiting, nil
	}
	h.done = true
	if h.loseResponse {
		return lifecycle.CleanupUnknown, context.DeadlineExceeded
	}
	return lifecycle.CleanupDone, nil
}
func (h *cleanupHost) InspectCleanup(_ context.Context, _ lifecycle.CleanupRequest) (lifecycle.CleanupState, error) {
	h.inspected++
	if h.done {
		return lifecycle.CleanupDone, nil
	}
	return lifecycle.CleanupWaiting, nil
}

func cleanupFixture(t *testing.T, now *time.Time, port *cleanupHost) (*lifecycle.Cleaner, *filestore.Store) {
	t.Helper()
	_, store, dense, tensor := executorFixture(t)
	executor, err := lifecycle.NewExecutor(lifecycle.ExecutorConfig[string]{
		Store: store, Now: func() time.Time { return *now },
		Targets:         []lifecycle.Registration[string]{{Name: "dense", Port: dense}, {Name: "tensor", Port: tensor}},
		ClonePayload:    func(p string) (string, error) { return p, nil },
		ValidatePayload: func(_ lifecycle.Manifest, _ string) error { return nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	if _, err = executor.Prepare(ctx, plannedManifest()); err != nil {
		t.Fatal(err)
	}
	for _, target := range []string{"dense", "tensor"} {
		if _, err = executor.Stage(ctx, "n", "pub1", target, "payload"); err != nil {
			t.Fatal(err)
		}
	}
	if _, err = executor.Publish(ctx, "n", "pub1"); err != nil {
		t.Fatal(err)
	}
	tombstone := plannedManifest()
	tombstone.ID, tombstone.Key, tombstone.ExpectedPublication = "deleted", "delete-key", "pub1"
	tombstone.Tombstone = true
	tombstone.Targets = nil
	if _, err = executor.Prepare(ctx, tombstone); err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Publish(ctx, "n", "deleted"); err != nil {
		t.Fatal(err)
	}
	cleaner := newCleaner(t, store, now, port)
	if _, err = cleaner.Begin(ctx, "n", "deleted"); err != nil {
		t.Fatal(err)
	}
	return cleaner, store
}
func newCleaner(t *testing.T, store lifecycle.Store, now *time.Time, port *cleanupHost) *lifecycle.Cleaner {
	t.Helper()
	cleaner, err := lifecycle.NewCleaner(lifecycle.CleanerConfig{
		Store:   store,
		Now:     func() time.Time { return *now },
		Targets: []lifecycle.CleanupRegistration{{Name: "dense", Port: port}, {Name: "tensor", Port: port}},
		Policy: lifecycle.CleanupPolicy{
			Deadline: time.Minute,
			Backoff:  []time.Duration{time.Second, 2 * time.Second, 4 * time.Second},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return cleaner
}

func TestCleanupFakeClockBackoffDeadlineAndExplicitRecovery(t *testing.T) {
	// Arrange: acknowledged tombstone, unavailable cleanup target, real durable state.
	now := time.Unix(100, 0).UTC()
	port := &cleanupHost{waiting: true}
	cleaner, store := cleanupFixture(t, &now, port)
	ctx := context.Background()
	// Act/Assert: one attempt per due call; capped 1/2/4/4 backoff, no hidden retries.
	for _, delay := range []time.Duration{time.Second, 2 * time.Second, 4 * time.Second, 4 * time.Second} {
		job, err := cleaner.Attempt(ctx, "n", "deleted", "pub1", "dense", false)
		if err != nil || job.Items[0].NextAt.Sub(now) != delay {
			t.Fatal("wrong cleanup backoff", err)
		}
		calls := port.calls
		if _, err = cleaner.Attempt(
			ctx,
			"n",
			"deleted",
			"pub1",
			"dense",
			false,
		); !errors.Is(err, lifecycle.ErrCleanupNotDue) ||
			port.calls != calls {
			t.Fatal("not-due cleanup dispatched")
		}
		now = now.Add(delay)
	}
	now = time.Unix(160, 0).UTC()
	calls := port.calls
	overdue, err := cleaner.Attempt(ctx, "n", "deleted", "pub1", "dense", false)
	if !errors.Is(err, lifecycle.ErrCleanupOverdue) || !overdue.Overdue || port.calls != calls {
		t.Fatal("deadline did not stop cleanup")
	}
	restarted := newCleaner(t, store, &now, port)
	job, err := restarted.Begin(ctx, "n", "deleted")
	if err != nil || !job.Overdue || job.Deadline != time.Unix(160, 0).UTC() {
		t.Fatal("restart reset cleanup deadline", err)
	}
	port.waiting = false
	for _, target := range []string{"dense", "tensor"} {
		if _, err = restarted.Attempt(ctx, "n", "deleted", "pub1", target, true); err != nil {
			t.Fatal(err)
		}
	}
	snapshot, err := store.Load(ctx, "n")
	if err != nil || !snapshot.Cleanups[0].Complete || !snapshot.Manifests[1].Tombstone ||
		snapshot.Publications[0].Manifest != "deleted" || snapshot.Manifests[1].State != lifecycle.Complete {
		t.Fatal("cleanup completion reopened source or lost state", err)
	}
}

func TestUnknownCleanupInspectedAfterRestartWithoutDeletingAgain(t *testing.T) {
	// Arrange: cleanup commits but loses its response.
	now := time.Unix(100, 0).UTC()
	port := &cleanupHost{loseResponse: true}
	cleaner, store := cleanupFixture(t, &now, port)
	ctx := context.Background()
	// Act.
	_, err := cleaner.Attempt(ctx, "n", "deleted", "pub1", "dense", false)
	// Assert.
	if !errors.Is(err, lifecycle.ErrOutcomeUnknown) || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal("cleanup inferred rollback", err)
	}
	restarted := newCleaner(t, store, &now, port)
	if _, err = restarted.Attempt(
		ctx,
		"n",
		"deleted",
		"pub1",
		"dense",
		false,
	); !errors.Is(err, lifecycle.ErrOutcomeUnknown) ||
		port.calls != 1 {
		t.Fatal("cleanup repeated uncertain deletion")
	}
	job, err := restarted.Reconcile(ctx, "n", "deleted", "pub1", "dense")
	if err != nil || job.Items[0].State != lifecycle.CleanupDone || port.calls != 1 || port.inspected != 1 {
		t.Fatal("cleanup inspection did not confirm actual outcome", err)
	}
}

func TestCleanupCapturesAbandonedPlansButExcludesNewerWork(t *testing.T) {
	// Arrange: simulate crash after tombstone publication before Begin.
	now := time.Unix(100, 0).UTC()
	port := &cleanupHost{}
	_, store := cleanupFixture(t, &now, port)
	ctx := context.Background()
	snapshot, err := store.Load(ctx, "n")
	if err != nil {
		t.Fatal(err)
	}
	snapshot.Cleanups = nil
	snapshot.Manifests[1].State = lifecycle.Published
	stale := plannedManifest()
	stale.ID, stale.Key, stale.ExpectedPublication = "abandoned", "abandoned-key", "pub1"
	newer := plannedManifest()
	newer.ID, newer.Key, newer.ExpectedPublication = "new-plan", "new-key", "deleted"
	// Each operation owns distinct artifacts even when source supports are shared.
	for i := range stale.Targets {
		for j := range stale.Targets[i].Artifacts {
			stale.Targets[i].Artifacts[j].Reference.Artifact += "-abandoned"
		}
	}
	for i := range newer.Targets {
		for j := range newer.Targets[i].Artifacts {
			newer.Targets[i].Artifacts[j].Reference.Artifact += "-future"
		}
	}
	snapshot.Manifests = append(snapshot.Manifests, stale, newer)
	// Build an independent pre-Begin crash fixture; committed cleanup fences cannot be erased by CAS.
	store, err = filestore.New(t.TempDir(), 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	snapshot.Generation = 0
	if _, err = store.CompareSwap(ctx, snapshot.Generation, snapshot); err != nil {
		t.Fatal(err)
	}
	cleaner := newCleaner(t, store, &now, port)
	// Act.
	job, err := cleaner.Begin(ctx, "n", "deleted")
	// Assert: old actual manifest plus stale prepared plan, never the newer plan.
	if err != nil || len(job.Items) != 4 {
		t.Fatal("known abandoned plan omitted", err)
	}
	for _, item := range job.Items {
		if item.Manifest == "new-plan" {
			t.Fatal("newer artifacts retired")
		}
	}
	loaded, err := store.Load(ctx, "n")
	if err != nil {
		t.Fatal(err)
	}
	loaded.Cleanups[0].Items[0].Manifest = "new-plan"
	if loaded.Validate() == nil {
		t.Fatal("future plan accepted as retired inventory")
	}
}
