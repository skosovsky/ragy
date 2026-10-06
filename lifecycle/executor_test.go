//go:build darwin || linux

package lifecycle_test

import (
	"context"
	"errors"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
)

type stageHost struct {
	calls       int
	inspections int
	fail        bool
	revision    string
}

func (h *stageHost) Stage(_ context.Context, request lifecycle.StageRequest, _ string) (lifecycle.StageResult, error) {
	h.calls++
	h.revision = request.Manifest.Identity.Revision
	request.Manifest.Targets[0].Name = "mutated-host-request"
	if h.fail {
		return lifecycle.StageResult{}, context.DeadlineExceeded
	}
	return lifecycle.StageResult{State: lifecycle.TargetReady, Revision: h.revision}, nil
}
func (h *stageHost) Inspect(_ context.Context, _ lifecycle.StageRequest) (lifecycle.StageResult, error) {
	h.inspections++
	return lifecycle.StageResult{State: lifecycle.TargetReady, Revision: h.revision}, nil
}

func executorFixture(t *testing.T) (*lifecycle.Executor[string], *filestore.Store, *stageHost, *stageHost) {
	t.Helper()
	store, err := filestore.New(t.TempDir(), 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	dense, tensor := &stageHost{}, &stageHost{}
	executor, err := lifecycle.NewExecutor(lifecycle.ExecutorConfig[string]{
		Store:        store,
		Targets:      []lifecycle.Registration[string]{{Name: "dense", Port: dense}, {Name: "tensor", Port: tensor}},
		ClonePayload: func(payload string) (string, error) { return payload, nil },
		ValidatePayload: func(m lifecycle.Manifest, payload string) error {
			if payload != m.Payload {
				return ragy.ErrInvalidArgument
			}
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return executor, store, dense, tensor
}
func plannedManifest() lifecycle.Manifest {
	manifest := manifestFixture()
	manifest.State = lifecycle.Planned
	manifest.PublishedAt = time.Time{}
	for i := range manifest.Targets {
		manifest.Targets[i].State = lifecycle.TargetPending
		manifest.Targets[i].Revision = ""
	}
	return manifest
}

func TestExecutorUnknownStageReconcilesWithoutBlindRetry(t *testing.T) {
	// Arrange: real durable manifest store; target can commit then lose its response.
	executor, store, dense, tensor := executorFixture(t)
	ctx := context.Background()
	plan := plannedManifest()
	if _, err := executor.Prepare(ctx, plan); err != nil {
		t.Fatal(err)
	}
	if _, err := executor.Stage(ctx, "n", "pub1", "dense", "payload"); err != nil {
		t.Fatal(err)
	}
	// Act/Assert: unready tensor prevents default publication.
	if _, err := executor.Publish(ctx, "n", "pub1"); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatal("unready joint source published")
	}
	tensor.fail = true
	unknown, err := executor.Stage(ctx, "n", "pub1", "tensor", "payload")
	if !errors.Is(err, lifecycle.ErrOutcomeUnknown) || !errors.Is(err, context.DeadlineExceeded) ||
		unknown.Checkpoint != lifecycle.Staging {
		t.Fatal("uncertain dispatch lost cause or checkpoint", err)
	}
	if _, err = executor.Stage(
		ctx,
		"n",
		"pub1",
		"tensor",
		"payload",
	); !errors.Is(err, lifecycle.ErrOutcomeUnknown) ||
		tensor.calls != 1 {
		t.Fatal("unknown stage retried blindly")
	}
	if _, err = executor.Reconcile(ctx, "n", "pub1", "tensor"); err != nil {
		t.Fatal(err)
	}
	if tensor.inspections != 1 || dense.calls != 1 {
		t.Fatal("reconcile repeated target writes")
	}
	before, err := store.Load(ctx, "n")
	if err != nil {
		t.Fatal(err)
	}
	if len(before.Publications) != 0 {
		t.Fatal("staging leaked into publication")
	}
	if _, err = executor.Publish(ctx, "n", "pub1"); err != nil {
		t.Fatal(err)
	}
	after, err := store.Load(ctx, "n")
	if err != nil || after.Publications[0].Manifest != "pub1" || after.Manifests[0].Targets[0].Name != "dense" {
		t.Fatal("publication lost identity or aliased target request", err)
	}
	if len(before.Publications) != 0 {
		t.Fatal("published swap mutated prior snapshot")
	}
	if _, err = executor.Stage(ctx, "n", "pub1", "dense", "payload"); !errors.Is(err, lifecycle.ErrConflict) {
		t.Fatal("published manifest mutated")
	}
}

func TestPrepareIdempotencyConflictsAndReadyPayloadValidation(t *testing.T) {
	// Arrange.
	executor, store, dense, _ := executorFixture(t)
	ctx := context.Background()
	plan := plannedManifest()
	if _, err := executor.Prepare(ctx, plan); err != nil {
		t.Fatal(err)
	}
	first, err := store.Load(ctx, "n")
	if err != nil {
		t.Fatal(err)
	}
	// Act/Assert.
	if _, err = executor.Prepare(ctx, plannedManifest()); err != nil {
		t.Fatal(err)
	}
	second, err := store.Load(ctx, "n")
	if err != nil || second.Generation != first.Generation {
		t.Fatal("idempotent prepare wrote again")
	}
	changed := plannedManifest()
	changed.Payload = "different"
	if _, err = executor.Prepare(ctx, changed); !errors.Is(err, lifecycle.ErrIdempotencyConflict) {
		t.Fatal("same key changed payload accepted")
	}
	if _, err = executor.Stage(ctx, "n", "pub1", "dense", "payload"); err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Stage(
		ctx,
		"n",
		"pub1",
		"dense",
		"wrong",
	); !errors.Is(err, ragy.ErrInvalidArgument) ||
		dense.calls != 1 {
		t.Fatal("ready target bypassed payload fingerprint validation")
	}
}

func TestExpectedPublicationPreventsStaleSourceWriter(t *testing.T) {
	// Arrange: competing source operations were both prepared against absence.
	executor, store, _, _ := executorFixture(t)
	ctx := context.Background()
	first := plannedManifest()
	stale := plannedManifest()
	stale.ID, stale.Key = "pub2", "key2"
	stale.Identity.Revision = "r2"
	for i := range stale.Targets {
		for j := range stale.Targets[i].Artifacts {
			stale.Targets[i].Artifacts[j].Reference.Revision = "r2"
		}
	}
	if _, err := executor.Prepare(ctx, first); err != nil {
		t.Fatal(err)
	}
	if _, err := executor.Prepare(ctx, stale); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"pub1", "pub2"} {
		for _, target := range []string{"dense", "tensor"} {
			if _, err := executor.Stage(ctx, "n", id, target, "payload"); err != nil {
				t.Fatal(err)
			}
		}
	}
	// Act.
	if _, err := executor.Publish(ctx, "n", "pub1"); err != nil {
		t.Fatal(err)
	}
	_, err := executor.Publish(ctx, "n", "pub2")
	// Assert.
	if !errors.Is(err, lifecycle.ErrConflict) {
		t.Fatal("stale source writer won")
	}
	snapshot, err := store.Load(ctx, "n")
	if err != nil || snapshot.Publications[0].Manifest != "pub1" {
		t.Fatal("source pointer changed", err)
	}
}

type uncertainStore struct {
	lifecycle.Store

	failNext bool
}

func (s *uncertainStore) CompareSwap(
	ctx context.Context,
	expected uint64,
	next lifecycle.Snapshot,
) (lifecycle.Snapshot, error) {
	committed, err := s.Store.CompareSwap(ctx, expected, next)
	if err == nil && s.failNext {
		s.failNext = false
		return lifecycle.Snapshot{}, context.DeadlineExceeded
	}
	return committed, err
}
func TestPublishUnknownResponseReconcilesDurableCommit(t *testing.T) {
	// Arrange: publication CAS succeeds but its response is lost.
	_, durable, dense, tensor := executorFixture(t)
	wrapped := &uncertainStore{Store: durable}
	executor, err := lifecycle.NewExecutor(lifecycle.ExecutorConfig[string]{
		Store:           wrapped,
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
	wrapped.failNext = true
	// Act.
	_, err = executor.Publish(ctx, "n", "pub1")
	// Assert: uncertainty is preserved, and retry observes durable completion.
	if !errors.Is(err, lifecycle.ErrOutcomeUnknown) || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal("commit response inferred rollback", err)
	}
	committed, err := durable.Load(ctx, "n")
	if err != nil || committed.Publications[0].Manifest != "pub1" {
		t.Fatal("unknown commit not durable", err)
	}
	if _, err = executor.Publish(ctx, "n", "pub1"); err != nil {
		t.Fatal("durable replay did not reconcile", err)
	}
	replay, err := durable.Load(ctx, "n")
	if err != nil || replay.Generation != committed.Generation {
		t.Fatal("replayed publication performed another write", err)
	}
}
