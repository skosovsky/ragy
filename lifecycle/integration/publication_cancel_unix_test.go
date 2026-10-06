//go:build darwin || linux

package integration_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/skosovsky/ragy/lifecycle"
)

type cancelCommittedStore struct {
	base   lifecycle.Store
	cancel context.CancelFunc
	writes int
}

func (s *cancelCommittedStore) Load(ctx context.Context, namespace string) (lifecycle.Snapshot, error) {
	return s.base.Load(ctx, namespace)
}
func (s *cancelCommittedStore) CompareSwap(
	ctx context.Context, generation uint64, snapshot lifecycle.Snapshot,
) (lifecycle.Snapshot, error) {
	committed, err := s.base.CompareSwap(ctx, generation, snapshot)
	if err == nil {
		s.writes++
		s.cancel()
	}
	return committed, err
}

func TestActualJointCancellationAfterPublicationNeverRollsBack(t *testing.T) {
	for _, target := range []string{"lexical", "tensor", "graph"} {
		t.Run("dense+"+target, func(t *testing.T) { cancellationAfterPublicationCase(t, target) })
	}
}

func cancellationAfterPublicationCase(t *testing.T, target string) {
	t.Helper()
	// Arrange: actual durable r1 plus a complete staged replacement, with live access.
	f := newFixture(t, target)
	old := sourceBatch("policy", "r1", []string{"p1", "p2"})
	faq := sourceBatch("faq", "r1", []string{"f1"})
	f.ingest(t, plan("policy-r1", "", target, old), old)
	f.ingest(t, plan("faq-r1", "", target, faq), faq)
	oldRead := f.pin(t)
	newer := sourceBatch("policy", "r2", []string{"p3"})
	prepareStages(t, f, plan("policy-r2", "policy-r1", target, newer), newer)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	fault := &cancelCommittedStore{base: f.store, cancel: cancel, writes: 0}
	executor, err := lifecycle.NewExecutor(lifecycle.ExecutorConfig[batch]{
		Store: fault,
		Targets: []lifecycle.Registration[batch]{
			{Name: "dense", Port: densePort{adapter: f.dense}},
			{Name: target, Port: f.secondary},
		},
		ClonePayload: cloneBatch, ValidatePayload: validateBatch,
		Now: func() time.Time { return f.now },
	})
	if err != nil {
		t.Fatal(err)
	}
	// Act: cancel after the underlying filestore has actually committed publication.
	published, err := executor.Publish(ctx, "fixture-a", "policy-r2")
	// Assert: uncertain acknowledgment preserves both cancellation and committed fact.
	if !errors.Is(err, context.Canceled) || !errors.Is(err, lifecycle.ErrOutcomeUnknown) ||
		published.State != lifecycle.Published || published.PublishedAt.IsZero() || fault.writes != 1 {
		t.Fatal("committed cancellation became rollback or success", published, err)
	}
	newDense, newSecondary := f.read(t, f.pin(t), old, newer, faq)
	assertRevision(t, newDense, "r2", 1)
	assertRevision(t, newSecondary, "r2", 1)
	oldDense, oldSecondary := f.read(t, oldRead, old, newer, faq)
	assertRevision(t, oldDense, "r1", 2)
	assertRevision(t, oldSecondary, "r1", 2)
	// Act: fresh context reconciles the known commit without staging or another CAS.
	replayed, err := executor.Publish(t.Context(), "fixture-a", "policy-r2")
	// Assert.
	if err != nil || replayed.State != lifecycle.Published || fault.writes != 1 {
		t.Fatal("reconciliation repeated committed mutation", replayed, err)
	}
}
