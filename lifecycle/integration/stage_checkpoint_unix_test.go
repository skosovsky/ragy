//go:build darwin || linux

package integration_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/skosovsky/ragy/lifecycle"
)

type checkpointFaultStore struct {
	base        lifecycle.Store
	failAt      int
	afterCommit bool
	calls       int
	fired       bool
}

func (s *checkpointFaultStore) Load(ctx context.Context, namespace string) (lifecycle.Snapshot, error) {
	return s.base.Load(ctx, namespace)
}

func (s *checkpointFaultStore) CompareSwap(
	ctx context.Context,
	generation uint64,
	snapshot lifecycle.Snapshot,
) (lifecycle.Snapshot, error) {
	s.calls++
	if s.calls != s.failAt {
		return s.base.CompareSwap(ctx, generation, snapshot)
	}
	s.fired = true
	if !s.afterCommit {
		return lifecycle.Snapshot{}, context.DeadlineExceeded
	}
	committed, err := s.base.CompareSwap(ctx, generation, snapshot)
	if err != nil {
		return committed, err
	}
	return lifecycle.Snapshot{}, context.DeadlineExceeded
}

type countedActualStage struct {
	base        lifecycle.StagePort[batch]
	stages      int
	inspections int
}

func (p *countedActualStage) Stage(
	ctx context.Context,
	request lifecycle.StageRequest,
	input batch,
) (lifecycle.StageResult, error) {
	p.stages++
	return p.base.Stage(ctx, request, input)
}

func (p *countedActualStage) Inspect(
	ctx context.Context,
	request lifecycle.StageRequest,
) (lifecycle.StageResult, error) {
	p.inspections++
	return p.base.Inspect(ctx, request)
}

func checkpointExecutor(
	t *testing.T,
	f *fixture,
	store lifecycle.Store,
	name string,
	port lifecycle.StagePort[batch],
) *lifecycle.Executor[batch] {
	t.Helper()
	var dense lifecycle.StagePort[batch] = densePort{adapter: f.dense}
	var secondary lifecycle.StagePort[batch] = f.secondary
	if name == "dense" {
		dense = port
	} else {
		secondary = port
	}
	executor, err := lifecycle.NewExecutor(
		lifecycle.ExecutorConfig[batch]{
			Store: store,
			Now:   func() time.Time { return f.now },
			Targets: []lifecycle.Registration[batch]{
				{Name: "dense", Port: dense},
				{Name: f.target, Port: secondary},
			},
			ClonePayload:    cloneBatch,
			ValidatePayload: validateBatch,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return executor
}

func TestActualStageCheckpointFailureBeforeAndAfterDurableCommit(t *testing.T) {
	for _, target := range []string{"lexical", "tensor", "graph"} {
		for _, selected := range []string{"dense", target} {
			for _, phase := range []int{1, 2} {
				for _, after := range []bool{false, true} {
					label := target + "/" + selected + "/" + map[int]string{1: "dispatch", 2: "ready"}[phase] + "/" + map[bool]string{false: "before", true: "after"}[after]
					t.Run(label, func(t *testing.T) { stageCheckpointCase(t, target, selected, phase, after) })
				}
			}
		}
	}
}
func stageCheckpointCase(t *testing.T, target, selected string, phase int, after bool) {
	t.Helper()
	// Arrange: actual published r1 remains available while r2 staging is uncertain.
	f := newFixture(t, target)
	old := sourceBatch("policy", "r1", []string{"p1", "p2"})
	faq := sourceBatch("faq", "r1", []string{"f1"})
	f.ingest(t, plan("policy-r1", "", target, old), old)
	f.ingest(t, plan("faq-r1", "", target, faq), faq)
	captured := f.pin(t)
	newer := sourceBatch("policy", "r2", []string{"p3"})
	manifest := plan("policy-r2", "policy-r1", target, newer)
	if _, err := f.executor.Prepare(t.Context(), manifest); err != nil {
		t.Fatal(err)
	}
	var base lifecycle.StagePort[batch] = densePort{adapter: f.dense}
	if selected != "dense" {
		base = f.secondary
	}
	port := &countedActualStage{base: base}
	fault := &checkpointFaultStore{base: f.store, failAt: phase, afterCommit: after}
	executor := checkpointExecutor(t, f, fault, selected, port)
	// Act: fail one actual durable checkpoint either before or after its CAS effect.
	_, err := executor.Stage(t.Context(), "fixture-a", manifest.ID, selected, newer)
	if !fault.fired || fault.calls != phase || !errors.Is(err, context.DeadlineExceeded) ||
		!errors.Is(err, lifecycle.ErrOutcomeUnknown) {
		t.Fatal("checkpoint fault not surfaced", fault, err)
	}
	want := lifecycle.TargetUnknown
	if phase == 1 && !after {
		want = lifecycle.TargetPending
	}
	if phase == 2 && after {
		want = lifecycle.TargetReady
	}
	assertDurableTargetCheckpoint(t, f, manifest.ID, selected, want)
	expectedWrites := 0
	if phase == 2 {
		expectedWrites = 1
	}
	if port.stages != expectedWrites {
		t.Fatal("target dispatch crossed unknown durable outcome", port.stages, expectedWrites)
	}
	denseDocs, secondaryDocs := f.read(t, f.pin(t), old, newer, faq)
	assertRevision(t, denseDocs, "r1", 2)
	assertRevision(t, secondaryDocs, "r1", 2)
	// A fresh executor loads truth, inspects uncertainty once and never blind-restages.
	restarted := checkpointExecutor(t, f, f.store, selected, port)
	if _, err = restarted.Reconcile(t.Context(), "fixture-a", manifest.ID, selected); err != nil {
		t.Fatal("checkpoint reconciliation", err)
	}
	expectedInspections := 0
	if want == lifecycle.TargetUnknown {
		expectedInspections = 1
	}
	if port.inspections != expectedInspections || port.stages != expectedWrites {
		t.Fatal("reconcile repeated staging", port)
	}
	if _, err = restarted.Stage(t.Context(), "fixture-a", manifest.ID, selected, newer); err != nil {
		t.Fatal("explicit stage continuation", err)
	}
	if port.stages != 1 {
		t.Fatal("stage repeated confirmed backend commit", port.stages)
	}
	other := target
	if selected == target {
		other = "dense"
	}
	if _, err = restarted.Stage(t.Context(), "fixture-a", manifest.ID, other, newer); err != nil {
		t.Fatal(err)
	}
	if _, err = restarted.Publish(t.Context(), "fixture-a", manifest.ID); err != nil {
		t.Fatal(err)
	}
	denseDocs, secondaryDocs = f.read(t, f.pin(t), old, newer, faq)
	assertRevision(t, denseDocs, "r2", 1)
	assertRevision(t, secondaryDocs, "r2", 1)
	denseDocs, secondaryDocs = f.read(t, captured, old, newer, faq)
	assertRevision(t, denseDocs, "r1", 2)
	assertRevision(t, secondaryDocs, "r1", 2)
}
func assertDurableTargetCheckpoint(t *testing.T, f *fixture, id, target string, want lifecycle.TargetState) {
	t.Helper()
	snapshot, err := f.store.Load(t.Context(), "fixture-a")
	if err != nil {
		t.Fatal(err)
	}
	for _, manifest := range snapshot.Manifests {
		if manifest.ID == id {
			for _, item := range manifest.Targets {
				if item.Name == target {
					if item.State != want {
						t.Fatal("inferred checkpoint differs from durable truth", item, want)
					}
					return
				}
			}
		}
	}
	t.Fatal("durable target missing")
}
