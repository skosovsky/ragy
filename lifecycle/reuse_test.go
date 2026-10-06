//go:build darwin || linux

package lifecycle_test

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/lifecycle"
)

func TestReuseRequiresExactFingerprintsAndCompleteTargetProfile(t *testing.T) {
	// Arrange: durable published identity with actual target inspection ports.
	executor, store, dense, tensor := executorFixture(t)
	manifest := plannedManifest()
	if _, err := executor.Prepare(t.Context(), manifest); err != nil {
		t.Fatal(err)
	}
	for _, target := range []string{"dense", "tensor"} {
		if _, err := executor.Stage(t.Context(), "n", manifest.ID, target, "payload"); err != nil {
			t.Fatal(err)
		}
	}
	published, err := executor.Publish(t.Context(), "n", manifest.ID)
	if err != nil {
		t.Fatal(err)
	}
	before, err := store.Load(t.Context(), "n")
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	decision, err := executor.CheckReuse(t.Context(), published.Identity, []string{"tensor", "dense"})
	// Assert: both targets inspected once, no staging/write repeated.
	if err != nil || !decision.CanSkip() || decision.Publication != manifest.ID || dense.inspections != 1 ||
		tensor.inspections != 1 ||
		dense.calls != 1 ||
		tensor.calls != 1 {
		t.Fatal("unverified reuse or repeated ingestion", decision, err)
	}
	for _, field := range []string{"content", "access", "transformation", "revision"} {
		t.Run(field, func(t *testing.T) {
			desired := published.Identity
			switch field {
			case "content":
				desired.Content = "changed"
			case "access":
				desired.Access = "new-acl"
			case "transformation":
				desired.Transformation = "new-transform"
			case "revision":
				desired.Revision = "r2"
			}
			result, checkErr := executor.CheckReuse(t.Context(), desired, []string{"dense", "tensor"})
			if checkErr != nil || result.CanSkip() || result.Reason != lifecycle.ReuseChanged {
				t.Fatal("changed identity skipped", result, checkErr)
			}
		})
	}
	incomplete, err := executor.CheckReuse(t.Context(), published.Identity, []string{"dense"})
	if err != nil || incomplete.CanSkip() || incomplete.Reason != lifecycle.ReuseIncomplete {
		t.Fatal("partial profile skipped", err)
	}
	after, err := store.Load(t.Context(), "n")
	if err != nil || after.Generation != before.Generation || dense.inspections != 1 || tensor.inspections != 1 {
		t.Fatal("negative decision inspected or mutated", err)
	}
}

type reuseInspector struct {
	inspect func(context.Context, lifecycle.StageRequest) (lifecycle.StageResult, error)
}

func (p reuseInspector) Stage(context.Context, lifecycle.StageRequest, string) (lifecycle.StageResult, error) {
	return lifecycle.StageResult{}, ragy.ErrProtocol
}
func (p reuseInspector) Inspect(ctx context.Context, req lifecycle.StageRequest) (lifecycle.StageResult, error) {
	return p.inspect(ctx, req)
}

func TestReuseFailsClosedOnInspectionUncertaintyAndConcurrentPublication(t *testing.T) {
	for _, profile := range []string{"missing", "unknown", "invalid", "canceled", "concurrent"} {
		t.Run(profile, func(t *testing.T) { reuseFailureCase(t, profile) })
	}
}
func reuseFailureCase(t *testing.T, profile string) {
	t.Helper()
	// Arrange: durable published manifest; inspection can lose data or race its ledger.
	_, store, _, _ := executorFixture(t)
	manifest := manifestFixture()
	snapshot := lifecycle.Snapshot{
		Schema:       lifecycle.SchemaIdentity,
		Namespace:    "n",
		Manifests:    []lifecycle.Manifest{manifest},
		Publications: []lifecycle.Publication{{Source: "policy", Manifest: manifest.ID}},
	}
	if _, err := store.CompareSwap(t.Context(), 0, snapshot); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	calls := 0
	port := reuseInspector{inspect: func(_ context.Context, _ lifecycle.StageRequest) (lifecycle.StageResult, error) {
		calls++
		switch profile {
		case "missing":
			return lifecycle.StageResult{State: lifecycle.TargetPending}, nil
		case "unknown":
			return lifecycle.StageResult{}, context.DeadlineExceeded
		case "invalid":
			return lifecycle.StageResult{State: lifecycle.TargetReady, Revision: "foreign"}, nil
		case "canceled":
			cancel()
		case "concurrent":
			loaded, err := store.Load(t.Context(), "n")
			if err != nil {
				t.Fatal(err)
			}
			if _, err = store.CompareSwap(t.Context(), loaded.Generation, loaded); err != nil {
				t.Fatal(err)
			}
		}
		return lifecycle.StageResult{State: lifecycle.TargetReady, Revision: "r1"}, nil
	}}
	executor, err := lifecycle.NewExecutor(
		lifecycle.ExecutorConfig[string]{
			Store: store,
			Targets: []lifecycle.Registration[string]{
				{Name: "dense", Port: port},
				{Name: "tensor", Port: port},
			},
			ClonePayload:    func(s string) (string, error) { return s, nil },
			ValidatePayload: func(lifecycle.Manifest, string) error { return nil },
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	decision, err := executor.CheckReuse(ctx, manifest.Identity, []string{"dense", "tensor"})
	// Assert: no negative/unknown observation can authorize skipping ingestion.
	if decision.CanSkip() || calls == 0 {
		t.Fatal("unsafe reuse decision", decision, err)
	}
	if profile == "missing" {
		if err != nil || decision.Reason != lifecycle.ReuseIncomplete {
			t.Fatal("missing target masked", err)
		}
		return
	}
	if err == nil {
		t.Fatal("uncertain decision succeeded")
	}
	if profile == "concurrent" && !errors.Is(err, lifecycle.ErrConflict) {
		t.Fatal("generation race ignored", err)
	}
}
