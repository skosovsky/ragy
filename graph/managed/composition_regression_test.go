package managed_test

import (
	"context"
	"errors"
	"slices"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/graph/managed"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/source"
)

func TestReleasedHostBasisCannotBeRegisteredAgain(t *testing.T) {
	// Arrange: capture a reader of an explicitly identified host foundation.
	f := newFixture(t)
	snapshot := hostSnapshot(payload("policy", "r1"))
	if err := f.adapter.SetHostBasis(t.Context(), "foundation", snapshot); err != nil {
		t.Fatal(err)
	}
	req := request(f.pin(t))
	req.HostBasis = "foundation"
	if _, err := f.adapter.Traverse(t.Context(), req); err != nil {
		t.Fatal(err)
	}
	// Act.
	if err := f.adapter.ReleaseHostBasis(t.Context(), "foundation"); err != nil {
		t.Fatal(err)
	}
	identicalErr := f.adapter.SetHostBasis(t.Context(), "foundation", snapshot)
	snapshot.Nodes[0].Content = "new facts"
	changedErr := f.adapter.SetHostBasis(t.Context(), "foundation", snapshot)
	out, readErr := f.adapter.Traverse(t.Context(), req)
	// Assert: neither identical nor changed registration resurrects the retired ID.
	if !errors.Is(identicalErr, lifecycle.ErrConflict) || !errors.Is(changedErr, lifecycle.ErrConflict) ||
		!errors.Is(readErr, ragy.ErrUnavailable) ||
		len(out.Snapshot.Nodes) != 0 {
		t.Fatalf("retirement: %v %v %v %#v", identicalErr, changedErr, readErr, out)
	}
	if err := f.adapter.ReleaseHostBasis(t.Context(), "foundation"); err != nil {
		t.Fatal(err)
	}
	if err := f.adapter.SetHostBasis(t.Context(), "foundation-r2", snapshot); err != nil {
		t.Fatal(err)
	}
}

type otherTarget struct{}

func (otherTarget) Stage(
	_ context.Context,
	r lifecycle.StageRequest,
	_ managed.Payload[metadata],
) (lifecycle.StageResult, error) {
	return lifecycle.StageResult{State: lifecycle.TargetReady, Revision: r.Manifest.Identity.Revision}, nil
}
func (otherTarget) Inspect(ctx context.Context, r lifecycle.StageRequest) (lifecycle.StageResult, error) {
	return otherTarget{}.Stage(ctx, r, managed.Payload[metadata]{})
}

func TestGraphSupportsBelongToSelectedTarget(t *testing.T) {
	for _, reverse := range []bool{false, true} {
		t.Run(map[bool]string{false: "other-first", true: "graph-first"}[reverse], func(t *testing.T) {
			f := publishedSharedReferences(t, reverse)
			// Act.
			result, err := f.adapter.Traverse(t.Context(), request(f.pin(t)))
			// Assert: no support is taken from the other target, regardless of ordering.
			if err != nil || len(result.Supports) != 3 {
				t.Fatalf("graph read: %v %#v", err, result)
			}
			for _, support := range result.Supports {
				if len(support.References) != 1 || support.References[0].Artifact != "graph-original" {
					t.Fatalf("cross-target support: %#v", support)
				}
			}
		})
	}
}

func publishedSharedReferences(t *testing.T, reverse bool) *fixture {
	t.Helper()
	// Arrange: a real lifecycle manifest reuses artifact references in two targets.
	f := newFixture(t)
	input := payload("policy", "r1")
	manifest := plan("joint", "", input, "graph-original")
	other := lifecycle.Target{Name: "other", Required: true, State: lifecycle.TargetPending}
	wrong := ref("policy", "r1", "other-original", "utf8")
	for _, artifact := range manifest.Targets[0].Artifacts {
		other.Artifacts = append(
			other.Artifacts,
			lifecycle.Artifact{Reference: artifact.Reference, Supports: []source.Reference{wrong}},
		)
	}
	manifest.Targets = append([]lifecycle.Target{other}, manifest.Targets...)
	if reverse {
		slices.Reverse(manifest.Targets)
	}
	executor, err := lifecycle.NewExecutor(lifecycle.ExecutorConfig[managed.Payload[metadata]]{
		Store: f.store,
		Targets: []lifecycle.Registration[managed.Payload[metadata]]{
			{Name: "graph", Port: f.adapter},
			{Name: "other", Port: otherTarget{}},
		},
		ClonePayload:    func(p managed.Payload[metadata]) (managed.Payload[metadata], error) { return p, nil },
		ValidatePayload: func(lifecycle.Manifest, managed.Payload[metadata]) error { return nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Prepare(t.Context(), manifest); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"graph", "other"} {
		if _, err = executor.Stage(t.Context(), "n", manifest.ID, name, input); err != nil {
			t.Fatal(err)
		}
	}
	if _, err = executor.Publish(t.Context(), "n", manifest.ID); err != nil {
		t.Fatal(err)
	}
	return f
}
