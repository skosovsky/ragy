//go:build darwin || linux

package lifecycle_test

import (
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
	"github.com/skosovsky/ragy/source"
)

func batchReservationPlan(id, content, payload string) lifecycle.Manifest {
	plan := manifestFixture()
	plan.ID, plan.Key, plan.Identity.Content, plan.Payload = id, id, content, payload
	plan.State = lifecycle.Planned
	plan.PublishedAt = time.Time{}
	for i := range plan.Targets {
		plan.Targets[i].State = lifecycle.TargetPending
		plan.Targets[i].Revision = ""
	}
	return plan
}

func TestCASRejectsConflictingNewInventoryBatchAtomically(t *testing.T) {
	for _, withCurrent := range []bool{false, true} {
		name := "empty namespace"
		if withCurrent {
			name = "existing namespace"
		}
		t.Run(name, func(t *testing.T) {
			// Arrange: two distinct payload plans share the same exact target/reference;
			// the optional prior inventory is independent of both new reservations.
			store, current := batchNamespaceFixture(t, withCurrent)
			next, err := store.Load(t.Context(), "n")
			if err != nil {
				t.Fatal(err)
			}
			next.Manifests = append(
				next.Manifests,
				batchReservationPlan("first", "first-content", "first-payload"),
				batchReservationPlan("second", "different-content", "different-payload"),
			)
			if err = next.Validate(); err != nil {
				t.Fatal("invalid individual inventory fixture", err)
			}
			// Act: exercise both custom-store validation and actual filesystem commit.
			replacementErr := lifecycle.ValidateReplacement(current, next)
			returned, casErr := store.CompareSwap(t.Context(), current.Generation, next)
			retained, loadErr := store.Load(t.Context(), "n")
			// Assert: the whole generation stays unchanged and no new payload plan persists.
			if !errors.Is(replacementErr, lifecycle.ErrConflict) || !errors.Is(casErr, lifecycle.ErrConflict) ||
				loadErr != nil ||
				returned.Generation != 0 ||
				!reflect.DeepEqual(current, retained) {
				t.Fatal(
					"batch aliased inventory or partially committed",
					replacementErr,
					casErr,
					loadErr,
					current.Generation,
					retained.Generation,
				)
			}
		})
	}
}

func batchNamespaceFixture(t *testing.T, withCurrent bool) (*filestore.Store, lifecycle.Snapshot) {
	t.Helper()
	store, err := filestore.New(t.TempDir(), 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	current, err := store.Load(t.Context(), "n")
	if err != nil {
		t.Fatal(err)
	}
	if withCurrent {
		prior := batchReservationPlan("prior", "prior-content", "prior-payload")
		prior.Targets[0].Artifacts[0].Reference.Artifact = "independent"
		prior.Targets[0].Artifacts[0].Supports = []source.Reference{prior.Targets[0].Artifacts[0].Reference}
		current.Manifests = append(current.Manifests, prior)
		current, err = store.CompareSwap(t.Context(), current.Generation, current)
		if err != nil {
			t.Fatal(err)
		}
	}
	return store, current
}

func TestCASAllowsSharedSupportsAndDistinctTargetReservations(t *testing.T) {
	// Arrange: support provenance may be shared; exact ownership includes target.
	store, err := filestore.New(t.TempDir(), 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	first := batchReservationPlan("first", "first-content", "first-payload")
	second := batchReservationPlan("second", "second-content", "second-payload")
	second.Targets[0].Artifacts[0].Reference.Artifact = "independent"
	third := batchReservationPlan("third", "third-content", "third-payload")
	third.Targets = third.Targets[:1]
	third.Targets[0].Name = "graph"
	next := lifecycle.Snapshot{
		Schema:    lifecycle.SchemaIdentity,
		Namespace: "n",
		Manifests: []lifecycle.Manifest{first, second, third},
	}
	// Act.
	committed, err := store.CompareSwap(t.Context(), 0, next)
	// Assert: sharing support does not collapse independent artifact/target ownership.
	if err != nil || committed.Generation != 1 || len(committed.Manifests) != 3 {
		t.Fatal("independent reservations rejected", err)
	}
}
