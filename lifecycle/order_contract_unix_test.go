//go:build darwin || linux

package lifecycle_test

import (
	"errors"
	"reflect"
	"slices"
	"testing"

	"github.com/skosovsky/ragy/lifecycle"
)

func TestReservedPlanOrderCannotChangeOnReplayOrCAS(t *testing.T) {
	for _, order := range []string{"targets", "artifacts"} {
		t.Run(order, func(t *testing.T) {
			// Arrange: real durable reservation includes two ordered artifact entries.
			executor, store, _, _ := executorFixture(t)
			plan := plannedManifest()
			artifact := plan.Targets[0].Artifacts[0]
			artifact.Reference.Artifact = "p2"
			plan.Targets[0].Artifacts = append(plan.Targets[0].Artifacts, artifact)
			if _, err := executor.Prepare(t.Context(), plan); err != nil {
				t.Fatal(err)
			}
			before, err := store.Load(t.Context(), "n")
			if err != nil {
				t.Fatal(err)
			}
			changed, err := store.Load(t.Context(), "n")
			if err != nil {
				t.Fatal(err)
			}
			if order == "targets" {
				slices.Reverse(changed.Manifests[0].Targets)
			} else {
				slices.Reverse(changed.Manifests[0].Targets[0].Artifacts)
			}
			// Act: identical members in another order are a different reserved plan.
			_, replayErr := executor.Prepare(t.Context(), changed.Manifests[0])
			_, swapErr := store.CompareSwap(t.Context(), changed.Generation, changed)
			after, loadErr := store.Load(t.Context(), "n")
			// Assert: both paths reject; the exact original reservation/generation survives.
			if !errors.Is(replayErr, lifecycle.ErrIdempotencyConflict) ||
				!errors.Is(swapErr, lifecycle.ErrIdempotencyConflict) ||
				loadErr != nil ||
				!reflect.DeepEqual(before, after) {
				t.Fatal(replayErr, swapErr, loadErr, before, after)
			}
		})
	}
}
