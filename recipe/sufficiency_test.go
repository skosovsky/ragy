package recipe_test

import (
	"context"
	"testing"

	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/retrieval"
)

func TestSufficiencyIsObservedAndSnapshotOwned(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.SingleRewrite)
	f.results["original"] = []retrieval.Document[meta]{document("one")}
	f.selected = []int{0}
	f.sufficient = false
	// Act.
	result, err := f.run(context.Background(), t)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := recipe.SnapshotResult(context.Background(), f.read, result, f.config.CloneMeta)
	if err != nil {
		t.Fatal(err)
	}
	*result.Sufficiency = true
	// Assert.
	if snapshot.Sufficiency == nil || *snapshot.Sufficiency {
		t.Fatal("snapshot did not retain observed false independently")
	}
}

func TestSufficiencyIsUnavailableWhenAssessmentNotDispatched(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.SingleRewrite)
	f.results["original"] = []retrieval.Document[meta]{document("one")}
	f.config.Limits.ModelCalls = 1
	// Act.
	result, err := f.run(context.Background(), t)
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	if result.Sufficiency != nil || result.Stop != recipe.BudgetExhausted {
		t.Fatalf("unexpected assessment observation: %+v", result)
	}
	if f.modelCalls != 1 {
		t.Fatalf("model dispatches: %d", f.modelCalls)
	}
}
