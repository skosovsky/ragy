package recording

import (
	"testing"

	"github.com/skosovsky/ragy/evidence"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/budget"
)

func TestAccountingDoesNotRoundLargeIntegers(t *testing.T) {
	// Arrange.
	result := recipe.Result[struct{}]{
		Budget: budget.Snapshot{
			Actual: budget.Usage{InputTokens: 1 << 53, OutputTokens: (1 << 53) + 1, Cost: ^uint64(0)},
		},
	}
	// Act.
	values := diagnostics(result)
	// Assert.
	if values[2].Number.State != evidence.Observed || *values[2].Number.Value != 1<<53 {
		t.Fatal(values[2])
	}
	for _, value := range values[3:] {
		if value.Number.State != evidence.Unavailable || value.Number.Value != nil {
			t.Fatal(value)
		}
	}
	result.Budget.UnknownUsage = 1
	values = diagnostics(result)
	if values[2].Number.State != evidence.Unavailable {
		t.Fatal(values[2])
	}
}

func TestDispatchedButUnobservedRetrievalHasDistinctStages(t *testing.T) {
	// Arrange: two dispatched operations have no retained observation.
	result := recipe.Result[struct{}]{
		Stages: []recipe.Stage{{Operation: recipe.Retrieve}, {Operation: recipe.Retrieve}},
		Fusion: recipe.FusionNotRun,
	}
	// Act.
	output, err := stages(result)
	// Assert.
	if err != nil || output[0].Name != "retrieve/0" || output[1].Name != "retrieve/1" ||
		output[0].Status != evidence.MissingObservation ||
		output[1].Status != evidence.MissingObservation {
		t.Fatal(output, err)
	}
}

func TestFusionObservationDistinguishesUnstartedAndMissing(t *testing.T) {
	for _, observation := range []recipe.FusionObservation{recipe.FusionNotRun, recipe.FusionMissing} {
		t.Run(string(observation), func(t *testing.T) {
			// Arrange.
			result := recipe.Result[struct{}]{Fusion: observation}
			// Act.
			output, err := stages(result)
			// Assert.
			if err != nil {
				t.Fatal(err)
			}
			last := output[len(output)-1]
			expected := evidence.NotRun
			if observation == recipe.FusionMissing {
				expected = evidence.MissingObservation
			}
			if last.Name != "fusion" || last.Status != expected || len(last.Hits) != 0 ||
				last.Scores != evidence.Unavailable {
				t.Fatal("fusion observation fabricated hits", last)
			}
		})
	}
}
