package recording

import (
	"testing"

	"errors"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/evidence"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/retrieval"
)

func TestAccountingDoesNotRoundLargeIntegers(t *testing.T) {
	// Arrange.
	result := recipe.Result[struct{}]{
		Stages: []recipe.Stage{
			{
				Operation: recipe.Encode,
				Usage: recipe.Usage{
					Known: true,
					Value: budget.Usage{InputTokens: 1 << 53, OutputTokens: (1 << 53) + 1, Cost: ^uint64(0)},
				},
			},
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
	result.Stages[0].Usage.Known = false
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

func TestOwnDiagnosticsExcludeOtherSharedLedgerCallers(t *testing.T) {
	// Arrange.
	result := recipe.Result[struct{}]{
		Budget: budget.Snapshot{
			Actual:       budget.Usage{InputTokens: 9999, OutputTokens: 9999, Cost: 9999},
			UnknownUsage: 1,
		},
		Stages: []recipe.Stage{
			{
				Operation: recipe.Encode,
				Usage:     recipe.Usage{Known: true, Value: budget.Usage{InputTokens: 7, OutputTokens: 2, Cost: 3}},
			},
		},
	}
	// Act.
	values := diagnostics(result)
	// Assert.
	for i, want := range []float64{7, 2, 3} {
		value := values[i+2].Number
		if value.State != evidence.Observed || value.Value == nil || *value.Value != want {
			t.Fatal(value)
		}
	}
}

func TestDeliveryUsesPackedInputContributors(t *testing.T) {
	// Arrange.
	result := recipe.Result[struct{}]{
		Selected: []recipe.SelectedEvidence[struct{}]{
			{Contributors: []recipe.Contribution{{QueryIndex: 0, DocumentID: "same", Rank: 1}}},
			{Contributors: []recipe.Contribution{{QueryIndex: 1, DocumentID: "same", Rank: 1}}},
		},
		Artifact: &retrieval.RetrievalContextArtifact[struct{}]{
			Snippets: []retrieval.ContextSnippet[struct{}]{
				{
					DocumentID:   "same",
					Content:      "onlysecond",
					Contributors: []retrieval.ArtifactContribution{{InputIndex: 1, FullDocument: true}},
				},
			},
		},
	}
	// Act.
	stage, err := deliveryStage(result)
	// Assert.
	if err != nil || len(stage.Hits) != 1 || len(stage.Hits[0].Contributions) != 1 ||
		stage.Hits[0].Contributions[0].QueryIndex != 1 {
		t.Fatal(stage, err)
	}
	result.Artifact.Snippets[0].Contributors[0].InputIndex = 2
	_, err = deliveryStage(result)
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatal(err)
	}
}
