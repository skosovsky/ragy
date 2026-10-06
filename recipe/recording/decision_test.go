package recording

import (
	"testing"

	"github.com/skosovsky/ragy/evidence"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

func TestDecisionUsesPackedContributorOrdinalsAndUncertainty(t *testing.T) {
	result := recipe.Result[struct{}]{
		Queries: []recipe.QueryEvidence[struct{}]{{Index: 0, Text: "private"}, {Index: 1, Text: "other"}},
		Selected: []recipe.SelectedEvidence[struct{}]{
			{Contributors: []recipe.Contribution{{QueryIndex: 0, Rank: 1}}},
			{Contributors: []recipe.Contribution{{QueryIndex: 1, Rank: 1}}},
		},
		Fusion: recipe.FusionObserved,
		Stop:   recipe.Assessed,
		Artifact: &retrieval.RetrievalContextArtifact[struct{}]{
			Snippets: []retrieval.ContextSnippet[struct{}]{
				{
					Contributors: []retrieval.ArtifactContribution{
						{InputIndex: 0, FullDocument: false, DeliveryUncertain: true},
					},
				},
			},
		},
	}
	decision := decisions(result, evidence.HostRevisions{})
	if !decision.Selected[0].Delivered || !decision.Selected[0].Uncertain || decision.Selected[1].Delivered ||
		!decision.Queries[0].Selected ||
		!decision.Queries[0].Uncertain ||
		decision.Queries[1].Delivered ||
		decision.Sufficiency != nil {
		t.Fatalf("fabricated decision: %+v", decision)
	}
}

func TestFailedModelDispatchIsMissingRatherThanObservedEmpty(t *testing.T) {
	// Arrange.
	result := recipe.Result[struct{}]{
		Stages: []recipe.Stage{{Operation: recipe.Plan, Completed: false}},
		Fusion: recipe.FusionNotRun,
	}
	// Act.
	output, err := stages(result)
	// Assert.
	if err != nil || output[0].Status != evidence.MissingObservation || len(output[0].Hits) != 0 {
		t.Fatal(output, err)
	}
	result.Stages[0].Completed = true
	output, err = stages(result)
	if err != nil || output[0].Status != evidence.StageObserved {
		t.Fatal(output, err)
	}
}

func TestDeliverySourceInventoryExcludesUnpackedSameIDContributor(t *testing.T) {
	// Arrange: equal document IDs do not authorize a discarded selected input.
	first := source.Locator{
		Kind: source.DocumentLocation,
		Reference: source.Reference{
			Namespace:         "n",
			Source:            "first",
			Revision:          "r1",
			Transformation:    "original",
			AccessFingerprint: "acl",
			Artifact:          "a",
			Representation:    "text",
		},
	}
	foreign := first
	foreign.Reference.Source = "foreign"
	result := recipe.Result[struct{}]{
		Selected: []recipe.SelectedEvidence[struct{}]{
			{
				Contributors: []recipe.Contribution{
					{QueryIndex: 0, DocumentID: "same", Rank: 1, Supports: []source.Locator{first}},
				},
			},
			{
				Contributors: []recipe.Contribution{
					{QueryIndex: 1, DocumentID: "same", Rank: 1, Supports: []source.Locator{foreign}},
				},
			},
		},
		Artifact: &retrieval.RetrievalContextArtifact[struct{}]{
			Snippets: []retrieval.ContextSnippet[struct{}]{
				{
					DocumentID: "same",
					Content:    "derived",
					Contributors: []retrieval.ArtifactContribution{
						{InputIndex: 0, FullDocument: false, DeliveryUncertain: true},
					},
				},
			},
		},
	}
	// Act.
	stage, err := deliveryStage(result)
	// Assert: exact retained input contributes lineage; the same ID does not.
	if err != nil || len(stage.Hits) != 1 || len(stage.Hits[0].Sources) != 1 ||
		stage.Hits[0].Sources[0] != first.Reference ||
		len(stage.Hits[0].Contributions) != 1 ||
		stage.Hits[0].Contributions[0].QueryIndex != 0 {
		t.Fatal(stage, err)
	}
}
