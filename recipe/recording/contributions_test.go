package recording

import (
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

func TestContributorMustMatchRecordedQueryHit(t *testing.T) {
	for _, failure := range []string{"valid", "query", "rank", "document", "support"} {
		t.Run(failure, func(t *testing.T) {
			// Arrange.
			loc := source.Locator{
				Kind: source.DocumentLocation,
				Reference: source.Reference{
					Namespace:         "n",
					Source:            "s",
					Revision:          "r1",
					Transformation:    "original",
					AccessFingerprint: "acl",
					Artifact:          "p",
					Representation:    "text",
				},
			}
			item := recipe.Contribution{QueryIndex: 0, DocumentID: "p", Rank: 1, Supports: []source.Locator{loc}}
			switch failure {
			case "query":
				item.QueryIndex = 2
			case "rank":
				item.Rank = 0
			case "document":
				item.DocumentID = "foreign"
			case "support":
				item.Supports[0].Reference.Revision = "r2"
			}
			result := recipe.Result[struct{}]{
				Queries: []recipe.QueryEvidence[struct{}]{
					{
						Index:     0,
						Documents: []retrieval.Document[struct{}]{{ID: "p", Content: "text"}},
						Supports:  [][]source.Locator{{loc}},
					},
				},
				Selected: []recipe.SelectedEvidence[struct{}]{{Contributors: []recipe.Contribution{item}}},
			}
			// Act.
			err := validateQueryContributions(result)
			// Assert.
			if failure == "valid" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if !errors.Is(err, ragy.ErrProtocol) {
				t.Fatal("forged contributor admitted", err)
			}
		})
	}
}
