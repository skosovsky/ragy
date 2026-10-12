package retrieval

import (
	"context"
	"strconv"
	"testing"

	"github.com/skosovsky/ragy/source"
)

func compatibleEvidenceDocuments(n int) []Document[struct{}] {
	docs := make([]Document[struct{}], n)
	shared := source.Locator{
		Reference: source.Reference{
			Namespace:         "n",
			Source:            "shared",
			Revision:          "r",
			Transformation:    "t",
			AccessFingerprint: "a",
			Artifact:          "a",
			Representation:    "text",
		},
		Kind: source.DocumentLocation,
	}
	for i := range docs {
		distinct := shared
		distinct.Reference.Source = strconv.Itoa(i)
		docs[i] = Document[struct{}]{
			ID:             "same",
			Content:        "same",
			Score:          1,
			ScoreState:     ScorePresent,
			ScoreSemantics: "test",
			SourceSupports: []source.Locator{shared, distinct},
		}
	}
	return docs
}

func assertCompatibleEvidence(t *testing.T, docs []Document[struct{}], expected int) {
	t.Helper()
	if len(docs) != 1 || len(docs[0].SourceSupports) != expected+1 {
		t.Fatalf("unexpected evidence cardinality: %+v", docs)
	}
	for i, support := range docs[0].SourceSupports {
		want := "shared"
		if i > 0 {
			want = strconv.Itoa(i - 1)
		}
		if support.Reference.Source != want {
			t.Fatalf("support %d=%q, want %q", i, support.Reference.Source, want)
		}
	}
}

//nolint:gocognit // Keep the cross-operation evidence and ownership assertions together.
func TestCompatibleAggregationsEvidenceOrderAndOwnership(t *testing.T) {
	for _, operation := range []string{"rrf", "default-merge", "group-by"} {
		t.Run(operation, func(t *testing.T) {
			// Arrange: repeated payloads with overlapping evidence in known first-seen order.
			const n = 128
			input := compatibleEvidenceDocuments(n)
			original := NewResultSet(input, nil)
			var output []Document[struct{}]
			// Act.
			switch operation {
			case "rrf":
				rrf, err := NewReciprocalRankFusion[struct{}](60, nil)
				if err != nil {
					t.Fatal(err)
				}
				merged, err := rrf.Merge(context.Background(), original)
				if err != nil {
					t.Fatal(err)
				}
				output = merged.Documents()
				if len(output[0].ScoreHistory) != n {
					t.Fatalf("RRF observations=%d, want %d", len(output[0].ScoreHistory), n)
				}
				output[0].ScoreHistory[0].Value = 7
				if merged.Documents()[0].ScoreHistory[0].Value != 1 {
					t.Fatal("RRF history aliased public result")
				}
			case "default-merge":
				merged, err := DefaultMergeStrategy[struct{}]()(input)
				if err != nil {
					t.Fatal(err)
				}
				output = []Document[struct{}]{merged}
				if len(merged.ScoreHistory) != n {
					t.Fatalf("merge observations=%d, want %d", len(merged.ScoreHistory), n)
				}
			case "group-by":
				// The custom strategy contributes an overlapping prefix; GroupBy must union
				// that prefix with every input in encounter order without borrowed aliasing.
				strategy := func(docs []Document[struct{}]) (Document[struct{}], error) { return docs[0], nil }
				merged, err := GroupBy(
					func(struct{}) string { return "group" },
					strategy,
				).Process(context.Background(), UnrestrictedRead(), original)
				if err != nil {
					t.Fatal(err)
				}
				output = merged.Documents()
			}
			// Assert: ordered exact union and public/input slice isolation.
			assertCompatibleEvidence(t, output, n)
			output[0].SourceSupports[0].Reference.Source = "mutation"
			if input[0].SourceSupports[0].Reference.Source != "shared" ||
				original.Documents()[0].SourceSupports[0].Reference.Source != "shared" {
				t.Fatal("aggregation mutated input evidence")
			}
		})
	}
}
