package retrieval

import (
	"errors"
	"math"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/source"
)

func TestRRFOneVotePerSourcePreservesDuplicateEvidence(t *testing.T) {
	// Arrange: B appears twice in one source with distinct observed supports.
	rrf, err := NewReciprocalRankFusion[struct{}](60, nil)
	if err != nil {
		t.Fatal(err)
	}
	a := Document[struct{}]{ID: "a", Content: "A"}
	locator := func(id string) source.Locator {
		return source.Locator{
			Reference: source.Reference{
				Namespace:         "n",
				Source:            "s",
				Revision:          "r1",
				Transformation:    "original",
				AccessFingerprint: "acl",
				Artifact:          id,
				Representation:    "utf8",
			},
			Kind: source.DocumentLocation,
		}
	}
	b := Document[struct{}]{
		ID:             "b",
		Content:        "B",
		ScoreState:     ScorePresent,
		ScoreSemantics: "native",
		Score:          2,
		SourceSupports: []source.Locator{locator("p1")},
	}
	duplicate := b
	duplicate.Score = 3
	duplicate.SourceSupports = []source.Locator{locator("p2")}
	list := NewResultSet([]Document[struct{}]{a, b, duplicate}, nil)
	// Act.
	out, err := rrf.Merge(t.Context(), list)
	// Assert: rank positions, not duplicate frequency, determine the one-list score.
	if err != nil {
		t.Fatal(err)
	}
	docs := out.Documents()
	if len(docs) != 2 || docs[0].ID != "a" || docs[1].ID != "b" {
		t.Fatalf("one-list order: %#v", docs)
	}
	if math.Abs(docs[1].Score-61.0/62) > 1e-12 {
		t.Fatalf("B got duplicate vote: %g", docs[1].Score)
	}
	if len(docs[1].SourceLocations()) != 2 || len(docs[1].ScoreHistory) != 2 {
		t.Fatalf("evidence lost: %#v", docs[1])
	}
	// Act: another source independently ranks B first.
	consensus, err := rrf.Merge(t.Context(), list, NewResultSet([]Document[struct{}]{b}, nil))
	// Assert.
	if err != nil || consensus.Documents()[0].ID != "b" {
		t.Fatalf("cross-list consensus: %v %v", consensus.Documents(), err)
	}
	duplicate.Content = "different"
	_, err = rrf.Merge(t.Context(), NewResultSet([]Document[struct{}]{b, duplicate}, nil))
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("duplicate conflict hidden: %v", err)
	}
}
