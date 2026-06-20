package retrieval

import (
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
)

func TestRankOnlyDocumentDoesNotRequireFakeScore(t *testing.T) {
	t.Parallel()

	doc := Document[struct{}]{
		ID:         "rank-only",
		Content:    "ordered hit",
		ScoreState: ScoreAbsent,
		Rank:       1,
	}
	if err := ValidateDocument(doc); err != nil {
		t.Fatalf("ValidateDocument(): %v", err)
	}

	rs := NewResultSet([]Document[struct{}]{doc}, DocumentIDResolver[struct{}]{})
	got := rs.Documents()[0]
	if got.ScoreState != ScoreAbsent || got.Score != 0 {
		t.Fatalf("document score = (%v, %v), want absent zero score", got.ScoreState, got.Score)
	}
}

func TestRankOnlyDocumentRejectsCarriedScore(t *testing.T) {
	t.Parallel()

	doc := Document[struct{}]{
		ID:         "rank-only",
		Content:    "ordered hit",
		Score:      0.5,
		ScoreState: ScoreAbsent,
		Rank:       1,
	}
	err := ValidateDocument(doc)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("ValidateDocument() error = %v, want invalid argument", err)
	}
}

func TestRankOnlyMergeUsesRankWithoutInventingScore(t *testing.T) {
	t.Parallel()

	left := NewResultSet([]Document[struct{}]{
		{ID: "same-low", Content: "same", ScoreState: ScoreAbsent, Rank: 2},
	}, sameContentResolver{})
	right := NewResultSet([]Document[struct{}]{
		{ID: "same-high", Content: "same", ScoreState: ScoreAbsent, Rank: 1},
	}, sameContentResolver{})

	merged, err := left.Merge(right)
	if err != nil {
		t.Fatalf("Merge(): %v", err)
	}
	doc := merged.Documents()[0]
	if doc.ID != "same-high" {
		t.Fatalf("winner = %q, want best rank", doc.ID)
	}
	if doc.ScoreState != ScoreAbsent || doc.Score != 0 {
		t.Fatalf("winner score = (%v, %v), want scoreless winner", doc.ScoreState, doc.Score)
	}
}

func TestNormalizeRankOnlyResultSetRequiresExplicitPolicy(t *testing.T) {
	t.Parallel()

	rs := NewResultSet([]Document[struct{}]{
		{ID: "a", Content: "A", ScoreState: ScoreAbsent, Rank: 1},
		{ID: "b", Content: "B", ScoreState: ScoreAbsent, Rank: 2},
	}, DocumentIDResolver[struct{}]{})

	out, err := NormalizeRankOnlyResultSet(rs, LinearRankNormalizer{})
	if err != nil {
		t.Fatalf("NormalizeRankOnlyResultSet(): %v", err)
	}
	docs := out.Documents()
	if docs[0].ScoreState != ScoreNormalized || docs[0].Score != 1 {
		t.Fatalf("doc[0] score = (%v, %v), want normalized 1", docs[0].ScoreState, docs[0].Score)
	}
	if docs[1].ScoreState != ScoreNormalized || docs[1].Score != 0 {
		t.Fatalf("doc[1] score = (%v, %v), want normalized 0", docs[1].ScoreState, docs[1].Score)
	}
}

func TestNormalizeRankOnlyResultSetRejectsMissingPolicy(t *testing.T) {
	t.Parallel()

	rs := NewResultSet([]Document[struct{}]{
		{ID: "a", Content: "A", ScoreState: ScoreAbsent, Rank: 1},
	}, DocumentIDResolver[struct{}]{})

	out, err := NormalizeRankOnlyResultSet(rs, nil)
	if out == nil || !out.IsEmpty() {
		t.Fatalf("result = %#v, want empty preserved result on policy error", out)
	}
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("error = %v, want invalid argument", err)
	}
}

type sameContentResolver struct{}

func (sameContentResolver) Resolve(doc Document[struct{}]) Identity {
	return Identity{DocumentID: doc.ID, MergeKey: doc.Content}
}
