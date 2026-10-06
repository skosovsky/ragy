package retrieval_test

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

type sameSourceKey struct{}

func (sameSourceKey) Resolve(doc retrieval.Document[struct{}]) retrieval.Identity {
	return retrieval.Identity{DocumentID: doc.ID, MergeKey: "same"}
}

func mappedDocument(t *testing.T, id, text string) retrieval.Document[struct{}] {
	t.Helper()
	loc := artifactLocation(id)
	loc.Span = source.ByteSpan{Start: 0, End: len(text)}
	mapping, err := source.OriginalText(loc, text)
	if err != nil {
		t.Fatal(err)
	}
	return retrieval.Document[struct{}]{
		ID:             id,
		Content:        text,
		SourceMapping:  mapping,
		Score:          1,
		ScoreState:     retrieval.ScorePresent,
		ScoreSemantics: "fixture",
	}
}

func TestDocumentGroupMappingAndImplicitArtifactCoordinates(t *testing.T) {
	// Arrange: two source revisions with UTF-8 text and independent coordinates.
	docs := []retrieval.Document[struct{}]{
		mappedDocument(t, "a", "Привет"),
		mappedDocument(t, "b", "world"),
	}
	group := retrieval.GroupBy(
		func(struct{}) string { return "all" },
		retrieval.DefaultMergeStrategy[struct{}](),
	)
	// Act.
	result, err := group.Process(
		context.Background(),
		retrieval.UnrestrictedRead(),
		retrieval.NewResultSet(docs, nil),
	)
	if err != nil {
		t.Fatal(err)
	}
	artifact, err := (retrieval.DefaultArtifactRenderer[struct{}]{}).Render(
		context.Background(),
		retrieval.UnrestrictedRead(),
		result,
		retrieval.ArtifactRenderOptions[struct{}]{
			Resource:  retrieval.RuneResource(100000),
			CloneMeta: func(v struct{}) (struct{}, error) { return v, nil },
		},
	)
	// Assert: separator is derived and complete content preserves both source ranges.
	merged := result.Documents()[0]
	if err != nil || merged.Content != "Привет\n\nworld" || len(merged.SourceLocations()) != 2 ||
		len(artifact.Snippets) != 1 {
		t.Fatal(merged, artifact, err)
	}
	snippet := artifact.Snippets[0]
	fragments := snippet.Mapping.Fragments()
	if snippet.Content != "Привет\n\nworld" || len(fragments) != 3 ||
		fragments[1].Origin != source.DerivedContent ||
		fragments[2].Location.Reference.Source != "b" ||
		fragments[2].Location.Span.End != 5 {
		t.Fatal(snippet, fragments)
	}
}

func TestDocumentDedupAndRRFPreserveEverySource(t *testing.T) {
	// Arrange: equal payloads with different source support; RRF requires equal metadata/text.
	first, second := mappedDocument(t, "a", "same"), mappedDocument(t, "b", "same")
	second.Score = 2
	resolver := sameSourceKey{}
	fusion, err := retrieval.NewReciprocalRankFusion(60, resolver)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	dedup, err := retrieval.NewResultSet([]retrieval.Document[struct{}]{first, second}, resolver).
		Dedup()
	if err != nil {
		t.Fatal(err)
	}
	fused, err := fusion.Merge(
		context.Background(),
		retrieval.NewResultSet([]retrieval.Document[struct{}]{first}, resolver),
		retrieval.NewResultSet([]retrieval.Document[struct{}]{second}, resolver),
	)
	// Assert: exact winner mapping stays local, supports retain both contributors.
	if err != nil {
		t.Fatal(err)
	}
	for _, result := range []retrieval.ResultSet[struct{}]{dedup, fused} {
		doc := result.Documents()[0]
		if len(doc.SourceLocations()) != 2 || len(doc.SourceMapping.Fragments()) != 1 {
			t.Fatal(doc)
		}
		doc.SourceSupports[0].Reference.Source = "mutated"
		if result.Documents()[0].SourceSupports[0].Reference.Source == "mutated" {
			t.Fatal("result supports aliased")
		}
	}
	if dedup.Documents()[0].ID != "b" ||
		dedup.Documents()[0].SourceMapping.Supports()[0].Reference.Source != "b" {
		t.Fatal("loser coordinates substituted")
	}
}

func TestUnknownGroupPrecisionAndStringRewriteRemainExplicit(t *testing.T) {
	// Arrange: one fragment lacks mapping; one callback rewrites known content.
	first := mappedDocument(t, "a", "hello")
	unmapped := retrieval.Document[struct{}]{
		ID:             "b",
		Content:        "unknown",
		Score:          1,
		ScoreState:     retrieval.ScorePresent,
		ScoreSemantics: "fixture",
	}
	// Act.
	merged, err := retrieval.DefaultMergeStrategy[struct{}]()(
		[]retrieval.Document[struct{}]{first, unmapped},
	)
	artifact, renderErr := (retrieval.DefaultArtifactRenderer[struct{}]{}).Render(
		context.Background(),
		retrieval.UnrestrictedRead(),
		retrieval.NewResultSet([]retrieval.Document[struct{}]{first}, nil),
		retrieval.ArtifactRenderOptions[struct{}]{
			Resource:  retrieval.RuneResource(100000),
			Snippet:   func(retrieval.Document[struct{}]) string { return "changed" },
			CloneMeta: func(v struct{}) (struct{}, error) { return v, nil },
		},
	)
	// Assert: unknown mapping is not silently attributed to the winning source.
	if err != nil || merged.SourceMapping.Text() != "" || len(merged.SourceLocations()) != 1 {
		t.Fatal(merged, err)
	}
	if renderErr != nil || artifact.Snippets[0].Mapping.Text() != "" ||
		len(artifact.Snippets[0].Supports) != 1 {
		t.Fatal(artifact, renderErr)
	}
	first.Content = "stale mapping"
	if !errors.Is(retrieval.ValidateDocument(first), ragy.ErrInvalidArgument) {
		t.Fatal("stale coordinate mapping accepted")
	}
}

func TestBusinessMergeKeyDoesNotTransferDifferentEvidence(t *testing.T) {
	for _, equal := range []bool{false, true} {
		// Arrange.
		left := mappedDocument(t, "a", "winner")
		text := "loser"
		if equal {
			text = "winner"
		}
		right := mappedDocument(t, "b", text)
		right.Score = 0.5
		left.ScoreHistory = left.ObservedScores()
		right.ScoreHistory = right.ObservedScores()
		input := retrieval.NewResultSet(
			[]retrieval.Document[struct{}]{right, left},
			sameSourceKey{},
		)
		// Act.
		out, err := input.Dedup()
		// Assert: identical evidence can retain both original supports; other text cannot.
		if err != nil || out.Len() != 1 {
			t.Fatal(out, err)
		}
		winner := out.Documents()[0]
		want := 1
		if equal {
			want = 2
		}
		if winner.ID != "a" || winner.Content != "winner" ||
			len(winner.SourceLocations()) != want ||
			len(winner.ScoreHistory) != 2*want-1 {
			t.Fatal(equal, winner)
		}
		if len(input.Documents()[0].SourceLocations()) != 1 {
			t.Fatal("input mutated")
		}
	}
}
