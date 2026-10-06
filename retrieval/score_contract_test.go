package retrieval

import (
	"context"
	"errors"
	"math"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/tensor"
)

func nativeFixture(id string, value float64, semantics ScoreSemantics) Document[struct{}] {
	return Document[struct{}]{ID: id, Content: id, Score: value, ScoreState: ScorePresent, ScoreSemantics: semantics}
}

func TestNativeScoreContract(t *testing.T) {
	for _, value := range []float64{-1, 0, 1, 2} {
		// Arrange.
		doc := nativeFixture("doc", value, tensor.MaxSimSemantics)
		// Act.
		err := ValidateDocument(doc)
		// Assert.
		if err != nil {
			t.Fatalf("native %v rejected: %v", value, err)
		}
	}
	for _, doc := range []Document[struct{}]{
		nativeFixture("missing-semantics", 1, ""),
		nativeFixture("nan", math.NaN(), tensor.MaxSimSemantics),
		nativeFixture("inf", math.Inf(1), tensor.MaxSimSemantics),
		{ID: "invalid-normalized", Score: 2, ScoreState: ScoreNormalized, ScoreSemantics: "explicit-normalizer"},
		{ID: "invalid-absent", Score: 1},
	} {
		// Act.
		err := ValidateDocument(doc)
		// Assert.
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("invalid score accepted: %+v, %v", doc, err)
		}
	}
	if err := ValidateDocument(Document[struct{}]{ID: "rank-only", Rank: 1}); err != nil {
		t.Fatal(err)
	}
}

func TestScoreThresholdSupportsNegativeAndZero(t *testing.T) {
	// Arrange.
	rs := NewResultSet(
		[]Document[struct{}]{
			nativeFixture("t1", 2, tensor.MaxSimSemantics),
			nativeFixture("t2", 1, tensor.MaxSimSemantics),
			nativeFixture("t3", -1, tensor.MaxSimSemantics),
		},
		nil,
	)
	for _, tc := range []struct {
		value float64
		want  int
	}{{-1, 3}, {0, 2}, {2, 1}} {
		// Act.
		out, err := applyTerminalOptions(
			rs,
			RetrieveOptions{
				TopK:      10,
				Threshold: &ScoreThreshold{Value: tc.value, State: ScorePresent, Semantics: tensor.MaxSimSemantics},
			},
			DocumentIDResolver[struct{}]{},
		)
		// Assert.
		if err != nil || out.Len() != tc.want {
			t.Fatalf("threshold %v: len=%d, err=%v", tc.value, out.Len(), err)
		}
	}
	// Act: no threshold retains the negative score.
	out, err := applyTerminalOptions(rs, RetrieveOptions{TopK: 10}, DocumentIDResolver[struct{}]{})
	// Assert.
	if err != nil || out.Len() != 3 || out.Documents()[2].Score != -1 {
		t.Fatalf("nil threshold changed native evidence: %v, %v", out.Documents(), err)
	}
}

func TestIncompatibleScoresRequireExplicitPolicy(t *testing.T) {
	// Arrange.
	left := NewResultSet([]Document[struct{}]{nativeFixture("a", 2, tensor.MaxSimSemantics)}, nil)
	right := NewResultSet([]Document[struct{}]{nativeFixture("b", 0.9, "dense.cosine")}, nil)
	// Act: numeric merger may not compare unrelated scales.
	_, err := NewScoreMerger[struct{}](nil).Merge(context.Background(), left, right)
	// Assert.
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("incompatible numeric merge accepted: %v", err)
	}
	// Act: explicitly selected rank fusion accepts the ranked lists.
	fusion, err := NewReciprocalRankFusion[struct{}](60, nil)
	if err != nil {
		t.Fatal(err)
	}
	fused, err := fusion.Merge(context.Background(), left, right)
	// Assert.
	if err != nil || fused.Len() != 2 {
		t.Fatalf("explicit fusion failed: %v", err)
	}
	for _, doc := range fused.Documents() {
		if doc.ScoreState != ScoreNormalized || doc.ScoreSemantics != "rank.rrf.relative-max:k=60" {
			t.Fatalf("undeclared derived score: %+v", doc)
		}
	}
}

func TestThresholdRejectsIncompatibleAndAbsentEvidence(t *testing.T) {
	for _, doc := range []Document[struct{}]{nativeFixture("native", 2, tensor.MaxSimSemantics), {ID: "rank-only", Rank: 1}} {
		// Arrange.
		rs := NewResultSet([]Document[struct{}]{doc}, nil)
		threshold := &ScoreThreshold{Value: 0.5, State: ScorePresent, Semantics: "dense.cosine"}
		// Act.
		_, err := applyTerminalOptions(
			rs,
			RetrieveOptions{TopK: 1, Threshold: threshold},
			DocumentIDResolver[struct{}]{},
		)
		// Assert.
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("incompatible threshold accepted: %+v, %v", doc, err)
		}
	}
	if err := (ScoreThreshold{}).Validate(); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("rank-only threshold accepted: %v", err)
	}
}

func TestFusionAndRenderPreserveNativeScoreEvidence(t *testing.T) {
	// Arrange.
	original := nativeFixture("tensor-doc", 2, tensor.MaxSimSemantics)
	left := NewResultSet([]Document[struct{}]{original}, nil)
	right := NewResultSet([]Document[struct{}]{nativeFixture("dense-doc", -0.5, "dense.cosine")}, nil)
	fusion, err := NewReciprocalRankFusion[struct{}](60, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	fused, err := fusion.Merge(context.Background(), left, right)
	if err != nil {
		t.Fatal(err)
	}
	artifact, err := (DefaultArtifactRenderer[struct{}]{}).Render(
		context.Background(),
		UnrestrictedRead(),
		fused,
		ArtifactRenderOptions[struct{}]{CloneMeta: cloneArtifactValue[struct{}]},
	)
	docs := fused.Documents()
	docs[0].ScoreHistory[0].Value = 999
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	for _, snippet := range artifact.Snippets {
		if len(snippet.ScoreHistory) != 1 {
			t.Fatalf("raw observation lost: %+v", snippet)
		}
		want := 2.0
		if snippet.DocumentID == "dense-doc" {
			want = -0.5
		}
		if snippet.ScoreHistory[0].Value != want {
			t.Fatalf("transformed/clamped/aliased raw score: %+v", snippet)
		}
	}
	for _, doc := range fused.Documents() {
		if doc.ScoreHistory[0].Value == 999 {
			t.Fatal("mutable history escaped ResultSet")
		}
	}
}

func TestDedupKeepsLosingScoreObservation(t *testing.T) {
	// Arrange.
	left := nativeFixture("same", 2, tensor.MaxSimSemantics)
	right := nativeFixture("same", -1, tensor.MaxSimSemantics)
	rs := NewResultSet([]Document[struct{}]{left, right}, nil)
	// Act.
	deduped, err := rs.Dedup()
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	docs := deduped.Documents()
	if len(docs) != 1 || docs[0].Score != 2 || len(docs[0].ScoreHistory) != 1 || docs[0].ScoreHistory[0].Value != -1 {
		t.Fatalf("losing input evidence discarded: %+v", docs)
	}
}

func TestExplicitComparatorRankSurvivesTerminalTopK(t *testing.T) {
	// Arrange: incompatible scales can only be ordered by a supplied host policy.
	rs := NewResultSet(
		[]Document[struct{}]{nativeFixture("a", 2, tensor.MaxSimSemantics), nativeFixture("b", -0.5, "dense.cosine")},
		nil,
	)
	processor := Rerank(func(a, b Document[struct{}]) bool { return a.ID > b.ID })
	chain := NewPostProcessorChain[struct{}](processor)
	// Act.
	ranked, err := chain.Process(context.Background(), UnrestrictedRead(), RetrieveOptions{TopK: 1}, rs)
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	docs := ranked.Documents()
	if len(docs) != 1 || docs[0].ID != "b" || docs[0].ScoreState != ScoreAbsent || docs[0].Rank != 1 ||
		len(docs[0].ScoreHistory) != 1 ||
		docs[0].ScoreHistory[0].Value != -0.5 {
		t.Fatalf("explicit ordering lost: %+v", docs)
	}
}
