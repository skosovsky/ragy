package tensor_test

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/tensor"
)

func fixtureSpace() tensor.Space {
	return tensor.Space{
		Model:         "fixture-model",
		ModelRevision: "fixture-revision",
		Configuration: "normalized-tokens",
		VectorSpace:   "fixture-dot",
		Dimension:     2,
	}
}
func embedding(tokens tensor.Tensor) tensor.Embedding {
	return tensor.Embedding{Space: fixtureSpace(), Tokens: tokens}
}

func TestMaxSimReferenceFixture(t *testing.T) {
	// Arrange: load the specification's actual fixture, rather than duplicating it.
	data, err := os.ReadFile("../docs/task12/fixtures/tensor_maxsim.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Query     tensor.Tensor            `json:"query"`
		Documents map[string]tensor.Tensor `json:"documents"`
		Scores    map[string]float64       `json:"expected_scores"`
		Ranking   []string                 `json:"expected_ranking"`
	}
	if err = json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	candidates := make([]tensor.Candidate, 0, len(fixture.Ranking))
	for _, id := range fixture.Ranking {
		candidates = append(candidates, tensor.Candidate{ID: id, Embedding: embedding(fixture.Documents[id])})
	}
	// Act.
	result, err := tensor.Rerank(
		context.Background(),
		embedding(fixture.Query),
		candidates,
		tensor.RerankOptions{CandidateBudget: 100, TopK: 10},
	)
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]string, len(result.Ranking))
	for i, hit := range result.Ranking {
		ids[i] = hit.ID
		if hit.Score != fixture.Scores[hit.ID] || hit.Semantics != tensor.MaxSimSemantics || hit.Rank != i+1 {
			t.Fatalf("wrong native score evidence: %+v", hit)
		}
	}
	if !reflect.DeepEqual(ids, fixture.Ranking) {
		t.Fatalf("ranking %v, want %v", ids, fixture.Ranking)
	}
	for _, candidate := range candidates {
		score, err := tensor.MaxSim(context.Background(), embedding(fixture.Query), candidate.Embedding)
		if err != nil || score != fixture.Scores[candidate.ID] {
			t.Fatalf("oracle %s = %v, %v", candidate.ID, score, err)
		}
	}
}

func TestRerankMissingCandidateAndEvidenceOwnership(t *testing.T) {
	// Arrange: the best document is absent from the supplied candidate universe.
	candidates := []tensor.Candidate{
		{ID: "t2", Embedding: embedding(tensor.Tensor{{1, 0}})},
		{ID: "t3", Embedding: embedding(tensor.Tensor{{-1, 0}})},
	}
	// Act.
	result, err := tensor.Rerank(
		context.Background(),
		embedding(tensor.Tensor{{1, 0}, {0, 1}}),
		candidates,
		tensor.RerankOptions{CandidateBudget: 100, TopK: 1},
	)
	candidates[0].ID = "changed"
	candidates[0].Embedding.Tokens[0][0] = -1
	// Assert: output reports both input IDs, even when TopK keeps just one hit.
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.CandidateIDs, []string{"t2", "t3"}) || len(result.Ranking) != 1 ||
		result.Ranking[0].ID != "t2" ||
		result.Ranking[0].Score != 1 {
		t.Fatalf("lost candidate evidence: %+v", result)
	}
}

func TestEmbeddingValidation(t *testing.T) {
	cases := []struct {
		name   string
		tokens tensor.Tensor
	}{
		{"empty", nil}, {"empty-token", tensor.Tensor{{}}}, {"ragged", tensor.Tensor{{1, 0}, {1}}},
		{"nan", tensor.Tensor{{float32(math.NaN()), 0}}}, {"infinity", tensor.Tensor{{0, float32(math.Inf(1))}}},
		{"zero", tensor.Tensor{{0, 0}}}, {"unnormalized", tensor.Tensor{{2, 0}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Arrange.
			doc := embedding(tc.tokens)
			// Act.
			_, err := tensor.MaxSim(context.Background(), embedding(tensor.Tensor{{1, 0}}), doc)
			// Assert.
			if err == nil {
				t.Fatal("invalid tensor accepted")
			}
		})
	}
}

func TestSpaceMismatch(t *testing.T) {
	for _, field := range []string{"model", "revision", "configuration", "space", "dimension"} {
		t.Run(field, func(t *testing.T) {
			// Arrange.
			doc := embedding(tensor.Tensor{{1, 0}})
			switch field {
			case "model":
				doc.Space.Model = "other"
			case "revision":
				doc.Space.ModelRevision = "other"
			case "configuration":
				doc.Space.Configuration = "other"
			case "space":
				doc.Space.VectorSpace = "other"
			case "dimension":
				doc.Space.Dimension = 3
				doc.Tokens = tensor.Tensor{{1, 0, 0}}
			}
			// Act.
			_, err := tensor.MaxSim(context.Background(), embedding(tensor.Tensor{{1, 0}}), doc)
			// Assert.
			if !errors.Is(err, ragy.ErrInvalidArgument) {
				t.Fatalf("mismatched %s accepted: %v", field, err)
			}
		})
	}
}

func TestRerankRejectsLimitsDuplicatesAndCancellation(t *testing.T) {
	query := embedding(tensor.Tensor{{1, 0}})
	candidate := tensor.Candidate{ID: "d1", Embedding: query}
	for _, opts := range []tensor.RerankOptions{{}, {CandidateBudget: 1, TopK: 2}, {CandidateBudget: -1, TopK: 1}} {
		_, err := tensor.Rerank(context.Background(), query, []tensor.Candidate{candidate}, opts)
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("limits accepted: %+v, %v", opts, err)
		}
	}
	_, err := tensor.Rerank(
		context.Background(),
		query,
		[]tensor.Candidate{candidate, candidate},
		tensor.RerankOptions{CandidateBudget: 2, TopK: 1},
	)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("duplicate accepted: %v", err)
	}
	other := candidate
	other.ID = "d2"
	_, err = tensor.Rerank(
		context.Background(),
		query,
		[]tensor.Candidate{candidate, other},
		tensor.RerankOptions{CandidateBudget: 1, TopK: 1},
	)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("overflow silently truncated: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, err := tensor.Rerank(
		ctx,
		query,
		[]tensor.Candidate{candidate},
		tensor.RerankOptions{CandidateBudget: 1, TopK: 1},
	)
	if !errors.Is(err, context.Canceled) || len(result.Ranking) != 0 {
		t.Fatalf("cancelled result: %+v, %v", result, err)
	}
}

func TestTensorRecordRequiresExplicitValidatedSpace(t *testing.T) {
	record := tensor.Record[struct{}]{ID: "d1", Tensor: tensor.Tensor{{1, 0}}}
	if err := record.Validate(); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("missing space accepted: %v", err)
	}
	record.Space = fixtureSpace()
	if err := record.Validate(); err != nil {
		t.Fatal(err)
	}
	record.Tensor = tensor.Tensor{{1, 0}, {1}}
	if err := record.Validate(); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("ragged write accepted: %v", err)
	}
}
