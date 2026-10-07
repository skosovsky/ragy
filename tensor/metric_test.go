package tensor_test

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	encoding "github.com/skosovsky/ragy/embedding"
	"github.com/skosovsky/ragy/tensor"
)

func TestRawDotMaxSimDeclaresActualSemantics(t *testing.T) {
	// Arrange: nonunit token magnitudes are intentional in the raw-dot profile.
	space := fixtureSpace()
	space.Metric = encoding.Dot
	query := tensor.Embedding{Space: space, Tokens: tensor.Tensor{{2, 0}, {0, 3}}}
	document := tensor.Embedding{Space: space, Tokens: tensor.Tensor{{4, 0}, {0, 5}}}
	// Act.
	result, err := tensor.Rerank(
		context.Background(),
		query,
		[]tensor.Candidate{{ID: "d", Embedding: document}},
		tensor.RerankOptions{CandidateBudget: 1, TopK: 1},
	)
	// Assert.
	if err != nil || len(result.Ranking) != 1 || result.Ranking[0].Score != 23 ||
		result.Ranking[0].Semantics != "tensor.maxsim.dot.sum" {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	query.Space.Metric = encoding.Cosine
	if _, err = tensor.MaxSim(context.Background(), query, document); !errors.Is(err, ragy.ErrUnsupported) {
		t.Fatalf("unsupported metric accepted: %v", err)
	}
}

func TestNativeMaxSimNegativeScoresAreNotClamped(t *testing.T) {
	for _, metric := range []encoding.Metric{encoding.Dot, encoding.NormalizedDot} {
		t.Run(string(metric), func(t *testing.T) {
			// Arrange: all document-token similarities are negative.
			space := fixtureSpace()
			space.Metric = metric
			query := tensor.Embedding{Space: space, Tokens: tensor.Tensor{{1, 0}, {1, 0}}}
			document := tensor.Embedding{Space: space, Tokens: tensor.Tensor{{-1, 0}}}
			// Act.
			score, err := tensor.MaxSim(t.Context(), query, document)
			// Assert: sum of two -1 maxima stays -2 in both native metrics.
			if err != nil || score != -2 {
				t.Fatal("clamped native score", score, err)
			}
		})
	}
}
