package dense_test

import (
	"context"
	"errors"
	"math"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/embedding"
)

func TestExplicitMetricSimilarity(t *testing.T) {
	for _, tc := range []struct {
		metric      embedding.Metric
		left, right []float32
		want        float64
	}{
		{embedding.NormalizedDot, []float32{1, 0}, []float32{0, 1}, 0},
		{embedding.Dot, []float32{2, 0}, []float32{3, 0}, 6},
		{embedding.Cosine, []float32{2, 0}, []float32{3, 0}, 1},
		{embedding.SquaredL2, []float32{2, 0}, []float32{3, 0}, -1},
	} {
		t.Run(string(tc.metric), func(t *testing.T) {
			// Arrange: identical shape with explicit metric and unchanged input magnitude.
			space := dense.Space{
				Model:         "fixture",
				ModelRevision: "r1",
				Configuration: "c",
				VectorSpace:   "v",
				Dimension:     2,
				Metric:        tc.metric,
			}
			query, doc := dense.Embedding{
				Space:  space,
				Vector: tc.left,
			}, dense.Embedding{
				Space:  space,
				Vector: tc.right,
			}
			// Act.
			score, err := dense.Similarity(context.Background(), query, doc)
			// Assert.
			if err != nil || score != tc.want {
				t.Fatalf("score=%v err=%v want=%v", score, err, tc.want)
			}
			doc.Space.Configuration = "different"
			if _, err = dense.Similarity(context.Background(), query, doc); !errors.Is(err, ragy.ErrInvalidArgument) {
				t.Fatalf("incompatible space accepted: %v", err)
			}
		})
	}
}
func TestVectorProfileValidation(t *testing.T) {
	// Arrange.
	space := dense.Space{
		Model:         "m",
		ModelRevision: "r",
		Configuration: "c",
		VectorSpace:   "v",
		Dimension:     2,
		Metric:        embedding.Dot,
	}
	cases := [][]float32{{1}, {1, float32(math.NaN())}, {float32(math.Inf(1)), 0}}
	// Act/Assert.
	for _, vector := range cases {
		if (dense.Embedding{Space: space, Vector: vector}).Validate() == nil {
			t.Fatal("invalid vector accepted")
		}
	}
	space.Metric = embedding.Cosine
	if (dense.Embedding{Space: space, Vector: []float32{0, 0}}).Validate() == nil {
		t.Fatal("undefined cosine accepted")
	}
	space.Metric = embedding.NormalizedDot
	if (dense.Embedding{Space: space, Vector: []float32{2, 0}}).Validate() == nil {
		t.Fatal("unnormalized normalized-dot accepted")
	}
}
