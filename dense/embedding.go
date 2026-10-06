package dense

import (
	"context"
	"math"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/embedding"
)

// Space identifies compatible vectors including their implemented metric.
type Space = embedding.Space

// Embedding is a vector with explicit identity; it is never normalized implicitly.
type Embedding struct {
	Space  Space
	Vector []float32
}

func (e Embedding) Validate() error { return e.Space.ValidateVector(e.Vector) }

const NormalizedDotSemantics = "dense.normalized-dot"

func ScoreSemantics(metric embedding.Metric) string { return "dense." + string(metric) }

// Similarity computes the space's declared metric. Larger scores are better.
func Similarity(ctx context.Context, query, document Embedding) (float64, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if err := query.Validate(); err != nil {
		return 0, err
	}
	if err := document.Validate(); err != nil {
		return 0, err
	}
	if query.Space != document.Space {
		return 0, ragy.ErrInvalidArgument
	}
	var dot, left, right, distance float64
	for i, v := range query.Vector {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		a, b := float64(v), float64(document.Vector[i])
		dot += a * b
		left += a * a
		right += b * b
		delta := a - b
		distance += delta * delta
	}
	switch query.Space.Metric {
	case embedding.SquaredL2:
		return -distance, ctx.Err()
	case embedding.Cosine:
		return math.Max(-1, math.Min(1, dot/math.Sqrt(left*right))), ctx.Err()
	case embedding.NormalizedDot, embedding.Dot:
		return dot, ctx.Err()
	default:
		return 0, ragy.ErrUnsupported
	}
}

// NormalizedDot requires the explicit normalized-dot profile.
func NormalizedDot(ctx context.Context, query, document Embedding) (float64, error) {
	if query.Space.Metric != embedding.NormalizedDot {
		return 0, ragy.ErrUnsupported
	}
	return Similarity(ctx, query, document)
}
