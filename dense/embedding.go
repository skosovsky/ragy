package dense

import (
	"context"
	"math"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
)

// Space declares a compatible dense embedding profile. Equal dimensions alone
// do not establish compatibility; configuration includes preprocessing identity.
type Space struct {
	Model         string `json:"model"`
	ModelRevision string `json:"model_revision"`
	Configuration string `json:"configuration"`
	VectorSpace   string `json:"vector_space"`
	Dimension     int    `json:"dimension"`
}

func (s Space) Validate() error {
	if s.Dimension <= 0 {
		return ragy.ErrInvalidArgument
	}
	for _, value := range []string{s.Model, s.ModelRevision, s.Configuration, s.VectorSpace} {
		if value == "" || !utf8.ValidString(value) {
			return ragy.ErrInvalidArgument
		}
	}
	return nil
}

// Embedding is a normalized dense vector with explicit model/configuration identity.
// It is separate from a token matrix and is never silently normalized.
const unitNormTolerance = 1e-5

type Embedding struct {
	Space  Space
	Vector []float32
}

func (e Embedding) Validate() error {
	if err := e.Space.Validate(); err != nil {
		return err
	}
	if len(e.Vector) == 0 {
		return ragy.ErrEmptyVector
	}
	if len(e.Vector) != e.Space.Dimension {
		return ragy.ErrInvalidArgument
	}
	var norm float64
	for _, component := range e.Vector {
		value := float64(component)
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return ragy.ErrInvalidArgument
		}
		norm += value * value
	}
	if math.Abs(norm-1) > unitNormTolerance {
		return ragy.ErrInvalidArgument
	}
	return nil
}

const NormalizedDotSemantics = "dense.normalized-dot"

// NormalizedDot computes native similarity within one compatible normalized profile.
func NormalizedDot(ctx context.Context, query, document Embedding) (float64, error) {
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
	var score float64
	for i, value := range query.Vector {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		score += float64(value) * float64(document.Vector[i])
	}
	return score, ctx.Err()
}
