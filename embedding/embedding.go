// Package embedding defines provider-neutral encoding identity and accounting.
// Hosts own model revisions, preprocessing configuration and pricing.
package embedding

import (
	"math"
	"strings"
	"time"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
)

type Metric string

const (
	NormalizedDot Metric = "normalized-dot"
	Dot           Metric = "dot"
	Cosine        Metric = "cosine"
	SquaredL2     Metric = "negative-squared-l2"
)

// Space declares host-attested encoder identity and explicit scoring semantics.
// ModelRevision and Configuration are not verified remote facts.
type Space struct {
	Model         string `json:"model"`
	ModelRevision string `json:"model_revision"`
	Configuration string `json:"configuration"`
	VectorSpace   string `json:"vector_space"`
	Dimension     int    `json:"dimension"`
	Metric        Metric `json:"metric"`
}

func (s Space) Validate() error {
	if s.Dimension <= 0 {
		return ragy.ErrInvalidArgument
	}
	for _, v := range []string{s.Model, s.ModelRevision, s.Configuration, s.VectorSpace} {
		if strings.TrimSpace(v) == "" || !utf8.ValidString(v) {
			return ragy.ErrInvalidArgument
		}
	}
	switch s.Metric {
	case NormalizedDot, Dot, Cosine, SquaredL2:
		return nil
	default:
		return ragy.ErrUnsupported
	}
}
func (s Space) ValidateVector(vector []float32) error {
	if err := s.Validate(); err != nil {
		return err
	}
	if len(vector) == 0 {
		return ragy.ErrEmptyVector
	}
	if len(vector) != s.Dimension {
		return ragy.ErrInvalidArgument
	}
	var norm float64
	for _, v := range vector {
		f := float64(v)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return ragy.ErrInvalidArgument
		}
		norm += f * f
	}
	if s.Metric == NormalizedDot && math.Abs(norm-1) > 1e-5 {
		return ragy.ErrInvalidArgument
	}
	if s.Metric == Cosine && norm == 0 {
		return ragy.ErrInvalidArgument
	}
	return nil
}

type Purpose string

const (
	Query      Purpose = "query"
	Document   Purpose = "document"
	Similarity Purpose = "similarity"
)

func (p Purpose) Validate() error {
	switch p {
	case Query, Document, Similarity:
		return nil
	default:
		return ragy.ErrUnsupported
	}
}

// Request has ordered immutable host inputs. RequireRemoteTokenBound requires
// the encoder to enforce a hard remote token cap before dispatch. Encoders without
// that capability reject it; the core does not forbid capable host encoders.
type Request[T any] struct {
	Inputs                  []T
	Purpose                 Purpose
	RequireRemoteTokenBound bool
}

func (r Request[T]) Validate() error {
	if len(r.Inputs) == 0 {
		return ragy.ErrEmptyText
	}
	if err := r.Purpose.Validate(); err != nil {
		return err
	}
	return nil
}

// Usage never substitutes estimates for unavailable provider counters.
type Usage struct {
	InputTokens      int64
	InputTokensKnown bool
	BilledUnits      int64
	BilledUnitsKnown bool
}

func (u Usage) Validate() error {
	if u.InputTokens < 0 || u.BilledUnits < 0 || (!u.InputTokensKnown && u.InputTokens != 0) ||
		(!u.BilledUnitsKnown && u.BilledUnits != 0) {
		return ragy.ErrProtocol
	}
	return nil
}

// Result is ordered like Request.Inputs on success; payload ownership transfers
// to caller. Providers may return empty embeddings, observed Usage and an error
// after rejected materialization. Callers must inspect the error before use.
type Result[T any] struct {
	Embeddings []T
	Usage      Usage
}

const (
	defaultMaxInputs        = 128
	defaultMaxInputBytes    = 1 << 20
	defaultMaxRequestBytes  = 2 << 20
	defaultMaxResponseBytes = 16 << 20
	defaultMaxVectorRows    = 8192
	defaultTimeout          = 30 * time.Second
)

// Limits bound local work, not remote tokenization or price. Zero fields select
// finite defaults. MaxVectorRows bounds rows per returned token matrix.
type Limits struct {
	MaxInputs        int
	MaxInputBytes    int
	MaxRequestBytes  int64
	MaxResponseBytes int64
	MaxVectorRows    int
	Timeout          time.Duration
}

func (l Limits) Resolve() (Limits, error) {
	if l.MaxInputs < 0 || l.MaxInputBytes < 0 || l.MaxRequestBytes < 0 || l.MaxResponseBytes < 0 ||
		l.MaxVectorRows < 0 ||
		l.Timeout < 0 {
		return Limits{}, ragy.ErrInvalidArgument
	}
	if l.MaxInputs == 0 {
		l.MaxInputs = defaultMaxInputs
	}
	if l.MaxInputBytes == 0 {
		l.MaxInputBytes = defaultMaxInputBytes
	}
	if l.MaxRequestBytes == 0 {
		l.MaxRequestBytes = defaultMaxRequestBytes
	}
	if l.MaxResponseBytes == 0 {
		l.MaxResponseBytes = defaultMaxResponseBytes
	}
	if l.MaxVectorRows == 0 {
		l.MaxVectorRows = defaultMaxVectorRows
	}
	if l.Timeout == 0 {
		l.Timeout = defaultTimeout
	}
	if l.MaxRequestBytes == math.MaxInt64 || l.MaxResponseBytes == math.MaxInt64 {
		return Limits{}, ragy.ErrInvalidArgument
	}
	return l, nil
}
