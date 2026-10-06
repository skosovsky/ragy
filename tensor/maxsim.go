package tensor

import (
	"context"
	"fmt"
	"math"
	"sort"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/embedding"
)

// MaxSimSemantics identifies sum-of-query-token maximum dot-product scores.
// Scores are native: they may be negative or greater than one.
const MaxSimSemantics = "tensor.maxsim.normalized-dot.sum"

type Space = embedding.Space

func MaxSimSemanticsFor(metric embedding.Metric) string {
	return "tensor.maxsim." + string(metric) + ".sum"
}

// ValidateSpace rejects metrics whose MaxSim interpretation is not implemented.
func ValidateSpace(space Space) error {
	if err := space.Validate(); err != nil {
		return err
	}
	if space.Metric != embedding.Dot && space.Metric != embedding.NormalizedDot {
		return ragy.ErrUnsupported
	}
	return nil
}

// Embedding pairs a normalized token matrix with its declared space.
// The caller owns Tokens and must not mutate it during an operation.
type Embedding struct {
	Space  Space
	Tokens Tensor
}

// Validate checks every component and token shape. It never silently normalizes.
func (e Embedding) Validate() error {
	if len(e.Tokens) == 0 {
		return fmt.Errorf("%w: tensor tokens", ragy.ErrEmptyVector)
	}
	if err := ValidateSpace(e.Space); err != nil {
		return err
	}
	for _, token := range e.Tokens {
		if len(token) != e.Space.Dimension {
			return ragy.ErrInvalidArgument
		}
		if err := e.Space.ValidateVector(token); err != nil {
			return err
		}
	}

	return nil
}

// MaxSim computes the reference native score. Validation precedes computation;
// cancellation returns no usable score. Arithmetic accumulates in float64.
func MaxSim(ctx context.Context, query, document Embedding) (float64, error) {
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
		return 0, fmt.Errorf("%w: incompatible tensor spaces", ragy.ErrInvalidArgument)
	}
	return maxSimValidated(ctx, query.Tokens, document.Tokens)
}

func maxSimValidated(ctx context.Context, query, document Tensor) (float64, error) {
	var score float64
	for _, q := range query {
		best := math.Inf(-1)
		for _, d := range document {
			if err := ctx.Err(); err != nil {
				return 0, err
			}
			var dot float64
			for i, component := range q {
				dot += float64(component) * float64(d[i])
			}
			best = math.Max(best, dot)
		}
		score += best
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	return score, nil
}

// Candidate is a supplied candidate, not a request to scan an entire index.
type Candidate struct {
	ID        string
	Embedding Embedding
}

// ScoredCandidate preserves native MaxSim evidence and rank within candidates.
type ScoredCandidate struct {
	ID        string
	Score     float64
	Semantics string
	Rank      int
}

// RerankOptions requires explicit candidate and output limits.
type RerankOptions struct {
	CandidateBudget int
	TopK            int
}

// RerankResult describes only the supplied candidate universe. It makes no
// assertion that candidate generation included every relevant indexed document.
type RerankResult struct {
	Ranking         []ScoredCandidate
	CandidateIDs    []string
	CandidateBudget int
}

// Rerank computes exact MaxSim within a bounded caller-supplied candidate set.
// All shapes, identities and limits are checked before any scoring. Overflow is
// rejected rather than truncating the candidate set and concealing lost recall.
func Rerank(ctx context.Context, query Embedding, candidates []Candidate, options RerankOptions) (RerankResult, error) {
	if err := ctx.Err(); err != nil {
		return RerankResult{}, err
	}
	if options.CandidateBudget <= 0 || options.TopK <= 0 || options.TopK > options.CandidateBudget ||
		len(candidates) > options.CandidateBudget {
		return RerankResult{}, fmt.Errorf("%w: tensor candidate/output limits", ragy.ErrInvalidArgument)
	}
	if err := query.Validate(); err != nil {
		return RerankResult{}, err
	}
	seen := make(map[string]struct{}, len(candidates))
	for _, candidate := range candidates {
		if candidate.ID == "" {
			return RerankResult{}, ragy.ErrMissingID
		}
		if _, exists := seen[candidate.ID]; exists {
			return RerankResult{}, fmt.Errorf("%w: duplicate tensor candidate", ragy.ErrInvalidArgument)
		}
		seen[candidate.ID] = struct{}{}
		if err := candidate.Embedding.Validate(); err != nil {
			return RerankResult{}, err
		}
		if candidate.Embedding.Space != query.Space {
			return RerankResult{}, fmt.Errorf("%w: incompatible tensor candidate space", ragy.ErrInvalidArgument)
		}
	}
	result := RerankResult{Ranking: nil, CandidateIDs: nil, CandidateBudget: options.CandidateBudget}
	for _, candidate := range candidates {
		score, err := maxSimValidated(ctx, query.Tokens, candidate.Embedding.Tokens)
		if err != nil {
			return RerankResult{}, err
		}
		result.CandidateIDs = append(result.CandidateIDs, candidate.ID)
		result.Ranking = append(
			result.Ranking,
			ScoredCandidate{ID: candidate.ID, Score: score, Semantics: MaxSimSemanticsFor(query.Space.Metric), Rank: 0},
		)
	}
	sort.SliceStable(result.Ranking, func(i, j int) bool {
		return result.Ranking[i].Score > result.Ranking[j].Score
	})
	if len(result.Ranking) > options.TopK {
		result.Ranking = result.Ranking[:options.TopK]
	}
	for i := range result.Ranking {
		result.Ranking[i].Rank = i + 1
	}
	if err := ctx.Err(); err != nil {
		return RerankResult{}, err
	}
	return result, nil
}
