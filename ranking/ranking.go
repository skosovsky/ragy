// Package ranking provides query-aware reranking and list-merging contracts.
package ranking

import (
	"context"
	"fmt"
	"sort"

	ragy "github.com/skosovsky/ragy"
)

// QueryReranker reranks documents using query-aware scoring.
type QueryReranker interface {
	Rerank(ctx context.Context, query string, docs []ragy.Document) ([]ragy.Document, error)
}

// Merger merges already-ranked lists.
type Merger interface {
	Merge(ctx context.Context, lists ...[]ragy.Document) ([]ragy.Document, error)
}

// ReciprocalRankFusion merges ranked lists with RRF.
type ReciprocalRankFusion struct {
	k int
}

type fusedState struct {
	doc   ragy.Document
	score float64
}

// NewReciprocalRankFusion constructs an RRF merger.
func NewReciprocalRankFusion(k int) (*ReciprocalRankFusion, error) {
	if k <= 0 {
		return nil, fmt.Errorf("%w: RRF k must be > 0", ragy.ErrInvalidArgument)
	}

	return &ReciprocalRankFusion{k: k}, nil
}

// Merge merges ranked lists by stable document ID.
func (r *ReciprocalRankFusion) Merge(ctx context.Context, lists ...[]ragy.Document) ([]ragy.Document, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(lists) == 0 {
		return nil, nil
	}

	seen, mergeErr := r.mergeLists(ctx, lists...)
	if mergeErr != nil {
		return nil, mergeErr
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(seen) == 0 {
		return nil, nil
	}

	maxScore, scoreErr := maxMergedScore(ctx, seen)
	if scoreErr != nil {
		return nil, scoreErr
	}
	out, buildErr := buildMergedDocuments(ctx, seen, maxScore)
	if buildErr != nil {
		return nil, buildErr
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Relevance == out[j].Relevance {
			return out[i].ID < out[j].ID
		}

		return out[i].Relevance > out[j].Relevance
	})
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	return out, nil
}

func (r *ReciprocalRankFusion) mergeLists(
	ctx context.Context,
	lists ...[]ragy.Document,
) (map[string]fusedState, error) {
	seen := make(map[string]fusedState)
	for _, list := range lists {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := r.mergeList(ctx, seen, list); err != nil {
			return nil, err
		}
	}

	return seen, nil
}

func (r *ReciprocalRankFusion) mergeList(ctx context.Context, seen map[string]fusedState, list []ragy.Document) error {
	for rank, doc := range list {
		if err := ctx.Err(); err != nil {
			return err
		}

		normalized, err := ragy.NormalizeDocument(doc)
		if err != nil {
			return err
		}
		doc = normalized

		current := seen[doc.ID]
		if current.doc.ID == "" {
			current.doc = doc
		} else {
			same, err := samePayload(current.doc, doc)
			if err != nil {
				return err
			}
			if !same {
				return fmt.Errorf("%w: conflicting payload for document %q", ragy.ErrInvalidArgument, doc.ID)
			}
		}

		if err := ctx.Err(); err != nil {
			return err
		}
		current.score += 1.0 / float64(r.k+rank+1)
		seen[doc.ID] = current
	}

	return nil
}

func maxMergedScore(ctx context.Context, seen map[string]fusedState) (float64, error) {
	maxScore := 0.0
	for _, item := range seen {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		if item.score > maxScore {
			maxScore = item.score
		}
	}

	return maxScore, nil
}

func buildMergedDocuments(
	ctx context.Context,
	seen map[string]fusedState,
	maxScore float64,
) ([]ragy.Document, error) {
	out := make([]ragy.Document, 0, len(seen))
	for _, item := range seen {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		doc := item.doc
		if maxScore > 0 {
			doc.Relevance = ragy.ClampRelevance(item.score / maxScore)
		}
		out = append(out, doc)
	}

	return out, nil
}

func samePayload(left, right ragy.Document) (bool, error) {
	if left.Content != right.Content {
		return false, nil
	}

	leftAttrs, err := ragy.NormalizeAttributes(left.Attributes)
	if err != nil {
		return false, err
	}

	rightAttrs, err := ragy.NormalizeAttributes(right.Attributes)
	if err != nil {
		return false, err
	}

	return equalAttributes(leftAttrs, rightAttrs), nil
}

func equalAttributes(left, right ragy.Attributes) bool {
	if len(left) == 0 && len(right) == 0 {
		return true
	}
	if len(left) != len(right) {
		return false
	}

	for key, leftValue := range left {
		rightValue, ok := right[key]
		if !ok {
			return false
		}
		if leftValue != rightValue {
			return false
		}
	}

	return true
}
