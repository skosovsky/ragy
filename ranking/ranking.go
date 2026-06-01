// Package ranking provides query-aware reranking and list-merging contracts.
package ranking

import (
	"context"
	"fmt"
	"reflect"
	"sort"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/retrieval"
)

// QueryReranker reranks documents using query-aware scoring.
type QueryReranker[TMeta any] interface {
	Rerank(ctx context.Context, query string, docs []retrieval.Document[TMeta]) ([]retrieval.Document[TMeta], error)
}

// Merger merges already-ranked lists.
type Merger[TMeta any] interface {
	Merge(ctx context.Context, lists ...[]retrieval.Document[TMeta]) ([]retrieval.Document[TMeta], error)
}

// ReciprocalRankFusion merges ranked lists with RRF.
type ReciprocalRankFusion[TMeta any] struct {
	k int
}

type fusedState[TMeta any] struct {
	doc   retrieval.Document[TMeta]
	score float64
}

// NewReciprocalRankFusion constructs an RRF merger.
func NewReciprocalRankFusion[TMeta any](k int) (*ReciprocalRankFusion[TMeta], error) {
	if k <= 0 {
		return nil, fmt.Errorf("%w: RRF k must be > 0", ragy.ErrInvalidArgument)
	}

	return &ReciprocalRankFusion[TMeta]{k: k}, nil
}

// Merge merges ranked lists by stable document ID.
func (r *ReciprocalRankFusion[TMeta]) Merge(
	ctx context.Context,
	lists ...[]retrieval.Document[TMeta],
) ([]retrieval.Document[TMeta], error) {
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
		if out[i].Score == out[j].Score {
			return out[i].ID < out[j].ID
		}

		return out[i].Score > out[j].Score
	})
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	return out, nil
}

func (r *ReciprocalRankFusion[TMeta]) mergeLists(
	ctx context.Context,
	lists ...[]retrieval.Document[TMeta],
) (map[string]fusedState[TMeta], error) {
	seen := make(map[string]fusedState[TMeta])
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

func (r *ReciprocalRankFusion[TMeta]) mergeList(
	ctx context.Context,
	seen map[string]fusedState[TMeta],
	list []retrieval.Document[TMeta],
) error {
	for rank, doc := range list {
		if err := ctx.Err(); err != nil {
			return err
		}

		if err := retrieval.ValidateDocument(doc); err != nil {
			return err
		}

		current := seen[doc.ID]
		if current.doc.ID == "" {
			current.doc = doc
		} else if !samePayload(current.doc, doc) {
			return fmt.Errorf("%w: conflicting payload for document %q", ragy.ErrInvalidArgument, doc.ID)
		}

		if err := ctx.Err(); err != nil {
			return err
		}
		current.score += 1.0 / float64(r.k+rank+1)
		seen[doc.ID] = current
	}

	return nil
}

func maxMergedScore[TMeta any](ctx context.Context, seen map[string]fusedState[TMeta]) (float64, error) {
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

func buildMergedDocuments[TMeta any](
	ctx context.Context,
	seen map[string]fusedState[TMeta],
	maxScore float64,
) ([]retrieval.Document[TMeta], error) {
	out := make([]retrieval.Document[TMeta], 0, len(seen))
	for _, item := range seen {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		doc := item.doc
		if maxScore > 0 {
			doc.Score = ragy.ClampScore(item.score / maxScore)
		}
		out = append(out, doc)
	}

	return out, nil
}

func samePayload[TMeta any](left, right retrieval.Document[TMeta]) bool {
	return left.Content == right.Content && reflect.DeepEqual(left.Meta, right.Meta)
}
