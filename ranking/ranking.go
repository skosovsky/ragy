// Package ranking provides thin type aliases over retrieval merge and rerank primitives.
// Prefer importing github.com/skosovsky/ragy/retrieval directly in new code.
package ranking

import (
	"context"

	"github.com/skosovsky/ragy/retrieval"
)

// QueryReranker reranks documents using query-aware scoring.
type QueryReranker[TMeta any] interface {
	Rerank(ctx context.Context, query string, rs retrieval.ResultSet[TMeta]) (retrieval.ResultSet[TMeta], error)
}

// Merger merges already-ranked result sets.
type Merger[TMeta any] = retrieval.ResultMerger[TMeta]

// ReciprocalRankFusion merges ranked lists with RRF.
type ReciprocalRankFusion[TMeta any] = retrieval.ReciprocalRankFusion[TMeta]

// ScoreMerger merges result sets by max Score per MergeKey.
type ScoreMerger[TMeta any] = retrieval.ScoreMerger[TMeta]

// NewReciprocalRankFusion constructs an RRF merger.
func NewReciprocalRankFusion[TMeta any](
	k int,
	resolver retrieval.IdentityResolver[TMeta],
) (*ReciprocalRankFusion[TMeta], error) {
	return retrieval.NewReciprocalRankFusion(k, resolver)
}

// NewScoreMerger constructs a score-based merger for homogeneous ranked lists.
func NewScoreMerger[TMeta any](resolver retrieval.IdentityResolver[TMeta]) *ScoreMerger[TMeta] {
	return retrieval.NewScoreMerger(resolver)
}
