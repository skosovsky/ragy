package bridge

import (
	"cmp"
	"context"
	"slices"

	"github.com/skosovsky/memy"

	"github.com/skosovsky/ragy/retrieval"
)

type mapped[M any] struct {
	ref     Reference
	doc     retrieval.Document[M]
	ordinal int
}
type search[M any] struct {
	scope     memy.Scope
	items     []mapped[M]
	coverage  []memy.Coverage
	truncated bool
}

func (search[M]) Capabilities() memy.SearchCapabilities {
	return memy.SearchCapabilities{Scoped: true, BoundedCandidates: true}
}

func (s search[M]) Search(
	ctx context.Context,
	scope memy.Scope,
	_ string,
	opts memy.SearchOptions,
) (memy.SearchResult, error) {
	if err := ctx.Err(); err != nil {
		return memy.SearchResult{}, err
	}
	if scope != s.scope {
		return memy.SearchResult{}, memy.ErrScopeViolation
	}
	if opts.Minimum != nil {
		return memy.SearchResult{}, memy.ErrUnsupported
	}
	if opts.MaxCandidates < 1 {
		return memy.SearchResult{}, memy.ErrInvalid
	}
	out := memy.SearchResult{Coverage: slices.Clone(s.coverage), CandidatesTruncated: s.truncated}
	for i, item := range s.items {
		if i >= opts.MaxCandidates {
			out.CandidatesTruncated = true
			break
		}
		rank := i + 1
		score := 1 / float64(rank)
		out.Candidates = append(out.Candidates, memy.Candidate{
			RecordID: item.ref.RecordID,
			Revision: item.ref.Revision,
			Score: memy.ScoreOf(
				score,
			),
			Signals: []memy.SearchSignal{
				{
					Backend: "retrieval",
					Rank:    item.ordinal,
					Score:   memy.Score{Present: item.doc.ScoreState.IsScored(), Value: item.doc.Score},
				},
			},
		})
	}
	return out, nil
}

func (b Bridge[P, R, A, M, U]) mapBatch(ctx context.Context, batch Batch[M]) ([]mapped[M], error) {
	if batch.Omissions < 0 || len(batch.Documents) > b.MaxCandidates {
		return nil, memy.ErrBudget
	}
	items := make([]mapped[M], 0, len(batch.Documents))
	seen := make(map[string]bool)
	for i, doc := range batch.Documents {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := retrieval.ValidateDocument(doc); err != nil {
			return nil, memy.ErrInvalid
		}
		ref, err := b.Map(ctx, doc)
		if err != nil {
			return nil, err
		}
		if ref.Scope != b.Scope {
			return nil, memy.ErrScopeViolation
		}
		if err := (memy.RevisionRef{RecordID: ref.RecordID, Revision: ref.Revision}).Validate(); err != nil {
			return nil, err
		}
		if seen[ref.RecordID] {
			return nil, memy.ErrInvalid
		}
		seen[ref.RecordID] = true
		rank := doc.Rank
		if rank == 0 {
			rank = i + 1
		}
		items = append(items, mapped[M]{ref: ref, doc: doc, ordinal: rank})
	}
	slices.SortStableFunc(items, func(a, c mapped[M]) int {
		if order := cmp.Compare(a.ordinal, c.ordinal); order != 0 {
			return order
		}
		return cmp.Compare(a.ref.RecordID, c.ref.RecordID)
	})
	return items, ctx.Err()
}
