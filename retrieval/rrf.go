package retrieval

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sort"

	ragy "github.com/skosovsky/ragy"
)

// ReciprocalRankFusion merges ranked lists with RRF.
type ReciprocalRankFusion[TMeta any] struct {
	k        int
	resolver IdentityResolver[TMeta]
}

type fusedState[TMeta any] struct {
	doc   Document[TMeta]
	score float64
}

// NewReciprocalRankFusion constructs an RRF merger.
func NewReciprocalRankFusion[TMeta any](
	k int,
	resolver IdentityResolver[TMeta],
) (*ReciprocalRankFusion[TMeta], error) {
	if k <= 0 {
		return nil, fmt.Errorf("%w: RRF k must be > 0", ragy.ErrInvalidArgument)
	}
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}

	return &ReciprocalRankFusion[TMeta]{
		k:        k,
		resolver: resolver,
	}, nil
}

// Merge merges ranked result sets by stable MergeKey.
func (r *ReciprocalRankFusion[TMeta]) Merge(
	ctx context.Context,
	sets ...ResultSet[TMeta],
) (ResultSet[TMeta], error) {
	if ctx == nil {
		ctx = context.Background()
	}

	if len(sets) == 0 {
		return NewResultSet[TMeta](nil, r.resolver), nil
	}

	lists := make([][]Document[TMeta], 0, len(sets))
	for _, set := range sets {
		if set == nil || set.IsEmpty() {
			continue
		}
		lists = append(lists, set.Documents())
	}
	if len(lists) == 0 {
		return NewResultSet[TMeta](nil, r.resolver), nil
	}

	seen, mergeErr := r.mergeLists(ctx, lists...)
	if mergeErr != nil {
		return r.partialResultFromSeen(ctx, seen, mergeErr)
	}
	return r.materializeRRFFromSeen(ctx, seen)
}

func (r *ReciprocalRankFusion[TMeta]) partialResultFromSeen(
	_ context.Context,
	seen map[string]fusedState[TMeta],
	err error,
) (ResultSet[TMeta], error) {
	rs, matErr := r.materializeRRFFromSeen(context.Background(), seen)
	if matErr != nil {
		return preserveResultOnError(rs, errors.Join(err, matErr), r.resolver)
	}
	return preserveResultOnError(rs, err, r.resolver)
}

func (r *ReciprocalRankFusion[TMeta]) materializeRRFFromSeen(
	ctx context.Context,
	seen map[string]fusedState[TMeta],
) (ResultSet[TMeta], error) {
	if len(seen) == 0 {
		if err := ctx.Err(); err != nil {
			return NewResultSet[TMeta](nil, r.resolver), err
		}
		return NewResultSet[TMeta](nil, r.resolver), nil
	}

	maxScore, scoreErr := maxMergedScore(ctx, seen)
	out, buildErr := buildMergedDocuments(ctx, seen, maxScore)
	if buildErr != nil {
		return preserveResultOnError(NewResultSet(r.sortMergedDocuments(out), r.resolver), buildErr, r.resolver)
	}
	if scoreErr != nil {
		return preserveResultOnError(NewResultSet(r.sortMergedDocuments(out), r.resolver), scoreErr, r.resolver)
	}
	if err := ctx.Err(); err != nil {
		return preserveResultOnError(NewResultSet(r.sortMergedDocuments(out), r.resolver), err, r.resolver)
	}
	return NewResultSet(r.sortMergedDocuments(out), r.resolver), nil
}

func (r *ReciprocalRankFusion[TMeta]) sortMergedDocuments(docs []Document[TMeta]) []Document[TMeta] {
	if len(docs) == 0 {
		return docs
	}
	sort.SliceStable(docs, func(i, j int) bool {
		return docs[i].Score > docs[j].Score
	})
	return docs
}

func (r *ReciprocalRankFusion[TMeta]) mergeLists(
	ctx context.Context,
	lists ...[]Document[TMeta],
) (map[string]fusedState[TMeta], error) {
	seen := make(map[string]fusedState[TMeta])
	for i, list := range lists {
		if i > 0 {
			if err := ctx.Err(); err != nil {
				return seen, err
			}
		}
		if err := r.mergeList(ctx, seen, list); err != nil {
			return seen, err
		}
	}

	return seen, nil
}

func (r *ReciprocalRankFusion[TMeta]) mergeList(
	ctx context.Context,
	seen map[string]fusedState[TMeta],
	list []Document[TMeta],
) error {
	for rank, doc := range list {
		if err := ValidateDocument(doc); err != nil {
			return ragy.WrapProjectionError(err, "rrf validate")
		}

		mergeKey := r.resolver.Resolve(doc).MergeKey
		if mergeKey == "" {
			return fmt.Errorf("%w: empty merge key for document %q", ragy.ErrInvalidArgument, doc.ID)
		}
		current := seen[mergeKey]
		if current.doc.ID == "" {
			current.doc = doc
		} else if !samePayload(current.doc, doc) {
			return fmt.Errorf("%w: conflicting payload for merge key %q", ragy.ErrInvalidArgument, mergeKey)
		}

		current.score += 1.0 / float64(r.k+rank+1)
		seen[mergeKey] = current
		if err := ctx.Err(); err != nil {
			return err
		}
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
) ([]Document[TMeta], error) {
	keys := make([]string, 0, len(seen))
	for key := range seen {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	out := make([]Document[TMeta], 0, len(keys))
	for _, key := range keys {
		item := seen[key]
		doc := item.doc
		if maxScore > 0 {
			doc.Score = ragy.ClampScore(item.score / maxScore)
		}
		out = append(out, doc)
		if err := ctx.Err(); err != nil {
			return out, err
		}
	}

	return out, nil
}

func samePayload[TMeta any](left, right Document[TMeta]) bool {
	return left.Content == right.Content && reflect.DeepEqual(left.Meta, right.Meta)
}
