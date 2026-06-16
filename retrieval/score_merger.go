package retrieval

import "context"

// ScoreMerger merges result sets by max Score per MergeKey.
// Use only when all sources share a comparable score scale.
type ScoreMerger[TMeta any] struct {
	resolver IdentityResolver[TMeta]
}

// NewScoreMerger constructs a score-based merger for homogeneous ranked lists.
func NewScoreMerger[TMeta any](resolver IdentityResolver[TMeta]) *ScoreMerger[TMeta] {
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	return &ScoreMerger[TMeta]{resolver: resolver}
}

// Merge combines result sets keeping the highest score per MergeKey.
func (m *ScoreMerger[TMeta]) Merge(
	ctx context.Context,
	sets ...ResultSet[TMeta],
) (ResultSet[TMeta], error) {
	if err := ctx.Err(); err != nil {
		return NewResultSet[TMeta](nil, m.resolver), err
	}

	merged := NewResultSet[TMeta](nil, m.resolver)
	for _, set := range sets {
		if set == nil || set.IsEmpty() {
			continue
		}
		next, err := merged.Merge(set)
		if err != nil {
			return preserveResultOnError(merged, err, m.resolver)
		}
		merged = next
	}
	return merged, nil
}
