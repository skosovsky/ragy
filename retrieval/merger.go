package retrieval

import "context"

// ResultMerger merges already-ranked result sets.
type ResultMerger[TMeta any] interface {
	Merge(ctx context.Context, sets ...ResultSet[TMeta]) (ResultSet[TMeta], error)
}
