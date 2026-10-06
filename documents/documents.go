// Package documents provides canonical document-store contracts.
package documents

import (
	"context"

	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

// DeleteResult reports how many documents were deleted.
type DeleteResult struct {
	Deleted int
}

// RawStore provides explicit raw storage access. It has no scope/publication or
// retained-revision guarantee and must not be used as a scoped hydration path.
type RawStore[TMeta any] interface {
	FindByIDs(ctx context.Context, ids []string) ([]retrieval.Document[TMeta], error)
	DeleteByIDs(ctx context.Context, ids []string) (DeleteResult, error)
	DeleteByFilter(ctx context.Context, cond filter.Condition) (DeleteResult, error)
	Schema() filter.Schema
}
