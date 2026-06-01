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

// Store provides document lookup and destructive operations.
type Store[TMeta any] interface {
	FindByIDs(ctx context.Context, ids []string) ([]retrieval.Document[TMeta], error)
	DeleteByIDs(ctx context.Context, ids []string) (DeleteResult, error)
	DeleteByFilter(ctx context.Context, cond filter.Condition) (DeleteResult, error)
	Schema() filter.Schema
}
