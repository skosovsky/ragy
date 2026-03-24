// Package documents provides canonical document-store contracts.
package documents

import (
	"context"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

// DeleteResult reports how many documents were deleted.
type DeleteResult struct {
	Deleted int
}

// Store provides document lookup and destructive operations.
type Store interface {
	FindByIDs(ctx context.Context, ids []string) ([]ragy.Document, error)
	DeleteByIDs(ctx context.Context, ids []string) (DeleteResult, error)
	DeleteByFilter(ctx context.Context, expr filter.IR) (DeleteResult, error)
	Schema() filter.Schema
}
