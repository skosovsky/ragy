package retrieval

import (
	"context"

	"github.com/skosovsky/ragy/access"
)

// QueryReranker reranks documents using explicit query-aware scoring.
// Implementations honor read freshness and retain only their stated partial contract.
type QueryReranker[TMeta any] interface {
	Rerank(context.Context, access.Binding, string, ResultSet[TMeta]) (ResultSet[TMeta], error)
}
