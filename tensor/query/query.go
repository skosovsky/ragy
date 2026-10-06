// Package query composes admitted candidate retrieval with bounded tensor scoring.
package query

import (
	"context"
	"slices"

	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
	"github.com/skosovsky/ragy/tensor"
)

// Intent keeps a token matrix separate from the dense Vector tuning option.
// Candidates are an explicit universe, never a request for exhaustive search.
type Intent struct {
	Embedding       tensor.Embedding
	Candidates      []source.Reference
	CandidateBudget int
}

type Result[TMeta any] struct {
	Documents retrieval.ResultSet[TMeta]
	Evidence  tensor.RerankResult
}

// Target declares actual admission and bounded native scoring capabilities.
type Target[TMeta any] interface {
	retrieval.ReadCapabilityProvider
	QueryCapabilities() tensor.QueryCapabilities
	Query(context.Context, retrieval.Query[Intent]) (Result[TMeta], error)
}

// CopyRequest owns all core mutable query options, token matrices and candidate references.
func CopyRequest(request retrieval.Query[Intent]) retrieval.Query[Intent] {
	request = retrieval.CopyRequestOptions(request)
	request.Intent.Candidates = slices.Clone(request.Intent.Candidates)
	request.Intent.Embedding.Tokens = slices.Clone(request.Intent.Embedding.Tokens)
	for i := range request.Intent.Embedding.Tokens {
		request.Intent.Embedding.Tokens[i] = slices.Clone(request.Intent.Embedding.Tokens[i])
	}
	return request
}
