package retrievers

import (
	"context"
	"iter"

	"github.com/skosovsky/ragy"
)

// RouterFunc returns the name of the retriever target for the given query.
type RouterFunc func(ctx context.Context, query string) (string, error)

// RouterRetriever routes the request to one of the registered retrievers by name.
type RouterRetriever struct {
	Router  RouterFunc
	Targets map[string]ragy.Retriever
}

// NewRouterRetriever returns a RouterRetriever.
func NewRouterRetriever(router RouterFunc, targets map[string]ragy.Retriever) *RouterRetriever {
	return &RouterRetriever{Router: router, Targets: targets}
}

// Retrieve implements ragy.Retriever.
func (r *RouterRetriever) Retrieve(ctx context.Context, req ragy.SearchRequest) ([]ragy.Document, error) {
	targetName, err := r.Router(ctx, req.Query)
	if err != nil {
		return nil, err
	}
	retriever, ok := r.Targets[targetName]
	if !ok {
		return nil, ragy.ErrInvalidInput
	}
	return retriever.Retrieve(ctx, req)
}

// Stream implements ragy.Retriever.
func (r *RouterRetriever) Stream(ctx context.Context, req ragy.SearchRequest) iter.Seq2[ragy.Document, error] {
	targetName, err := r.Router(ctx, req.Query)
	if err != nil {
		return ragy.YieldDocuments(ctx, nil, err)
	}
	retriever, ok := r.Targets[targetName]
	if !ok {
		return ragy.YieldDocuments(ctx, nil, ragy.ErrInvalidInput)
	}
	return retriever.Stream(ctx, req)
}

var _ ragy.Retriever = (*RouterRetriever)(nil)
