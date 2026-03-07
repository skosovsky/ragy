package retrievers

import (
	"context"

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
func (r *RouterRetriever) Retrieve(ctx context.Context, req ragy.SearchRequest) (ragy.RetrievalResult, error) {
	targetName, err := r.Router(ctx, req.Query)
	if err != nil {
		return ragy.RetrievalResult{}, err
	}
	retriever, ok := r.Targets[targetName]
	if !ok {
		return ragy.RetrievalResult{}, ragy.ErrInvalidInput
	}
	res, err := retriever.Retrieve(ctx, req)
	if err != nil {
		return res, err
	}
	if res.EvalData == nil {
		res.EvalData = make(map[string]any)
	}
	res.EvalData["routed_to"] = targetName
	return res, nil
}

var _ ragy.Retriever = (*RouterRetriever)(nil)
