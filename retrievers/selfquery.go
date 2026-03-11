package retrievers

import (
	"context"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

// SelfQueryRetriever decorates an inner Retriever by parsing the natural-language query
// via QueryParser into a semantic query and optional filter, then merging with request filter (RBAC-safe AND).
type SelfQueryRetriever struct {
	inner  ragy.Retriever
	parser ragy.QueryParser
}

// NewSelfQueryRetriever returns a SelfQueryRetriever that wraps inner and uses parser to parse queries.
func NewSelfQueryRetriever(inner ragy.Retriever, parser ragy.QueryParser) *SelfQueryRetriever {
	return &SelfQueryRetriever{inner: inner, parser: parser}
}

// Retrieve implements ragy.Retriever.
func (r *SelfQueryRetriever) Retrieve(ctx context.Context, req ragy.SearchRequest) (ragy.RetrievalResult, error) {
	parsed, err := r.parser.Parse(ctx, req.Query)
	if err != nil {
		return ragy.RetrievalResult{}, err
	}

	finalReq := req
	finalReq.Query = parsed.SemanticQuery
	if parsed.Limit > 0 {
		finalReq.Limit = parsed.Limit
	}
	if parsed.Filter != nil {
		if req.Filter != nil {
			finalReq.Filter = filter.All(req.Filter, parsed.Filter)
		} else {
			finalReq.Filter = parsed.Filter
		}
	}

	res, err := r.inner.Retrieve(ctx, finalReq)
	if err != nil {
		return res, err
	}
	if res.EvalData == nil {
		res.EvalData = make(map[string]any)
	}
	res.EvalData["parsed_semantic_query"] = parsed.SemanticQuery
	res.EvalData["parsed_has_filter"] = parsed.Filter != nil
	return res, nil
}

var _ ragy.Retriever = (*SelfQueryRetriever)(nil)
