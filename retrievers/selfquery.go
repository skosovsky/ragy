package retrievers

import (
	"context"
	"iter"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

// SelfQueryRetriever merges SearchRequest.Filter with ParsedQuery from SearchRequest (set by the application).
// It does not call LLM; the app must parse the query and set req.ParsedQuery before calling Retrieve/Stream.
type SelfQueryRetriever struct {
	inner ragy.Retriever
}

// NewSelfQueryRetriever returns a SelfQueryRetriever that wraps inner.
func NewSelfQueryRetriever(inner ragy.Retriever) *SelfQueryRetriever {
	return &SelfQueryRetriever{inner: inner}
}

func (r *SelfQueryRetriever) buildRequest(req ragy.SearchRequest) (ragy.SearchRequest, error) {
	if req.ParsedQuery == nil {
		return req, ragy.ErrMissingParsedQuery
	}
	parsed := req.ParsedQuery
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
	return finalReq, nil
}

// Retrieve implements ragy.Retriever.
func (r *SelfQueryRetriever) Retrieve(ctx context.Context, req ragy.SearchRequest) ([]ragy.Document, error) {
	finalReq, err := r.buildRequest(req)
	if err != nil {
		return nil, err
	}
	return r.inner.Retrieve(ctx, finalReq)
}

// Stream implements ragy.Retriever.
func (r *SelfQueryRetriever) Stream(ctx context.Context, req ragy.SearchRequest) iter.Seq2[ragy.Document, error] {
	finalReq, err := r.buildRequest(req)
	if err != nil {
		return ragy.YieldDocuments(ctx, nil, err)
	}
	return r.inner.Stream(ctx, finalReq)
}

var _ ragy.Retriever = (*SelfQueryRetriever)(nil)
