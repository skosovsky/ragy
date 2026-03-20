package retrievers

import (
	"context"
	"iter"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/internal/mathutil"
)

// MultiQueryRetriever expands the query via QueryTransformer, runs the base retriever for each,
// then deduplicates and merges results.
type MultiQueryRetriever struct {
	Transformer ragy.QueryTransformer
	Retriever   ragy.Retriever
}

// NewMultiQueryRetriever returns a new MultiQueryRetriever.
func NewMultiQueryRetriever(transformer ragy.QueryTransformer, retriever ragy.Retriever) *MultiQueryRetriever {
	return &MultiQueryRetriever{Transformer: transformer, Retriever: retriever}
}

// Retrieve implements ragy.Retriever.
func (r *MultiQueryRetriever) Retrieve(ctx context.Context, req ragy.SearchRequest) ([]ragy.Document, error) {
	queries, err := r.Transformer.Transform(ctx, req.Query)
	if err != nil {
		return nil, err
	}
	if len(queries) == 0 {
		queries = []string{req.Query}
	}
	results := make([][]ragy.Document, len(queries))
	err = parallel(ctx, len(queries), func(gctx context.Context, i int) error {
		req2 := req
		req2.Query = queries[i]
		res, err := r.Retriever.Retrieve(gctx, req2)
		if err != nil {
			return err
		}
		results[i] = res
		return nil
	})
	if err != nil {
		return nil, err
	}
	var all []ragy.Document
	for _, docs := range results {
		all = append(all, docs...)
	}
	merged := mathutil.DeduplicateDocuments(all)
	limit := req.Limit
	if limit <= 0 {
		limit = 10
	}
	if len(merged) > limit {
		merged = merged[:limit]
	}
	return merged, nil
}

// Stream implements ragy.Retriever.
func (r *MultiQueryRetriever) Stream(ctx context.Context, req ragy.SearchRequest) iter.Seq2[ragy.Document, error] {
	docs, err := r.Retrieve(ctx, req)
	return ragy.YieldDocuments(ctx, docs, err)
}

var _ ragy.Retriever = (*MultiQueryRetriever)(nil)
