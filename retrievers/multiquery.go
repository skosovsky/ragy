package retrievers

import (
	"context"

	"golang.org/x/sync/errgroup"

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
func (r *MultiQueryRetriever) Retrieve(ctx context.Context, req ragy.SearchRequest) (ragy.RetrievalResult, error) {
	queries, err := r.Transformer.Transform(ctx, req.Query)
	if err != nil {
		return ragy.RetrievalResult{}, err
	}
	if len(queries) == 0 {
		queries = []string{req.Query}
	}
	g, gctx := errgroup.WithContext(ctx)
	results := make([][]ragy.Document, len(queries))
	for i, q := range queries {
		g.Go(func() error {
			req2 := req
			req2.Query = q
			res, err := r.Retriever.Retrieve(gctx, req2)
			if err != nil {
				return err
			}
			results[i] = res.Documents
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return ragy.RetrievalResult{}, err
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
	eval := map[string]any{
		"sub_queries":       queries,
		"per_query_results": results,
	}
	return ragy.RetrievalResult{Documents: merged, EvalData: eval}, nil
}

var _ ragy.Retriever = (*MultiQueryRetriever)(nil)
