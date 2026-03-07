package retrievers

import (
	"context"

	"github.com/skosovsky/ragy"
)

// HypothesisGenerator generates a hypothetical answer for the query (e.g. via LLM).
type HypothesisGenerator func(ctx context.Context, query string) (string, error)

// HyDERetriever retrieves by embedding a hypothetical answer instead of the query.
type HyDERetriever struct {
	Generate HypothesisGenerator
	Embedder ragy.DenseEmbedder
	Store    ragy.VectorStore
}

// NewHyDERetriever returns a new HyDERetriever.
func NewHyDERetriever(gen HypothesisGenerator, embedder ragy.DenseEmbedder, store ragy.VectorStore) *HyDERetriever {
	return &HyDERetriever{Generate: gen, Embedder: embedder, Store: store}
}

// Retrieve implements ragy.Retriever.
func (r *HyDERetriever) Retrieve(ctx context.Context, req ragy.SearchRequest) (ragy.RetrievalResult, error) {
	hypothesis, err := r.Generate(ctx, req.Query)
	if err != nil {
		return ragy.RetrievalResult{}, err
	}
	vecs, err := r.Embedder.Embed(ctx, []string{hypothesis})
	if err != nil {
		return ragy.RetrievalResult{}, err
	}
	if len(vecs) == 0 {
		return ragy.RetrievalResult{}, ragy.ErrEmbeddingFailed
	}
	req2 := req
	req2.DenseVector = vecs[0]
	docs, err := r.Store.Search(ctx, req2)
	if err != nil {
		return ragy.RetrievalResult{}, err
	}
	eval := map[string]any{"hypothesis": hypothesis}
	return ragy.RetrievalResult{Documents: docs, EvalData: eval}, nil
}

var _ ragy.Retriever = (*HyDERetriever)(nil)
