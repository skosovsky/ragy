package retrievers

import (
	"context"
	"iter"

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

func (r *HyDERetriever) retrieveDocs(ctx context.Context, req ragy.SearchRequest) ([]ragy.Document, error) {
	hypothesis, err := r.Generate(ctx, req.Query)
	if err != nil {
		return nil, err
	}
	vecs, err := r.Embedder.Embed(ctx, []string{hypothesis})
	if err != nil {
		return nil, err
	}
	if len(vecs) == 0 {
		return nil, ragy.ErrEmbeddingFailed
	}
	req2 := req
	req2.DenseVector = vecs[0]
	return r.Store.Search(ctx, req2)
}

// Retrieve implements ragy.Retriever.
func (r *HyDERetriever) Retrieve(ctx context.Context, req ragy.SearchRequest) ([]ragy.Document, error) {
	return r.retrieveDocs(ctx, req)
}

// Stream implements ragy.Retriever.
func (r *HyDERetriever) Stream(ctx context.Context, req ragy.SearchRequest) iter.Seq2[ragy.Document, error] {
	docs, err := r.retrieveDocs(ctx, req)
	return ragy.YieldDocuments(ctx, docs, err)
}

var _ ragy.Retriever = (*HyDERetriever)(nil)
