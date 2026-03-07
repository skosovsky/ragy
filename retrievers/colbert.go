// Package retrievers provides Retriever implementations (vector, ColBERT, graph, router, ensemble, multi-query, HyDE).
package retrievers

import (
	"context"

	"github.com/skosovsky/ragy"
)

// ColBERTRetriever uses late interaction: embeds the query as a token tensor and searches with TensorVector.
type ColBERTRetriever struct {
	Embedder ragy.TensorEmbedder
	Store    ragy.VectorStore
}

// NewColBERTRetriever returns a new ColBERTRetriever.
func NewColBERTRetriever(embedder ragy.TensorEmbedder, store ragy.VectorStore) *ColBERTRetriever {
	return &ColBERTRetriever{Embedder: embedder, Store: store}
}

// Retrieve implements ragy.Retriever.
func (r *ColBERTRetriever) Retrieve(ctx context.Context, req ragy.SearchRequest) (ragy.RetrievalResult, error) {
	if req.Query == "" {
		return ragy.RetrievalResult{}, ragy.ErrEmptyQuery
	}
	tensors, err := r.Embedder.EmbedTensors(ctx, []string{req.Query})
	if err != nil {
		return ragy.RetrievalResult{}, err
	}
	if len(tensors) == 0 || len(tensors[0]) == 0 {
		return ragy.RetrievalResult{}, ragy.ErrEmbeddingFailed
	}
	// Per-token vectors for the single query: [][]float32.
	req2 := req
	req2.TensorVector = tensors[0]
	docs, err := r.Store.Search(ctx, req2)
	if err != nil {
		return ragy.RetrievalResult{}, err
	}
	eval := make(map[string]any)
	if len(docs) > 0 {
		scores := make([]float32, len(docs))
		for i := range docs {
			scores[i] = docs[i].Score
		}
		eval["interaction_scores"] = scores
	}
	return ragy.RetrievalResult{Documents: docs, EvalData: eval}, nil
}

var _ ragy.Retriever = (*ColBERTRetriever)(nil)
