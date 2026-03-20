// Package retrievers provides Retriever implementations (vector, ColBERT, graph, router, ensemble, multi-query, HyDE).
package retrievers

import (
	"context"
	"iter"

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

func (r *ColBERTRetriever) retrieveDocs(ctx context.Context, req ragy.SearchRequest) ([]ragy.Document, error) {
	if req.Query == "" {
		return nil, ragy.ErrEmptyQuery
	}
	tensors, err := r.Embedder.EmbedTensors(ctx, []string{req.Query})
	if err != nil {
		return nil, err
	}
	if len(tensors) == 0 || len(tensors[0]) == 0 {
		return nil, ragy.ErrEmbeddingFailed
	}
	req2 := req
	req2.TensorVector = tensors[0]
	return r.Store.Search(ctx, req2)
}

// Retrieve implements ragy.Retriever.
func (r *ColBERTRetriever) Retrieve(ctx context.Context, req ragy.SearchRequest) ([]ragy.Document, error) {
	return r.retrieveDocs(ctx, req)
}

// Stream implements ragy.Retriever.
func (r *ColBERTRetriever) Stream(ctx context.Context, req ragy.SearchRequest) iter.Seq2[ragy.Document, error] {
	docs, err := r.retrieveDocs(ctx, req)
	return ragy.YieldDocuments(ctx, docs, err)
}

var _ ragy.Retriever = (*ColBERTRetriever)(nil)
