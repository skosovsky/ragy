package retrievers

import (
	"context"
	"iter"

	"github.com/skosovsky/ragy"
)

// BaseVectorRetriever performs dense vector search: embeds the query and searches the store.
type BaseVectorRetriever struct {
	Embedder ragy.DenseEmbedder
	Store    ragy.VectorStore
}

// NewBaseVectorRetriever returns a new BaseVectorRetriever.
func NewBaseVectorRetriever(embedder ragy.DenseEmbedder, store ragy.VectorStore) *BaseVectorRetriever {
	return &BaseVectorRetriever{Embedder: embedder, Store: store}
}

func (r *BaseVectorRetriever) retrieveDocs(ctx context.Context, req ragy.SearchRequest) ([]ragy.Document, error) {
	if req.Query == "" {
		return nil, ragy.ErrEmptyQuery
	}
	vecs, err := r.Embedder.Embed(ctx, []string{req.Query})
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
func (r *BaseVectorRetriever) Retrieve(ctx context.Context, req ragy.SearchRequest) ([]ragy.Document, error) {
	return r.retrieveDocs(ctx, req)
}

// Stream implements ragy.Retriever.
func (r *BaseVectorRetriever) Stream(ctx context.Context, req ragy.SearchRequest) iter.Seq2[ragy.Document, error] {
	docs, err := r.retrieveDocs(ctx, req)
	return ragy.YieldDocuments(ctx, docs, err)
}

var _ ragy.Retriever = (*BaseVectorRetriever)(nil)
