package retrievers

import (
	"context"

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

// Retrieve implements ragy.Retriever.
func (r *BaseVectorRetriever) Retrieve(ctx context.Context, req ragy.SearchRequest) (ragy.RetrievalResult, error) {
	if req.Query == "" {
		return ragy.RetrievalResult{}, ragy.ErrEmptyQuery
	}
	vecs, err := r.Embedder.Embed(ctx, []string{req.Query})
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
	eval := make(map[string]any)
	if len(docs) > 0 {
		scores := make([]float32, len(docs))
		for i := range docs {
			scores[i] = docs[i].Score
		}
		eval["raw_scores"] = scores
	}
	return ragy.RetrievalResult{Documents: docs, EvalData: eval}, nil
}

var _ ragy.Retriever = (*BaseVectorRetriever)(nil)
