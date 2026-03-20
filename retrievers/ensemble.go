package retrievers

import (
	"context"
	"iter"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/rerankers"
)

// EnsembleRetriever runs multiple retrievers in parallel and merges results via a RankedListsMerger (e.g. RRF).
type EnsembleRetriever struct {
	Retrievers []ragy.Retriever
	Merger     rerankers.RankedListsMerger
}

// EnsembleOption configures EnsembleRetriever.
type EnsembleOption func(*EnsembleRetriever)

// WithEnsembleMerger sets the merger for combining ranked lists (default: RRFReranker).
func WithEnsembleMerger(m rerankers.RankedListsMerger) EnsembleOption {
	return func(e *EnsembleRetriever) {
		e.Merger = m
	}
}

// NewEnsembleRetriever returns a new EnsembleRetriever.
func NewEnsembleRetriever(retrievers []ragy.Retriever, opts ...EnsembleOption) *EnsembleRetriever {
	ens := &EnsembleRetriever{
		Retrievers: retrievers,
		Merger:     rerankers.NewRRFReranker(),
	}
	for _, o := range opts {
		o(ens)
	}
	return ens
}

// Retrieve implements ragy.Retriever.
func (e *EnsembleRetriever) Retrieve(ctx context.Context, req ragy.SearchRequest) ([]ragy.Document, error) {
	if len(e.Retrievers) == 0 {
		return nil, nil
	}
	lists := make([][]ragy.Document, len(e.Retrievers))
	err := parallel(ctx, len(e.Retrievers), func(gctx context.Context, i int) error {
		res, err := e.Retrievers[i].Retrieve(gctx, req)
		if err != nil {
			return err
		}
		lists[i] = res
		return nil
	})
	if err != nil {
		return nil, err
	}
	topK := req.Limit
	if topK <= 0 {
		topK = 10
	}
	merged := e.Merger.MergeRankedLists(ctx, lists, topK)
	return merged, nil
}

// Stream implements ragy.Retriever.
func (e *EnsembleRetriever) Stream(ctx context.Context, req ragy.SearchRequest) iter.Seq2[ragy.Document, error] {
	docs, err := e.Retrieve(ctx, req)
	return ragy.YieldDocuments(ctx, docs, err)
}

var _ ragy.Retriever = (*EnsembleRetriever)(nil)
