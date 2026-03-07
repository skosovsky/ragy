package retrievers

import (
	"context"

	"golang.org/x/sync/errgroup"

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
func (e *EnsembleRetriever) Retrieve(ctx context.Context, req ragy.SearchRequest) (ragy.RetrievalResult, error) {
	if len(e.Retrievers) == 0 {
		return ragy.RetrievalResult{}, nil
	}
	g, gctx := errgroup.WithContext(ctx)
	lists := make([][]ragy.Document, len(e.Retrievers))
	for i := range e.Retrievers {
		g.Go(func() error {
			res, err := e.Retrievers[i].Retrieve(gctx, req)
			if err != nil {
				return err
			}
			lists[i] = res.Documents
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return ragy.RetrievalResult{}, err
	}
	topK := req.Limit
	if topK <= 0 {
		topK = 10
	}
	merged := e.Merger.MergeRankedLists(ctx, lists, topK)
	eval := map[string]any{"per_retriever_results": lists}
	return ragy.RetrievalResult{Documents: merged, EvalData: eval}, nil
}

var _ ragy.Retriever = (*EnsembleRetriever)(nil)
