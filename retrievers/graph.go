package retrievers

import (
	"context"
	"iter"

	"github.com/skosovsky/ragy"
)

// GraphRetriever retrieves by traversing the graph from entities (BFS) and returning linked documents.
// Use SearchRequest.GraphSeedEntityIDs to pass starting entity IDs (no LLM in ragy).
type GraphRetriever struct {
	Graph ragy.GraphStore
	Depth int
}

// GraphRetrieverOption configures GraphRetriever.
type GraphRetrieverOption func(*GraphRetriever)

// WithGraphDepth sets the BFS depth (default 2).
func WithGraphDepth(d int) GraphRetrieverOption {
	return func(g *GraphRetriever) {
		g.Depth = d
	}
}

// NewGraphRetriever returns a new GraphRetriever.
func NewGraphRetriever(graph ragy.GraphStore, opts ...GraphRetrieverOption) *GraphRetriever {
	gr := &GraphRetriever{Graph: graph, Depth: 2}
	for _, o := range opts {
		o(gr)
	}
	return gr
}

func (r *GraphRetriever) retrieveDocs(ctx context.Context, req ragy.SearchRequest) ([]ragy.Document, error) {
	entities := req.GraphSeedEntityIDs
	if len(entities) == 0 {
		return nil, ragy.ErrMissingGraphSeeds
	}
	nodes, _, err := r.Graph.SearchGraph(ctx, entities, r.Depth, req)
	if err != nil {
		return nil, err
	}
	docs := make([]ragy.Document, 0, len(nodes))
	for _, n := range nodes {
		content := n.ID
		if c, ok := n.Properties["content"].(string); ok {
			content = c
		}
		if c, ok := n.Properties["text"].(string); ok {
			content = c
		}
		docs = append(docs, ragy.Document{
			ID:       n.ID,
			Content:  content,
			Metadata: map[string]any{"label": n.Label, "properties": n.Properties},
		})
	}
	limit := req.Limit
	if limit <= 0 {
		limit = 10
	}
	if len(docs) > limit {
		docs = docs[:limit]
	}
	return docs, nil
}

// Retrieve implements ragy.Retriever.
func (r *GraphRetriever) Retrieve(ctx context.Context, req ragy.SearchRequest) ([]ragy.Document, error) {
	return r.retrieveDocs(ctx, req)
}

// Stream implements ragy.Retriever.
func (r *GraphRetriever) Stream(ctx context.Context, req ragy.SearchRequest) iter.Seq2[ragy.Document, error] {
	docs, err := r.retrieveDocs(ctx, req)
	return ragy.YieldDocuments(ctx, docs, err)
}

var _ ragy.Retriever = (*GraphRetriever)(nil)
