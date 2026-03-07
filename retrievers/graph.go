package retrievers

import (
	"context"

	"github.com/skosovsky/ragy"
)

// GraphRetriever retrieves by traversing the graph from entities (BFS) and returning linked documents.
type GraphRetriever struct {
	Graph   ragy.GraphStore
	Extract ragy.EntityExtractor
	Depth   int
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

// NewGraphRetrieverWithExtractor returns a GraphRetriever that uses extractor to get entities from the query.
func NewGraphRetrieverWithExtractor(graph ragy.GraphStore, extract ragy.EntityExtractor, opts ...GraphRetrieverOption) *GraphRetriever {
	gr := NewGraphRetriever(graph, opts...)
	gr.Extract = extract
	return gr
}

// Retrieve implements ragy.Retriever.
func (r *GraphRetriever) Retrieve(ctx context.Context, req ragy.SearchRequest) (ragy.RetrievalResult, error) {
	var entities []string
	if r.Extract != nil && req.Query != "" {
		nodes, _, err := r.Extract(ctx, req.Query)
		if err != nil {
			return ragy.RetrievalResult{}, err
		}
		for _, n := range nodes {
			entities = append(entities, n.ID)
		}
	}
	if len(entities) == 0 {
		return ragy.RetrievalResult{Documents: nil, EvalData: map[string]any{}}, nil
	}
	nodes, edges, err := r.Graph.SearchGraph(ctx, entities, r.Depth, req)
	if err != nil {
		return ragy.RetrievalResult{}, err
	}
	// Convert nodes to documents (content from node properties or ID).
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
	return ragy.RetrievalResult{
		Documents: docs,
		EvalData:  map[string]any{"nodes": len(nodes), "edges": len(edges)},
	}, nil
}

var _ ragy.Retriever = (*GraphRetriever)(nil)
