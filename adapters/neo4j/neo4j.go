package neo4j

import (
	"context"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/retrieval"
)

// Query is the explicit traversal query handed to the runner.
type Query struct {
	Seeds      []string
	Direction  graph.Direction
	Depth      int
	NodeFilter filter.Condition
	EdgeFilter filter.Condition
	Page       *ragy.Page
}

// Runner executes graph operations.
type Runner[TMeta any] interface {
	Traverse(ctx context.Context, query Query) (graph.Snapshot[TMeta], error)
	Upsert(ctx context.Context, snapshot graph.Snapshot[TMeta]) error
}

// Store is a Neo4j retrieval backend and graph store.
type Store[TMeta any] struct {
	runner Runner[TMeta]
	schema graph.Schema
}

// New constructs a store.
func New[TMeta any](runner Runner[TMeta], schema graph.Schema) (*Store[TMeta], error) {
	if runner == nil {
		return nil, fmt.Errorf("%w: neo4j runner", ragy.ErrInvalidArgument)
	}
	if err := schema.Validate(); err != nil {
		return nil, err
	}

	return &Store[TMeta]{runner: runner, schema: schema}, nil
}

// Retrieve implements retrieval.Backend by traversing the graph and projecting nodes to documents.
func (s *Store[TMeta]) Retrieve(
	ctx context.Context,
	_ string,
	opts retrieval.RetrieveOptions,
) ([]retrieval.Document[TMeta], error) {
	if opts.Graph == nil {
		return nil, fmt.Errorf("%w: neo4j retrieve requires graph options", ragy.ErrInvalidArgument)
	}

	snapshot, err := s.Traverse(ctx, graph.TraversalRequest{
		Seeds:      append([]string(nil), opts.Graph.Seeds...),
		Direction:  opts.Graph.Direction,
		Depth:      opts.Graph.Depth,
		NodeFilter: opts.Graph.NodeFilter,
		EdgeFilter: opts.Graph.EdgeFilter,
		Page:       opts.Graph.Page,
	})
	if err != nil {
		return nil, err
	}

	if len(snapshot.Nodes) == 0 {
		return nil, nil
	}

	docs := make([]retrieval.Document[TMeta], 0, len(snapshot.Nodes))
	for _, node := range snapshot.Nodes {
		doc := retrieval.Document[TMeta]{
			ID:      node.ID,
			Content: node.Content,
			Score:   0,
			Meta:    node.Meta,
		}
		if err := retrieval.ValidateDocument(doc); err != nil {
			return nil, err
		}
		docs = append(docs, doc)
	}

	if opts.TopK > 0 && len(docs) > opts.TopK {
		docs = docs[:opts.TopK]
	}

	return docs, nil
}

// Traverse implements graph.Store.
func (s *Store[TMeta]) Traverse(ctx context.Context, req graph.TraversalRequest) (graph.Snapshot[TMeta], error) {
	if err := s.schema.ValidateTraversal(req); err != nil {
		return graph.Snapshot[TMeta]{}, err
	}

	snapshot, err := s.runner.Traverse(ctx, Query{
		Seeds:      append([]string(nil), req.Seeds...),
		Direction:  req.Direction,
		Depth:      req.Depth,
		NodeFilter: req.NodeFilter,
		EdgeFilter: req.EdgeFilter,
		Page:       req.Page,
	})
	if err != nil {
		return graph.Snapshot[TMeta]{}, err
	}
	return graph.NormalizeSnapshot(s.schema, snapshot)
}

// Upsert implements graph.Store.
func (s *Store[TMeta]) Upsert(ctx context.Context, snapshot graph.Snapshot[TMeta]) error {
	normalized, err := graph.NormalizeSnapshot(s.schema, snapshot)
	if err != nil {
		return err
	}
	return s.runner.Upsert(ctx, normalized)
}

// Schema returns the finalized graph schema used by the store.
func (s *Store[TMeta]) Schema() graph.Schema {
	return s.schema
}

var (
	_ retrieval.Backend[any] = (*Store[any])(nil)
	_ graph.Store[any]       = (*Store[any])(nil)
)
