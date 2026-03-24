package neo4j

import (
	"context"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
)

// Query is the explicit traversal query handed to the runner.
type Query struct {
	Seeds      []string
	Direction  graph.Direction
	Depth      int
	NodeFilter filter.IR
	EdgeFilter filter.IR
	Page       *ragy.Page
}

// Runner executes graph operations.
type Runner interface {
	Traverse(ctx context.Context, query Query) (graph.Snapshot, error)
	Upsert(ctx context.Context, snapshot graph.Snapshot) error
}

// Store is a graph-only Neo4j adapter.
type Store struct {
	runner Runner
	schema graph.Schema
}

// New constructs a store.
func New(runner Runner, schema graph.Schema) (*Store, error) {
	if runner == nil {
		return nil, fmt.Errorf("%w: neo4j runner", ragy.ErrInvalidArgument)
	}
	if err := schema.Validate(); err != nil {
		return nil, err
	}

	return &Store{runner: runner, schema: schema}, nil
}

// Traverse implements graph.Store.
func (s *Store) Traverse(ctx context.Context, req graph.TraversalRequest) (graph.Snapshot, error) {
	if err := s.schema.ValidateTraversal(req); err != nil {
		return graph.Snapshot{}, err
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
		return graph.Snapshot{}, err
	}
	return s.schema.NormalizeSnapshot(snapshot)
}

// Upsert implements graph.Store.
func (s *Store) Upsert(ctx context.Context, snapshot graph.Snapshot) error {
	normalized, err := s.schema.NormalizeSnapshot(snapshot)
	if err != nil {
		return err
	}
	return s.runner.Upsert(ctx, normalized)
}

// Schema returns the finalized graph schema used by the store.
func (s *Store) Schema() graph.Schema {
	return s.schema
}

var _ graph.Store = (*Store)(nil)
