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

// Config configures the store.
type Config[TMeta any] struct {
	Resolver retrieval.IdentityResolver[TMeta]
}

// Store is a Neo4j retrieval backend and graph store.
type Store[TMeta any] struct {
	runner   Runner[TMeta]
	schema   graph.Schema
	resolver retrieval.IdentityResolver[TMeta]
}

// New constructs a store.
func New[TMeta any](runner Runner[TMeta], schema graph.Schema, cfg Config[TMeta]) (*Store[TMeta], error) {
	if runner == nil {
		return nil, fmt.Errorf("%w: neo4j runner", ragy.ErrInvalidArgument)
	}
	if err := schema.Validate(); err != nil {
		return nil, err
	}

	return &Store[TMeta]{
		runner:   runner,
		schema:   schema,
		resolver: retrieval.DefaultResolver(cfg.Resolver),
	}, nil
}

// Retrieve implements retrieval.Backend by traversing the graph and projecting nodes to documents.
func (s *Store[TMeta]) Retrieve(
	ctx context.Context,
	req retrieval.Query[struct{}],
) (retrieval.ResultSet[TMeta], error) {
	opts := req.Options
	if err := opts.Validate(); err != nil {
		return retrieval.NewResultSet[TMeta](nil, s.resolver), err
	}
	if opts.Graph == nil {
		return retrieval.NewResultSet[TMeta](nil, s.resolver),
			fmt.Errorf("%w: neo4j retrieve requires graph options", ragy.ErrInvalidArgument)
	}

	if err := s.schema.ValidateTraversal(graph.TraversalRequest{
		Seeds:      opts.Graph.Seeds,
		Direction:  opts.Graph.Direction,
		Depth:      opts.Graph.Depth,
		NodeFilter: opts.Graph.NodeFilter,
		EdgeFilter: opts.Graph.EdgeFilter,
		Page:       opts.Graph.Page,
	}); err != nil {
		return retrieval.NewResultSet[TMeta](nil, s.resolver), err
	}

	snapshot, err := s.runner.Traverse(ctx, Query{
		Seeds:      append([]string(nil), opts.Graph.Seeds...),
		Direction:  opts.Graph.Direction,
		Depth:      opts.Graph.Depth,
		NodeFilter: opts.Graph.NodeFilter,
		EdgeFilter: opts.Graph.EdgeFilter,
		Page:       opts.Graph.Page,
	})
	if err != nil {
		return retrieval.NewResultSet[TMeta](nil, s.resolver),
			ragy.WrapBackendError(err, "neo4j traverse")
	}

	if len(snapshot.Nodes) == 0 {
		return retrieval.NewResultSet[TMeta](nil, s.resolver), nil
	}

	docs := make([]retrieval.Document[TMeta], 0, len(snapshot.Nodes))
	for i, node := range snapshot.Nodes {
		doc := retrieval.Document[TMeta]{
			ID:         node.ID,
			Content:    node.Content,
			ScoreState: retrieval.ScoreAbsent,
			Rank:       i + 1,
			Meta:       node.Meta,
		}
		if err := retrieval.ValidateDocument(doc); err != nil {
			rs := retrieval.NewResultSet(docs, s.resolver)
			return retrieval.PreserveResultOnError(
				rs,
				ragy.WrapProjectionError(err, "neo4j validate"),
				s.resolver,
			)
		}
		docs = append(docs, doc)
	}

	limit := opts.BackendFetchLimit()
	// BackendFetchLimit truncates by traversal order; scores are not ranked for graph nodes.
	if limit > 0 && len(docs) > limit {
		docs = docs[:limit]
	}

	return retrieval.NewResultSet(docs, s.resolver), nil
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
		return graph.Snapshot[TMeta]{}, ragy.WrapBackendError(err, "neo4j traverse")
	}
	return graph.NormalizeSnapshot(s.schema, snapshot)
}

// Upsert implements graph.Store.
func (s *Store[TMeta]) Upsert(ctx context.Context, snapshot graph.Snapshot[TMeta]) error {
	normalized, err := graph.NormalizeSnapshot(s.schema, snapshot)
	if err != nil {
		return err
	}
	if err := s.runner.Upsert(ctx, normalized); err != nil {
		return ragy.WrapBackendError(err, "neo4j upsert")
	}
	return nil
}

// Schema returns the finalized graph schema used by the store.
func (s *Store[TMeta]) Schema() graph.Schema {
	return s.schema
}

var (
	_ retrieval.Backend[struct{}, any] = (*Store[any])(nil)
	_ graph.Store[any]                 = (*Store[any])(nil)
)
