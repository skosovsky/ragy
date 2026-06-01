package retrieval

import (
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
)

// GraphOptions carries graph traversal parameters for graph backends.
type GraphOptions struct {
	Seeds      []string
	Direction  graph.Direction
	Depth      int
	NodeFilter filter.Condition
	EdgeFilter filter.Condition
	Page       *ragy.Page
}

// Validate checks graph option invariants.
func (g *GraphOptions) Validate() error {
	if g == nil {
		return nil
	}
	if len(g.Seeds) == 0 {
		return fmt.Errorf("%w: graph seeds", ragy.ErrInvalidGraph)
	}
	if g.Depth <= 0 {
		return fmt.Errorf("%w: graph depth must be > 0", ragy.ErrInvalidGraph)
	}
	switch g.Direction {
	case graph.DirectionOutbound, graph.DirectionInbound, graph.DirectionUndirected:
	default:
		return fmt.Errorf("%w: graph direction %q", ragy.ErrInvalidGraph, g.Direction)
	}
	return g.Page.Validate()
}

// RetrieveOptions separates search tuning from domain filters.
type RetrieveOptions struct {
	TopK          int
	MinSimilarity float64
	HybridWeight  float64
	Filters       filter.Condition
	Vector        []float32
	Graph         *GraphOptions
}

// Validate checks option invariants.
func (o RetrieveOptions) Validate() error {
	if o.TopK < 0 {
		return fmt.Errorf("%w: top_k must be >= 0", ragy.ErrInvalidArgument)
	}
	if o.MinSimilarity < 0 || o.MinSimilarity > 1 {
		return fmt.Errorf("%w: min_similarity must be in [0,1]", ragy.ErrInvalidArgument)
	}
	if o.HybridWeight < 0 || o.HybridWeight > 1 {
		return fmt.Errorf("%w: hybrid_weight must be in [0,1]", ragy.ErrInvalidArgument)
	}
	if err := filter.ValidateCondition(o.Filters); err != nil {
		return err
	}
	return o.Graph.Validate()
}
