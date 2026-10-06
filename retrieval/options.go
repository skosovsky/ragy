package retrieval

import (
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/embedding"
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
	FetchLimit int
	TopK       int
	Threshold  *ScoreThreshold
	Filters    filter.Condition
	Vector     []float32
	Space      embedding.Space
	Graph      *GraphOptions
}

// Validate checks option invariants.
func (o RetrieveOptions) Validate() error {
	if len(o.Vector) > 0 {
		if err := o.Space.ValidateVector(o.Vector); err != nil {
			return err
		}
	}
	if o.FetchLimit < 0 {
		return fmt.Errorf("%w: fetch_limit must be >= 0", ragy.ErrInvalidArgument)
	}
	if o.TopK < 0 {
		return fmt.Errorf("%w: top_k must be >= 0", ragy.ErrInvalidArgument)
	}
	if o.TopK == 0 && o.FetchLimit == 0 {
		return fmt.Errorf("%w: top_k and fetch_limit cannot both be zero", ragy.ErrInvalidArgument)
	}
	if o.FetchLimit > 0 && o.TopK > 0 && o.FetchLimit < o.TopK {
		return fmt.Errorf(
			"%w: fetch_limit (%d) cannot be less than top_k (%d)",
			ragy.ErrInvalidArgument,
			o.FetchLimit,
			o.TopK,
		)
	}
	if o.Threshold != nil {
		if err := o.Threshold.Validate(); err != nil {
			return err
		}
	}
	if err := filter.ValidateCondition(o.Filters); err != nil {
		return err
	}
	return o.Graph.Validate()
}

// BackendFetchLimit returns the effective row limit for backend queries.
// When FetchLimit is unset (0), TopK is used as the fetch cap.
func (o RetrieveOptions) BackendFetchLimit() int {
	limit := o.FetchLimit
	if limit <= 0 {
		return o.TopK
	}
	return limit
}

// ScoreThreshold is an inclusive minimum in one explicitly declared scale.
// Nil means no threshold; negative and zero native thresholds are valid.
type ScoreThreshold struct {
	Value     float64        `json:"value"`
	State     ScoreState     `json:"state"`
	Semantics ScoreSemantics `json:"semantics"`
}

// Validate checks that a threshold has a finite value and a scored scale.
func (t ScoreThreshold) Validate() error {
	if !t.State.IsScored() {
		return fmt.Errorf("%w: threshold requires scored state", ragy.ErrInvalidArgument)
	}
	return ValidateDocument(
		Document[struct{}]{
			ID:             "threshold",
			Content:        "",
			Score:          t.Value,
			ScoreState:     t.State,
			ScoreSemantics: t.Semantics,
			Rank:           0,
			Meta:           struct{}{},
		},
	)
}
