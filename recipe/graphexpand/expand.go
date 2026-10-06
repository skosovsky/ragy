// Package graphexpand provides optional model-free bounded local graph expansion.
// Hosts supply seeds, traversal filters, authorization and per-attempt prices.
package graphexpand

import (
	"context"
	"errors"
	"math"
	"slices"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/graph/managed"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/source"
)

// Config uses the scoped, pinned, model-free managed traversal profile. The host
// quote is a fixed cost per dispatched graph attempt, including failed attempts.
// Quotes/cloners must be bounded, pure and safe for concurrent use.
type Config[TMeta any] struct {
	Adapter   *managed.Adapter[TMeta]
	MaxDepth  int
	MaxNodes  int
	MaxEdges  int
	Duration  time.Duration
	Now       func() time.Time
	Quote     func(context.Context) (budget.Reservation, error)
	CloneMeta func(TMeta) (TMeta, error)
}
type Request struct {
	Read      access.Binding
	HostBasis string
	Traversal graph.TraversalRequest
}
type Result[TMeta any] struct {
	Evidence   managed.Result[TMeta]
	Outcome    recipe.Outcome
	Stop       StopReason
	GraphCalls uint64
}

type StopReason string

const (
	Expanded         StopReason = "expanded"
	BudgetExhausted  StopReason = "budget-exhausted"
	PriceUnavailable StopReason = "price-unavailable"
)

type Recipe[TMeta any] struct{ config Config[TMeta] }

func New[TMeta any](cfg Config[TMeta]) (*Recipe[TMeta], error) {
	if cfg.Adapter == nil || cfg.MaxDepth <= 0 || cfg.MaxNodes <= 0 || cfg.MaxEdges <= 0 ||
		cfg.MaxNodes > math.MaxInt-cfg.MaxEdges || cfg.Duration <= 0 || cfg.Now == nil || cfg.Quote == nil || cfg.CloneMeta == nil {
		return nil, ragy.ErrInvalidArgument
	}
	return &Recipe[TMeta]{config: cfg}, nil
}

// Run executes one bounded breadth-first expansion, with the target's visited set
// cutting cycles. It reserves the shared ledger before dispatch and has no retry,
// planner, model call, answer generation or implicit source/identity decisions.
func (r *Recipe[TMeta]) Run(ctx context.Context, request Request, ledger *budget.Ledger) (Result[TMeta], error) {
	var empty Result[TMeta]
	if r == nil || ledger == nil || request.Traversal.Depth > r.config.MaxDepth ||
		len(request.Traversal.Seeds) > r.config.MaxNodes {
		return empty, ragy.ErrInvalidArgument
	}
	child, cancel := context.WithTimeout(ctx, r.config.Duration)
	defer cancel()
	deadline := r.config.Now().Add(r.config.Duration)
	gate := func() error {
		if err := request.Read.Check(child); err != nil {
			return err
		}
		if !r.config.Now().Before(deadline) {
			return context.DeadlineExceeded
		}
		return nil
	}
	if err := gate(); err != nil {
		return empty, err
	}
	request.Traversal.Seeds = slices.Clone(request.Traversal.Seeds)
	call := managed.Request{
		Read:      request.Read,
		HostBasis: request.HostBasis,
		Traversal: request.Traversal,
		MaxNodes:  r.config.MaxNodes,
		MaxEdges:  r.config.MaxEdges,
	}
	if err := r.config.Adapter.AdmitTraversal(child, call); err != nil {
		return empty, err
	}
	quote, err := r.config.Quote(child)
	if err != nil {
		return empty, err
	}
	if err = gate(); err != nil {
		return empty, err
	}
	if quote.Kind != budget.Retrieval || quote.Usage.InputTokens != 0 || quote.Usage.OutputTokens != 0 {
		return empty, ragy.ErrInvalidArgument
	}
	lease, err := ledger.Reserve(child, quote)
	if err != nil {
		return budgetStop[TMeta](child, request.Read, err)
	}
	if err = gate(); err != nil {
		_ = lease.Settle(budget.Usage{InputTokens: 0, OutputTokens: 0, Cost: 0}, false)
		return empty, err
	}
	evidence, callErr := r.config.Adapter.Traverse(child, call)
	settleErr := lease.Settle(quote.Usage, quote.CostKnown)
	if err = gate(); err != nil {
		return empty, err
	}
	if err = errors.Join(callErr, settleErr); err != nil {
		return empty, err
	}
	owned, err := r.snapshot(evidence, gate)
	if err != nil {
		return empty, err
	}
	if err = gate(); err != nil {
		return empty, err
	}
	outcome := recipe.Complete
	if len(owned.Snapshot.Edges) == 0 {
		outcome = recipe.Insufficient
	}
	return Result[TMeta]{Evidence: owned, Outcome: outcome, Stop: Expanded, GraphCalls: 1}, nil
}

func budgetStop[TMeta any](ctx context.Context, read access.Binding, err error) (Result[TMeta], error) {
	var empty Result[TMeta]
	if check := read.Check(ctx); check != nil {
		return empty, check
	}
	stop := BudgetExhausted
	if errors.Is(err, budget.ErrUnknownPrice) {
		stop = PriceUnavailable
	} else if !errors.Is(err, budget.ErrExhausted) {
		return empty, err
	}
	return Result[TMeta]{
		Evidence: managed.Result[TMeta]{
			Snapshot:  graph.Snapshot[TMeta]{Nodes: nil, Edges: nil},
			Supports:  nil,
			Conflicts: nil,
		},
		Outcome:    recipe.Insufficient,
		Stop:       stop,
		GraphCalls: 0,
	}, nil
}

func (r *Recipe[TMeta]) snapshot(input managed.Result[TMeta], gate func() error) (managed.Result[TMeta], error) {
	var empty managed.Result[TMeta]
	owned := managed.Result[TMeta]{
		Snapshot: graph.Snapshot[TMeta]{
			Nodes: slices.Clone(input.Snapshot.Nodes),
			Edges: slices.Clone(input.Snapshot.Edges),
		},
		Supports:  slices.Clone(input.Supports),
		Conflicts: slices.Clone(input.Conflicts),
	}
	for i, node := range owned.Snapshot.Nodes {
		if err := gate(); err != nil {
			return empty, err
		}
		meta, err := r.config.CloneMeta(node.Meta)
		if err != nil {
			return empty, err
		}
		owned.Snapshot.Nodes[i].Meta = meta
		owned.Snapshot.Nodes[i].Labels = slices.Clone(node.Labels)
	}
	for i, edge := range owned.Snapshot.Edges {
		if err := gate(); err != nil {
			return empty, err
		}
		meta, err := r.config.CloneMeta(edge.Meta)
		if err != nil {
			return empty, err
		}
		owned.Snapshot.Edges[i].Meta = meta
	}
	for i, support := range owned.Supports {
		owned.Supports[i].References = slices.Clone(support.References)
		owned.Supports[i].HostBases = slices.Clone(support.HostBases)
	}
	for i, conflict := range owned.Conflicts {
		owned.Conflicts[i].References = slices.Clone(conflict.References)
		owned.Conflicts[i].HostBases = slices.Clone(conflict.HostBases)
	}
	return owned, nil
}

// SourceReferences exports only observed original supports, never graph IDs as
// synthetic source citations. Explicit host foundations may have no source refs.
func (r Result[TMeta]) SourceReferences() []source.Reference {
	var result []source.Reference
	for _, support := range r.Evidence.Supports {
		for _, reference := range support.References {
			if !slices.Contains(result, reference) {
				result = append(result, reference)
			}
		}
	}
	return result
}
