// Package graphingest composes typed source-bound graph stages. It prepares a
// managed payload and planned manifest; the host explicitly drives lifecycle writes.
package graphingest

import (
	"context"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/graphingest/extraction"
	"github.com/skosovsky/ragy/graphingest/materialization"
	"github.com/skosovsky/ragy/graphingest/resolution"
	"github.com/skosovsky/ragy/recipe/budget"
)

// Extractor implements bounded extraction under the host's attempt ledger.
type Extractor[TAccess any, TKind, TRel comparable, TAttr any] interface {
	Extract(
		context.Context,
		access.Binding,
		*budget.Ledger,
		[]extraction.Snippet[TAccess],
	) (extraction.Result[TKind, TRel, TAttr], error)
}

// Resolver applies explicit host ontology/identity/support policies.
type Resolver[TKind, TRel comparable, TAttr any] interface {
	Resolve(
		context.Context,
		access.Binding,
		resolution.Extraction[TKind, TRel, TAttr],
	) (resolution.Result[TKind, TRel, TAttr], error)
}

// Materializer builds a planned inventory without publishing or writing indexes.
type Materializer[TKind, TRel comparable, TAttr, TMeta any] interface {
	Build(
		context.Context,
		access.Binding,
		materialization.Request,
		resolution.Result[TKind, TRel, TAttr],
	) (materialization.Result[TMeta], error)
}

type Config[TAccess any, TKind, TRel comparable, TAttr, TMeta any] struct {
	Extraction      Extractor[TAccess, TKind, TRel, TAttr]
	Resolution      Resolver[TKind, TRel, TAttr]
	Materialization Materializer[TKind, TRel, TAttr, TMeta]
}
type Pipeline[TAccess any, TKind, TRel comparable, TAttr, TMeta any] struct {
	config Config[TAccess, TKind, TRel, TAttr, TMeta]
}
type Result[TKind, TRel comparable, TAttr, TMeta any] struct {
	Plan      materialization.Result[TMeta]
	Decisions resolution.Result[TKind, TRel, TAttr]
	Usage     extraction.Usage
}

func New[TAccess any, TKind, TRel comparable, TAttr, TMeta any](
	config Config[TAccess, TKind, TRel, TAttr, TMeta],
) (*Pipeline[TAccess, TKind, TRel, TAttr, TMeta], error) {
	if config.Extraction == nil || config.Resolution == nil || config.Materialization == nil {
		return nil, ragy.ErrInvalidArgument
	}
	return &Pipeline[TAccess, TKind, TRel, TAttr, TMeta]{config: config}, nil
}

// Build requires immutable host input and cooperative ports. Each stage enforces
// its own complete contract and supplies owned output. No failed stage returns a
// publishable plan. Hosts pass Plan to an explicit lifecycle.Executor handoff.
func (p *Pipeline[TAccess, TKind, TRel, TAttr, TMeta]) Build(
	ctx context.Context,
	read access.Binding,
	ledger *budget.Ledger,
	input []extraction.Snippet[TAccess],
	request materialization.Request,
) (Result[TKind, TRel, TAttr, TMeta], error) {
	var empty Result[TKind, TRel, TAttr, TMeta]
	if p == nil || ledger == nil {
		return empty, ragy.ErrInvalidArgument
	}
	if err := read.Check(ctx); err != nil {
		return empty, err
	}
	extracted, err := p.config.Extraction.Extract(ctx, read, ledger, input)
	if err != nil {
		return empty, fmt.Errorf("graph ingestion extraction: %w", access.NonSkippable(err))
	}
	if err = read.Check(ctx); err != nil {
		return empty, err
	}
	resolved, err := p.config.Resolution.Resolve(ctx, read, extracted.Extraction)
	if err != nil {
		return empty, fmt.Errorf("graph ingestion resolution: %w", access.NonSkippable(err))
	}
	if err = read.Check(ctx); err != nil {
		return empty, err
	}
	plan, err := p.config.Materialization.Build(ctx, read, request, resolved)
	if err != nil {
		return empty, fmt.Errorf("graph ingestion materialization: %w", access.NonSkippable(err))
	}
	if err = read.Check(ctx); err != nil {
		return empty, err
	}
	return Result[TKind, TRel, TAttr, TMeta]{Plan: plan, Decisions: resolved, Usage: extracted.Usage}, nil
}
