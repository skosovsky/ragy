package structured

import (
	"context"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/graphingest/extraction"
	"github.com/skosovsky/ragy/recipe/budget"
)

// Extractor binds the HTTP transport to the core typed extraction client port.
// Price returns actual host-defined integer units. It must match the reservation's
// pricing policy. No provider price catalog or retry policy is introduced here.
type Extractor[TKind, TRel comparable, TAttr any] struct {
	client *Client[extraction.ModelOutput[TKind, TRel, TAttr]]
	price  func(Usage) (uint64, error)
}

func NewExtractor[TKind, TRel comparable, TAttr any](
	cfg Config, price func(Usage) (uint64, error),
) (*Extractor[TKind, TRel, TAttr], error) {
	if price == nil {
		return nil, ragy.ErrInvalidArgument
	}
	client, err := New[extraction.ModelOutput[TKind, TRel, TAttr]](cfg)
	if err != nil {
		return nil, err
	}
	return &Extractor[TKind, TRel, TAttr]{client: client, price: price}, nil
}

func (e *Extractor[TKind, TRel, TAttr]) CountInputTokens(input extraction.ModelInput) (uint64, error) {
	if e == nil {
		return 0, ragy.ErrInvalidArgument
	}
	return e.client.CountInputTokens(input, input.MaxOutputTokens)
}

func (e *Extractor[TKind, TRel, TAttr]) Model(
	ctx context.Context, input extraction.ModelInput,
) (extraction.ModelOutput[TKind, TRel, TAttr], extraction.Usage, error) {
	var empty extraction.ModelOutput[TKind, TRel, TAttr]
	if e == nil {
		return empty, extraction.Usage{}, ragy.ErrInvalidArgument
	}
	output, tokens, err := e.client.Call(ctx, input, Limits{
		InputTokens: input.MaxInputTokens, OutputTokens: input.MaxOutputTokens,
	})
	usage := extraction.Usage{
		Value: budget.Usage{InputTokens: tokens.InputTokens, OutputTokens: tokens.OutputTokens, Cost: 0},
		Known: false,
	}
	if tokens.Known {
		cost, priceErr := e.price(tokens)
		if priceErr == nil {
			usage.Value.Cost, usage.Known = cost, true
		}
	}
	if ctx != nil && ctx.Err() != nil {
		return empty, usage, ctx.Err()
	}
	return output, usage, err
}
