package structured

import (
	"context"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/recipe/graphsummary"
)

// Summarizer binds structured HTTP output to community/global summary ports.
// The host supplies the executable output schema, exact tokenizer and actual price
// conversion consistent with core reservations. Canonical refs never go to models.
type Summarizer struct {
	client *Client[graphsummary.ModelOutput]
	price  func(Usage) (uint64, error)
}

func NewSummarizer(cfg Config, price func(Usage) (uint64, error)) (*Summarizer, error) {
	if price == nil {
		return nil, ragy.ErrInvalidArgument
	}
	client, err := New[graphsummary.ModelOutput](cfg)
	if err != nil {
		return nil, err
	}
	return &Summarizer{client: client, price: price}, nil
}

func (s *Summarizer) CountInputTokens(input graphsummary.ModelInput) (uint64, error) {
	if s == nil {
		return 0, ragy.ErrInvalidArgument
	}
	return s.client.CountInputTokens(input, input.MaxOutputTokens)
}

func (s *Summarizer) Model(
	ctx context.Context,
	input graphsummary.ModelInput,
) (graphsummary.ModelOutput, graphsummary.Usage, error) {
	if s == nil {
		return graphsummary.ModelOutput{}, graphsummary.Usage{}, ragy.ErrInvalidArgument
	}
	output, tokens, err := s.client.Call(
		ctx,
		input,
		Limits{InputTokens: input.MaxInputTokens, OutputTokens: input.MaxOutputTokens},
	)
	usage := graphsummary.Usage{
		Value: budget.Usage{InputTokens: tokens.InputTokens, OutputTokens: tokens.OutputTokens, Cost: 0},
		Known: false,
	}
	if tokens.Known {
		cost, priceErr := s.price(tokens)
		if priceErr == nil {
			usage.Value.Cost, usage.Known = cost, true
		}
	}
	if ctx != nil && ctx.Err() != nil {
		return graphsummary.ModelOutput{}, usage, ctx.Err()
	}
	return output, usage, err
}
