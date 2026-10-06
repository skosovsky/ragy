//go:build darwin || linux

package main

import (
	"context"
	"encoding/json"

	"github.com/skosovsky/ragy/adapters/openai/structured"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/recipe/graphsummary"
)

// Each factory binds the context-free counter to this particular host attempt.
func newProviderSummaryPorts(
	ctx context.Context,
	cfg structured.Config,
	counter func(context.Context, []byte) (uint64, error),
) (summaryModelPorts, error) {
	if ctx == nil || counter == nil {
		return summaryModelPorts{}, errInvalid
	}
	if _, ok := ctx.Deadline(); !ok {
		return summaryModelPorts{}, errInvalid
	}
	cfg.CountTokens = func(request []byte) (uint64, error) { return counter(ctx, request) }
	cfg.Instructions = summaryInstructions
	cfg.SchemaName = "graph_summary"
	cfg.Schema = json.RawMessage(summarySchema)
	cfg.Validate = validateSummaryOutput
	cfg.Duration = attemptDuration
	cfg.MaxRequestBytes = summaryInputBytes
	cfg.MaxResponseBytes = summaryInputBytes
	client, transportCalls := observeModelClient(cfg.HTTPClient)
	cfg.HTTPClient = client
	summarizer, err := structured.NewSummarizer(
		cfg,
		func(structured.Usage) (uint64, error) { return summaryCallCost, nil },
	)
	if err != nil {
		return summaryModelPorts{}, err
	}
	return summaryModelPorts{
		model:          summarizer.Model,
		count:          summarizer.CountInputTokens,
		quote:          summaryQuote,
		transportCalls: transportCalls,
		modelIdentity:  cfg.Model,
		configuration:  providerSummaryIdentity(cfg.Model, cfg.BaseURL),
	}, nil
}
func summaryQuote(_ context.Context, stage graphsummary.Stage) (budget.Reservation, error) {
	input, output := uint64(summaryMapInput), uint64(summaryMapOutput)
	if stage == graphsummary.Reduce {
		input, output = summaryReduceInput, summaryReduceOutput
	} else if stage != graphsummary.Map {
		return budget.Reservation{}, errInvalid
	}
	return budget.Reservation{
		Kind:      budget.Model,
		CostKnown: true,
		Usage:     budget.Usage{InputTokens: input, OutputTokens: output, Cost: summaryCallCost},
	}, nil
}
func validateSummaryOutput(raw json.RawMessage) error {
	var out struct {
		Text     *string `json:"text"`
		Selected []int   `json:"selected"`
	}
	if err := decodeStrict(raw, &out); err != nil || out.Text == nil || out.Selected == nil {
		return errInvalid
	}
	return nil
}
