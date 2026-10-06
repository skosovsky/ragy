//go:build darwin || linux

package main

import (
	"context"
	"encoding/json"
	"slices"

	"example.com/ragyconsumer/internal/codexcall"

	"github.com/skosovsky/ragy/graphingest/extraction"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/recipe/graphsummary"
)

// This consumer deliberately exposes its estimate/advisory limitations. It does
// not qualify CLI input/output enforcement or supply a required-budget adapter.
type codexGraphPorts struct {
	config codexcall.Config
	calls  []codexcall.Result
}

func (p *codexGraphPorts) receipts() []codexcall.Result { return slices.Clone(p.calls) }
func (p *codexGraphPorts) call(ctx context.Context, instructions, schema string, input any) (codexcall.Result, error) {
	result, err := codexcall.Call(ctx, p.config, instructions, json.RawMessage(schema), input)
	p.calls = append(p.calls, result)
	return result, err
}
func estimatedCLIInput(input any) (uint64, error) {
	raw, err := json.Marshal(input)
	if err != nil || len(raw) > summaryInputBytes {
		return 0, errInvalid
	}
	// JSON bytes plus a fixed calibration margin are only a scheduling estimate.
	// The full executor context cannot be bounded by this callback.
	return cliCalibrationInputMargin + uint64(len(raw)), nil
}
func cliUsage(call codexcall.Result) (budget.Usage, bool) {
	if call.Usage == nil {
		return budget.Usage{}, false
	}
	return budget.Usage{InputTokens: call.Usage.Input, OutputTokens: call.Usage.Output}, true
}
func (p *codexGraphPorts) extractionPorts() extractionPorts {
	return extractionPorts{
		advisory:      true,
		receipts:      p.receipts,
		modelIdentity: p.config.Model,
		configuration: digest([]byte(extractionConfigurationIdentity() + p.config.Model + "codex-advisory")),
		count:         func(input extraction.ModelInput) (uint64, error) { return estimatedCLIInput(input) },
		model: func(ctx context.Context, input extraction.ModelInput) (extraction.ModelOutput[string, string, graphAttributes], extraction.Usage, error) {
			call, err := p.call(ctx, extractionInstructions, extractionSchema, input)
			value, known := cliUsage(call)
			usage := extraction.Usage{Value: value, Known: known}
			var out extraction.ModelOutput[string, string, graphAttributes]
			if err != nil {
				return out, usage, err
			}
			if err = validateExtractionOutput(call.Output); err != nil {
				return out, usage, err
			}
			err = codexcall.Decode(call.Output, &out)
			return out, usage, err
		},
	}
}
func (p *codexGraphPorts) summaryPorts() summaryModelPorts {
	return summaryModelPorts{
		advisory:      true,
		receipts:      p.receipts,
		modelIdentity: p.config.Model,
		configuration: digest([]byte(summaryInstructions + summarySchema + p.config.Model + "codex-advisory")),
		count:         func(input graphsummary.ModelInput) (uint64, error) { return estimatedCLIInput(input) },
		quote: func(_ context.Context, stage graphsummary.Stage) (budget.Reservation, error) {
			if stage != graphsummary.Map && stage != graphsummary.Reduce {
				return budget.Reservation{}, errInvalid
			}
			return budget.Reservation{
				Kind:  budget.Model,
				Usage: budget.Usage{InputTokens: cliInputReservation, OutputTokens: cliOutputReservation},
			}, nil
		},
		model: func(ctx context.Context, input graphsummary.ModelInput) (graphsummary.ModelOutput, graphsummary.Usage, error) {
			call, err := p.call(ctx, summaryInstructions, summarySchema, input)
			value, known := cliUsage(call)
			usage := graphsummary.Usage{Value: value, Known: known}
			var out graphsummary.ModelOutput
			if err != nil {
				return out, usage, err
			}
			if err = validateSummaryOutput(call.Output); err != nil {
				return out, usage, err
			}
			err = codexcall.Decode(call.Output, &out)
			return out, usage, err
		},
	}
}
