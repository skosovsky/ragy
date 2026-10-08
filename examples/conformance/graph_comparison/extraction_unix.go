//go:build darwin || linux

package main

import (
	"context"
	"time"

	"github.com/skosovsky/ragy/examples/conformance/internal/codexcall"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graphingest/extraction"
	"github.com/skosovsky/ragy/recipe/budget"
)

type extractionPorts struct {
	configuration  string
	modelIdentity  string
	transportCalls func() uint64
	advisory       bool
	receipts       func() []codexcall.Result
	model          extraction.Model[string, string, graphAttributes]
	count          func(extraction.ModelInput) (uint64, error)
}

func extractSource(
	parent context.Context,
	read access.Binding,
	schema filter.Schema,
	row sourceRow,
	f fixture,
	ports extractionPorts,
) (sourceExtraction, extractionObservation, error) {
	duration := attemptDuration
	if ports.advisory {
		duration = cliAttemptDuration
	}
	ctx, cancel := context.WithTimeout(parent, duration)
	defer cancel()
	started := time.Now()
	configuration := ports.configuration
	if configuration == "" {
		configuration = extractionConfigurationIdentity()
	}
	ports.configuration = configuration
	observed := extractionObservation{Reference: originalReference(row.ID), Configuration: configuration}
	if err := read.Check(ctx); err != nil {
		return sourceExtraction{}, observed, err
	}
	if ports.model == nil || ports.count == nil {
		return sourceExtraction{}, observed, errInvalid
	}
	adapter, err := sourceExtractor(schema, row, f, ports, &observed.ModelCalls)
	if err != nil {
		return sourceExtraction{}, observed, err
	}
	inputCap, outputCap, costCap := uint64(
		extractionInputTokens,
	), uint64(
		extractionOutputTokens,
	), uint64(
		referenceCostCap,
	)
	if ports.advisory {
		inputCap, outputCap, costCap = cliInputReservation, cliOutputReservation, 0
	}
	ledger, err := budget.New(budget.Config{Limits: budget.Limits{ModelCalls: 1, Usage: budget.Usage{
		InputTokens: inputCap, OutputTokens: outputCap, Cost: costCap,
	}}, Deadline: started.Add(duration), Now: time.Now, RequireKnownCost: !ports.advisory})
	if err != nil {
		return sourceExtraction{}, observed, err
	}
	mapping, err := mappedSource(row)
	if err != nil {
		return sourceExtraction{}, observed, err
	}
	result, runErr := adapter.Extract(ctx, read, ledger, []extraction.Snippet[baselineMetadata]{
		{Namespace: row.Namespace, Mapping: mapping, Access: baselineMetadata{Tenant: "a", SourceID: row.ID}},
	})
	snapshot := ledger.Snapshot()
	observed.Failed = runErr != nil
	observed.InputTokens, observed.OutputTokens, observed.Cost = snapshot.Actual.InputTokens, snapshot.Actual.OutputTokens, snapshot.Actual.Cost
	observed.UsageKnown = snapshot.UnknownUsage == 0 && !snapshot.UnknownCost
	if ports.transportCalls != nil {
		observed.TransportCallsKnown = true
		observed.TransportCalls = ports.transportCalls()
	}
	if ports.receipts != nil {
		observed.CLIReceipts = ports.receipts()
		observed.ProviderPriceState = "unavailable"
		observed.InputTokens, observed.OutputTokens, observed.ReportedTokensKnown = reportedCLIUsage(
			observed.CLIReceipts,
		)
	}
	observed.Nanos = time.Since(started).Nanoseconds()
	if runErr != nil {
		return sourceExtraction{}, observed, runErr
	}
	return sourceExtraction{SourceID: row.ID, Configuration: configuration, Value: result.Extraction}, observed, nil
}

func sourceExtractor(
	schema filter.Schema,
	row sourceRow,
	f fixture,
	ports extractionPorts,
	calls *uint64,
) (*extraction.Adapter[baselineMetadata, string, string, graphAttributes], error) {
	controls := referenceConfiguration()
	return extraction.New(extraction.Config[baselineMetadata, string, string, graphAttributes]{
		OntologyIdentity: controls.Ontology,
		Configuration:    ports.configuration,
		Schema:           schema,
		MaxSnippets:      1,
		MaxInputBytes:    summaryInputBytes,
		MaxEntities:      localNodeCap,
		MaxRelations:     localEdgeCap,
		MaxSupports:      localEdgeCap,
		Duration:         extractionPortDuration(ports),
		Now:              time.Now,
		CloneAccess:      func(m baselineMetadata) (baselineMetadata, error) { return m, nil },
		Attributes:       summaryAttributes,
		AdmitSnippet: func(ctx context.Context, read access.Binding, snippet extraction.Snippet[baselineMetadata]) error {
			if snippet.Namespace != row.Namespace || snippet.Access.Tenant != "a" ||
				snippet.Access.SourceID != row.ID ||
				snippet.Mapping.Text() != row.Text {
				return errInvalid
			}
			for _, loc := range snippet.Mapping.Supports() {
				if err := originalAdmission(f)(ctx, read, loc); err != nil {
					return err
				}
			}
			return read.Check(ctx)
		},
		CloneAttributes:  func(a graphAttributes) (graphAttributes, error) { return a, nil },
		ValidateEntity:   validateEntityKind,
		ValidateRelation: validateRelationKind,
		Quote: func(context.Context) (budget.Reservation, error) {
			if ports.advisory {
				return budget.Reservation{
					Kind:  budget.Model,
					Usage: budget.Usage{InputTokens: cliInputReservation, OutputTokens: cliOutputReservation},
				}, nil
			}
			return budget.Reservation{Kind: budget.Model, CostKnown: true, Usage: budget.Usage{
				InputTokens: extractionInputTokens, OutputTokens: extractionOutputTokens, Cost: summaryCallCost,
			}}, nil
		},
		CountInputTokens: ports.count,
		Model: func(ctx context.Context, input extraction.ModelInput) (extraction.ModelOutput[string, string, graphAttributes], extraction.Usage, error) {
			*calls++
			return ports.model(ctx, input)
		},
	})
}

func extractionPortDuration(ports extractionPorts) time.Duration {
	if ports.advisory {
		return cliAttemptDuration
	}
	return attemptDuration
}
