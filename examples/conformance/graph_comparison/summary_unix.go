//go:build darwin || linux

package main

import (
	"context"
	"slices"
	"time"

	"github.com/skosovsky/ragy/examples/conformance/internal/codexcall"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/recipe/graphsummary"
	"github.com/skosovsky/ragy/source"
)

const summaryMaxModelCalls = 3

// summaryModelPorts is supplied by the consumer. The token counter qualifies the
// entire provider request, and each Model invocation dispatches at most once.
type summaryModelPorts struct {
	modelIdentity  string
	configuration  string
	transportCalls func() uint64
	advisory       bool
	receipts       func() []codexcall.Result
	model          graphsummary.Model
	count          func(graphsummary.ModelInput) (uint64, error)
	quote          func(context.Context, graphsummary.Stage) (budget.Reservation, error)
}

func (s summarySources) summary(
	parent context.Context,
	read access.Binding,
	q query,
	ports summaryModelPorts,
) (observation, error) {
	if q.Recipe != communityProfile && q.Recipe != globalProfile {
		return observation{}, errInvalid
	}
	duration := attemptDuration
	if ports.advisory {
		duration = cliAttemptDuration
	}
	ctx, cancel := context.WithTimeout(parent, duration)
	defer cancel()
	started := time.Now()
	if err := read.Check(ctx); err != nil {
		return observation{}, err
	}
	if ports.model == nil || ports.count == nil || ports.quote == nil {
		return observation{}, errInvalid
	}
	var err error
	s, err = s.freshSources()
	if err != nil {
		return observation{}, err
	}
	communities, err := s.communities(ctx, read, q.Recipe == globalProfile)
	if err != nil {
		return observation{}, err
	}
	calls := uint64(0)
	instance, err := s.summaryRecipe(ports, &calls)
	if err != nil {
		return observation{}, err
	}
	callCap := uint64(communityCallCap)
	if q.Recipe == globalProfile {
		callCap = globalCallCap
	}
	ledger, err := summaryLedger(ports.advisory, callCap, started, duration)
	if err != nil {
		return observation{}, err
	}
	request := graphsummary.Request[baselineMetadata]{Read: read, Question: q.Text, Communities: communities}
	var result graphsummary.Result
	if q.Recipe == globalProfile {
		result, err = instance.Global(ctx, request, ledger)
	} else {
		result, err = instance.Community(ctx, request, ledger)
	}
	sample := observation{
		Configuration: ports.configuration,
		Query:         q.ID,
		Profile:       q.Recipe,
		Scope:         read.Snapshot().Identity,
		Publication:   read.Publication().Reference(),
		CallsKnown:    true,
		ModelCalls:    calls,
		Outcome:       string(result.Outcome),
		Stop:          string(result.Stop),
	}
	if err == nil {
		sample.Supports, err = s.summarySupports(ctx, read, result)
	}
	if err == nil {
		sample.stages, err = s.summaryStages(ctx, read, result, q.Recipe == globalProfile)
	}
	if err != nil {
		sample.Failed = true
		sample.Supports = nil
		sample.Outcome = failedOutcome
		sample.Stop = executionErrorStop
	}
	s.summaryAccounting(ports, ledger, &sample)
	sample.Nanos = time.Since(started).Nanoseconds()
	return sample, nil
}

func (s summarySources) summaryRecipe(
	ports summaryModelPorts,
	calls *uint64,
) (*graphsummary.Recipe[baselineMetadata], error) {
	return graphsummary.New(graphsummary.Config[baselineMetadata]{
		Schema:           s.schema,
		MaxCommunities:   2,
		MaxModelCalls:    summaryMaxModelCalls,
		MaxMembers:       localNodeCap,
		MaxSnippets:      communitySnippetCap,
		MaxSupports:      globalSnippetCap,
		MaxInputBytes:    summaryInputBytes,
		MaxSummaryBytes:  summaryOutputBytes,
		Duration:         summaryPortDuration(ports),
		Now:              time.Now,
		CloneAccess:      func(m baselineMetadata) (baselineMetadata, error) { return m, nil },
		Attributes:       summaryAttributes,
		Membership:       s.admitMembership,
		AdmitSource:      s.admitSource,
		Quote:            ports.quote,
		CountInputTokens: ports.count,
		Model: func(ctx context.Context, input graphsummary.ModelInput) (graphsummary.ModelOutput, graphsummary.Usage, error) {
			*calls++
			return ports.model(ctx, input)
		},
	})
}

func (s summarySources) summarySupports(
	ctx context.Context,
	read access.Binding,
	result graphsummary.Result,
) ([]source.Reference, error) {
	var summaries []graphsummary.Summary
	if result.Global != nil {
		summaries = []graphsummary.Summary{*result.Global}
	} else {
		summaries = result.Communities
	}
	var supports []source.Reference
	for _, item := range summaries {
		mapping, err := item.Resolve(ctx, read, s.admitSource)
		if err != nil {
			return nil, err
		}
		for _, loc := range mapping.Supports() {
			if !slices.Contains(supports, loc.Reference) {
				supports = append(supports, loc.Reference)
			}
		}
	}
	if err := read.Check(ctx); err != nil {
		return nil, err
	}
	return supports, nil
}

func summaryPortDuration(ports summaryModelPorts) time.Duration {
	if ports.advisory {
		return cliAttemptDuration
	}
	return attemptDuration
}

func summaryLedger(advisory bool, callCap uint64, started time.Time, duration time.Duration) (*budget.Ledger, error) {
	inputCap, outputCap, costCap := uint64(referenceInputCap), uint64(referenceOutputCap), uint64(referenceCostCap)
	if advisory {
		inputCap, outputCap, costCap = cliInputReservation*callCap, cliOutputReservation*callCap, 0
	}
	return budget.New(budget.Config{Limits: budget.Limits{ModelCalls: callCap, Usage: budget.Usage{
		InputTokens: inputCap, OutputTokens: outputCap, Cost: costCap,
	}}, Deadline: started.Add(duration), Now: time.Now, RequireKnownCost: !advisory})
}

func (s summarySources) summaryAccounting(ports summaryModelPorts, ledger *budget.Ledger, sample *observation) {
	snapshot := ledger.Snapshot()
	sample.InputTokens, sample.OutputTokens, sample.Cost = snapshot.Actual.InputTokens, snapshot.Actual.OutputTokens, snapshot.Actual.Cost
	sample.UsageKnown = snapshot.UnknownUsage == 0 && !snapshot.UnknownCost
	sample.SourceMetadataCalls = s.host.metadataCalls.Load()
	sample.SourcePayloadCalls = s.host.payloadCalls.Load()
	if ports.transportCalls != nil {
		sample.TransportCallsKnown = true
		sample.TransportCalls = ports.transportCalls()
	}
	if ports.receipts != nil {
		sample.CLIReceipts = ports.receipts()
		sample.ProviderPriceState = "unavailable"
		sample.InputTokens, sample.OutputTokens, sample.ReportedTokensKnown = reportedCLIUsage(sample.CLIReceipts)
	}
}
