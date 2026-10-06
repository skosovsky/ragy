//go:build darwin || linux

package main

import (
	"context"
	"time"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/recipe/graphexpand"
)

func (c graphCorpus) local(parent context.Context, read access.Binding, q query) (observation, error) {
	ctx, cancel := context.WithTimeout(parent, attemptDuration)
	defer cancel()
	started := time.Now()
	if q.Recipe != localProfile {
		return observation{}, errInvalid
	}
	if err := read.Check(ctx); err != nil {
		return observation{}, err
	}
	seed := ""
	for _, entity := range c.resolved.Entities {
		if entity.Identity.Namespace == productionNamespace && entity.Identity.Key == "Team/Team A" {
			seed = entity.ID
			break
		}
	}
	if seed == "" {
		return observation{}, errInvalid
	}
	ledger, err := budget.New(budget.Config{Limits: budget.Limits{
		RetrievalCalls: localCallCap,
		ModelCalls:     0,
		Usage: budget.Usage{
			InputTokens:  referenceInputCap,
			OutputTokens: referenceOutputCap,
			Cost:         referenceCostCap,
		},
	}, Deadline: started.Add(attemptDuration), Now: time.Now, RequireKnownCost: true})
	if err != nil {
		return observation{}, err
	}
	instance, err := graphexpand.New(
		graphexpand.Config[graphMetadata]{
			Adapter:  c.adapter,
			MaxDepth: localDepth,
			MaxNodes: localNodeCap,
			MaxEdges: localEdgeCap,
			Duration: attemptDuration,
			Now:      time.Now,
			Quote: func(context.Context) (budget.Reservation, error) {
				return budget.Reservation{Kind: budget.Retrieval, CostKnown: true}, nil
			},
			CloneMeta: func(m graphMetadata) (graphMetadata, error) { return m, nil },
		},
	)
	if err != nil {
		return observation{}, err
	}
	result, runErr := instance.Run(
		ctx,
		graphexpand.Request{
			Read: read,
			Traversal: graph.TraversalRequest{
				Seeds:     []string{seed},
				Direction: graph.DirectionUndirected,
				Depth:     localDepth,
			},
		},
		ledger,
	)
	snapshot := ledger.Snapshot()
	sample := observation{
		Query:        q.ID,
		Profile:      localProfile,
		Scope:        read.Snapshot().Identity,
		Publication:  read.Publication().Reference(),
		Nanos:        time.Since(started).Nanoseconds(),
		InputTokens:  snapshot.Actual.InputTokens,
		OutputTokens: snapshot.Actual.OutputTokens,
		Cost:         snapshot.Actual.Cost,
		UsageKnown:   snapshot.UnknownUsage == 0 && !snapshot.UnknownCost,
		CallsKnown:   runErr == nil,
		GraphCalls:   result.GraphCalls,
		ModelCalls:   snapshot.Occupied.ModelCalls,
		Outcome:      string(result.Outcome),
		Stop:         string(result.Stop),
	}
	if runErr != nil {
		sample.Failed = true
		sample.Outcome = failedOutcome
		sample.Stop = executionErrorStop
		return sample, nil
	}
	if err = read.Check(ctx); err != nil {
		return observation{}, err
	}
	sample.stages = append(sample.stages, observedGraphStage(result.Evidence))
	sample.Supports = result.SourceReferences()
	return sample, nil
}
