//go:build darwin || linux

package main

import (
	"context"
	"slices"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/evidence"
	"github.com/skosovsky/ragy/graph/managed"
	"github.com/skosovsky/ragy/recipe/graphsummary"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

// Graph facts are unranked; snapshot iteration order is not a similarity ranking.
func observedGraphStage(result managed.Result[graphMetadata]) evidence.Stage[baselineMetadata] {
	stage := evidence.Stage[baselineMetadata]{Name: localProfile, Status: evidence.StageObserved,
		Scores: evidence.Observed, Sources: evidence.Observed, Judgments: evidence.Unavailable}
	appendFact := func(kind, id string, meta graphMetadata) {
		var refs []source.Reference
		for _, support := range result.Supports {
			if support.Kind == kind && support.ID == id {
				refs = append(refs, support.References...)
			}
		}
		stage.Hits = append(stage.Hits, evidence.Hit[baselineMetadata]{
			Document: retrieval.Document[baselineMetadata]{
				ID:   id,
				Meta: baselineMetadata{Tenant: meta.Tenant, SourceID: meta.SourceID},
			},
			Sources: slices.Clone(refs),
		})
	}
	for _, node := range result.Snapshot.Nodes {
		appendFact("node", node.ID, node.Meta)
	}
	for _, edge := range result.Snapshot.Edges {
		appendFact("edge", edge.ID, edge.Meta)
	}
	return stage
}

func (s summarySources) summaryStages(
	ctx context.Context,
	read access.Binding,
	result graphsummary.Result,
	global bool,
) ([]evidence.Stage[baselineMetadata], error) {
	var stages []evidence.Stage[baselineMetadata]
	for _, summary := range result.Communities {
		ids := summary.CommunityIDs()
		if len(ids) != 1 {
			return nil, errInvalid
		}
		name := communityProfile
		if global {
			name = "community-map-" + ids[0]
		}
		stage, err := s.summaryStage(ctx, read, name, summary)
		if err != nil {
			return nil, err
		}
		stages = append(stages, stage)
	}
	if result.Global != nil {
		stage, err := s.summaryStage(ctx, read, "global-reduce", *result.Global)
		if err != nil {
			return nil, err
		}
		stages = append(stages, stage)
	}
	return stages, read.Check(ctx)
}

// Summary stages observe selected original associations, not a graded or ranked answer.
func (s summarySources) summaryStage(
	ctx context.Context,
	read access.Binding,
	name string,
	summary graphsummary.Summary,
) (evidence.Stage[baselineMetadata], error) {
	var empty evidence.Stage[baselineMetadata]
	mapping, err := summary.Resolve(ctx, read, s.admitSource)
	if err != nil {
		return empty, err
	}
	stage := evidence.Stage[baselineMetadata]{Name: name, Status: evidence.StageObserved,
		Scores: evidence.Observed, Sources: evidence.Observed, Judgments: evidence.Unavailable}
	var seen []source.Reference
	for _, loc := range mapping.Supports() {
		if slices.Contains(seen, loc.Reference) {
			continue
		}
		seen = append(seen, loc.Reference)
		// Admission has checked actual original payload/access; metadata is its fixture descriptor.
		stage.Hits = append(stage.Hits, evidence.Hit[baselineMetadata]{
			Document: retrieval.Document[baselineMetadata]{
				ID:   loc.Reference.Source,
				Meta: baselineMetadata{Tenant: "a", SourceID: loc.Reference.Source},
			},
			Sources: []source.Reference{loc.Reference},
		})
	}
	return stage, nil
}
