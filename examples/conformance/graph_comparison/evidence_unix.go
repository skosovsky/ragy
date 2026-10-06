//go:build darwin || linux

package main

import (
	"context"
	"slices"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/evidence"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

type captureSink struct{}

func (captureSink) Write(context.Context, evidence.Record) error { return nil }

func (s summarySources) evidenceInput(q query, sample observation) evidence.Input[baselineMetadata] {
	outcome, reason := supportOutcome(sample)
	var hits []evidence.Hit[baselineMetadata]
	for i, ref := range sample.Supports {
		hits = append(hits, evidence.Hit[baselineMetadata]{
			Document: retrieval.Document[baselineMetadata]{
				ID:   ref.Source,
				Rank: i + 1,
				Meta: baselineMetadata{Tenant: "a", SourceID: ref.Source},
			},
			Sources: []source.Reference{ref},
		})
	}
	// The source list has a delivered order, not a native graph similarity score.
	stages := unobservedStages(sample.Profile)
	for _, observed := range cloneObservedStages(sample.stages) {
		for i := range stages {
			if stages[i].Name == observed.Name {
				stages[i] = observed
				break
			}
		}
	}
	stages = append(stages, evidence.Stage[baselineMetadata]{Name: supportStage, Status: evidence.StageObserved,
		Scores: evidence.Observed, Sources: evidence.Observed, Judgments: evidence.Unavailable, Hits: hits})
	return evidence.Input[baselineMetadata]{
		Schema: s.schema,
		Codec:  retrieval.NewJSONCodec[baselineMetadata](s.schema),
		SourceAdmission: func(ctx context.Context, read access.Binding, ref source.Reference) error {
			values, err := s.reader.Lookup(ctx, source.LookupRequest{Read: read, References: []source.Reference{ref}})
			if err != nil {
				return err
			}
			if len(values) != 1 {
				return errInvalid
			}
			return read.Check(ctx)
		},
		RetrievalID:    q.ID + "/" + sample.Profile,
		RecipeRevision: sample.Configuration,
		Query:          q.Text,
		Outcome:        outcome,
		Reason:         reason,
		Coverage:       retrieval.UnobservedReadCoverage(),
		Required:       []evidence.Field{evidence.ScopeField, evidence.PublicationField},
		Stages:         stages,
		Diagnostics: []evidence.Diagnostic{
			{Kind: evidence.ModelCalls, Number: diagnosticNumber(sample.ModelCalls, sample.CallsKnown)},
			{
				Kind:   evidence.RetrievalCalls,
				Number: diagnosticRetrievalCalls(sample),
			},
			{Kind: evidence.InputTokens, Number: diagnosticNumber(sample.InputTokens, sample.UsageKnown)},
			{Kind: evidence.OutputTokens, Number: diagnosticNumber(sample.OutputTokens, sample.UsageKnown)},
			{Kind: evidence.CostUnits, Number: diagnosticNumber(sample.Cost, sample.UsageKnown)},
		},
	}
}
func diagnosticNumber(value uint64, known bool) evidence.Number {
	if !known || value > 1<<53-1 {
		return evidence.Number{State: evidence.Unavailable}
	}
	number := float64(value)
	return evidence.Number{State: evidence.Observed, Value: &number}
}
func unobservedStages(profile string) []evidence.Stage[baselineMetadata] {
	names := []string{profile}
	switch profile {
	case baselineProfile:
		names = []string{"dense", lexicalTarget, "rrf"}
	case globalProfile:
		names = []string{"community-map-C1", "community-map-C2", "global-reduce"}
	}
	var stages []evidence.Stage[baselineMetadata]
	for _, name := range names {
		stages = append(stages, evidence.Stage[baselineMetadata]{Name: name, Status: evidence.MissingObservation})
	}
	return stages
}
func supportPolicy(read access.Binding, q query, sample observation) evidence.Policy {
	allowed := map[string]bool{
		q.ID + "/" + sample.Profile:    true,
		read.Snapshot().Identity:       true,
		read.Publication().Reference(): true,
		supportStage:                   true,
	}
	if sample.Configuration != "" {
		allowed[sample.Configuration] = true
	}
	for _, ref := range sample.Supports {
		for _, id := range []string{ref.Namespace, ref.Source, ref.Revision, ref.Transformation, ref.Artifact, ref.Representation} {
			allowed[id] = true
		}
	}
	for _, stage := range unobservedStages(sample.Profile) {
		allowed[stage.Name] = true
	}
	for _, stage := range sample.stages {
		allowed[stage.Name] = true
		for _, hit := range stage.Hits {
			allowed[hit.Document.ID] = true
			if hit.Document.ScoreSemantics != "" {
				allowed[string(hit.Document.ScoreSemantics)] = true
			}
			for _, ref := range hit.Sources {
				for _, id := range []string{ref.Namespace, ref.Source, ref.Revision, ref.Transformation, ref.Artifact, ref.Representation} {
					allowed[id] = true
				}
			}
		}
	}
	return evidence.Policy{
		AllowIdentifier: func(_ evidence.IdentifierKind, id string) bool { return allowed[id] },
		AllowNumbers:    true,
	}
}

func (s summarySources) recordObservation(
	ctx context.Context,
	read access.Binding,
	q query,
	mode evidence.Mode,
	sink evidence.Sink,
	execute func(context.Context) (observation, error),
) (evidence.Execution[observation], error) {
	if mode != evidence.Disabled {
		var err error
		s, err = s.freshSources()
		if err != nil {
			return evidence.Execution[observation]{}, err
		}
	}
	execution, err := evidence.Run(
		ctx,
		read,
		evidence.RecordingConfig[observation]{
			Mode: mode,
			Sink: sink,
			Execute: func(ctx context.Context) (observation, error) {
				sample, runErr := execute(ctx)
				if sample.Configuration == "" {
					_, id, configErr := configurationBytes()
					if configErr != nil {
						return observation{}, configErr
					}
					sample.Configuration = id
				}
				return sample, runErr
			},
			CloneResult: func(sample observation) (observation, error) {
				sample.stages = cloneObservedStages(sample.stages)
				sample.Supports = slices.Clone(sample.Supports)
				sample.Evidence = slices.Clone(sample.Evidence)
				return sample, nil
			},
			Capture: func(ctx context.Context, read access.Binding, sample observation, _ error) (evidence.Record, error) {
				return evidence.Capture(ctx, read, s.evidenceInput(q, sample), supportPolicy(read, q, sample))
			},
		},
	)
	if access.IsProtectionFailure(err) {
		return evidence.Execution[observation]{}, err
	}
	if mode != evidence.Disabled {
		execution.Result.SourceMetadataCalls += s.host.metadataCalls.Load()
		execution.Result.SourcePayloadCalls += s.host.payloadCalls.Load()
	}
	return execution, err
}

func diagnosticRetrievalCalls(sample observation) evidence.Number {
	if sample.RetrievalCalls > ^uint64(0)-sample.GraphCalls {
		return evidence.Number{State: evidence.Unavailable}
	}
	return diagnosticNumber(sample.RetrievalCalls+sample.GraphCalls, sample.CallsKnown)
}

func cloneObservedStages(input []evidence.Stage[baselineMetadata]) []evidence.Stage[baselineMetadata] {
	out := slices.Clone(input)
	for i := range out {
		out[i].Hits = slices.Clone(out[i].Hits)
		for j := range out[i].Hits {
			hit := &out[i].Hits[j]
			hit.Sources = slices.Clone(hit.Sources)
			hit.Document.ScoreHistory = slices.Clone(hit.Document.ScoreHistory)
			hit.Document.SourceSupports = slices.Clone(hit.Document.SourceSupports)
		}
	}
	return out
}
