//go:build darwin || linux

package main

import (
	"context"
	"path/filepath"
	"time"

	"github.com/skosovsky/ragy/evidence"

	"github.com/skosovsky/ragy/access"
)

type captureFactories struct {
	extraction func(context.Context) (extractionPorts, error)
	summary    func(context.Context) (summaryModelPorts, error)
}
type captureIdentity struct {
	execution string
	adapter   string
	model     string
	tokenizer string
	profile   *codexHostProfile
}

// executeGraphCapture records real producer observations; provider factories are
// constructed separately for each bounded attempt and own their transport tracker.
func executeGraphCapture(
	parent context.Context,
	root string,
	identity captureIdentity,
	factories captureFactories,
) (capture, error) {
	if factories.extraction == nil || factories.summary == nil {
		return capture{}, errInvalid
	}
	ctx, cancel := context.WithTimeout(parent, captureProfileDuration(identity.profile))
	defer cancel()
	var f fixture
	if err := decodeStrict(fixtureJSON, &f); err != nil {
		return capture{}, err
	}
	controls, configurationID, err := graphConfigurationBytes(identity.profile)
	if err != nil {
		return capture{}, err
	}
	raw := capture{
		HostProfile:       identity.profile,
		FixtureIdentity:   f.Identity,
		ExecutionKind:     identity.execution,
		AdapterIdentity:   identity.adapter,
		ModelIdentity:     identity.model,
		TokenizerIdentity: identity.tokenizer,
		Configuration:     controls,
		ConfigIdentity:    configurationID,
	}
	dense, err := buildCaptureDenseCorpus(ctx, filepath.Join(root, "dense"), f, identity.profile)
	if err != nil {
		return capture{}, err
	}
	read, err := dense.bind(ctx, nil)
	if err != nil {
		return capture{}, err
	}
	batches, observed, err := captureExtractions(ctx, dense, read, f, identity, factories.extraction)
	raw.Preparation.Extractions = observed
	if err != nil {
		return raw, err
	}
	corpus, err := buildGraphCorpus(ctx, filepath.Join(root, "graph"), dense, read, batches)
	if err != nil {
		return raw, err
	}
	raw.Preparation.ResolutionHistoryID = corpus.history.ID
	targets, err := corpus.targets(ctx)
	if err != nil {
		return raw, err
	}
	read, err = dense.bind(ctx, targets)
	if err != nil {
		return raw, err
	}
	baseline, err := dense.baseline(ctx, read)
	if err != nil {
		return raw, err
	}
	prepared, err := corpus.summarySources(ctx, read, f, baseline.lexical.Schema())
	if err != nil {
		return raw, err
	}
	raw.Preparation.MembershipGraphCalls = prepared.preparationGraphCalls
	for _, q := range f.Queries {
		base, runErr := captureBaseline(ctx, baseline, prepared, read, q)
		if runErr != nil {
			return raw, runErr
		}
		raw.Samples = append(raw.Samples, base)
		sample, runErr := captureGraphRecipe(ctx, corpus, prepared, read, q, identity, factories.summary)
		if runErr != nil {
			return raw, runErr
		}
		raw.Samples = append(raw.Samples, sample)
	}
	if _, err = evaluate(raw); err != nil {
		return raw, err
	}
	return raw, nil
}

func captureExtractions(
	ctx context.Context,
	dense denseCorpus,
	read access.Binding,
	f fixture,
	identity captureIdentity,
	factory func(context.Context) (extractionPorts, error),
) ([]sourceExtraction, []extractionObservation, error) {
	var batches []sourceExtraction
	var observations []extractionObservation
	for _, row := range f.Sources {
		duration := attemptDuration
		if identity.profile != nil {
			duration = cliAttemptDuration
		}
		child, cancel := context.WithTimeout(ctx, duration)
		ports, err := factory(child)
		if err != nil {
			cancel()
			return batches, observations, err
		}
		if ports.modelIdentity != "" && ports.modelIdentity != identity.model {
			cancel()
			return batches, observations, errInvalid
		}
		if ports.configuration != "" {
			ports.configuration = bindTokenizerIdentity(ports.configuration, identity.tokenizer)
		}
		batch, observed, err := extractSource(child, read, dense.schema, row, f, ports)
		cancel()
		observations = append(observations, observed)
		if err != nil {
			return batches, observations, err
		}
		batches = append(batches, batch)
	}
	return batches, observations, nil
}

func captureGraphRecipe(
	ctx context.Context,
	corpus graphCorpus,
	prepared summarySources,
	read access.Binding,
	q query,
	identity captureIdentity,
	factory func(context.Context) (summaryModelPorts, error),
) (observation, error) {
	duration := attemptDuration
	if identity.profile != nil {
		duration = cliAttemptDuration
	}
	child, cancel := context.WithTimeout(ctx, duration)
	defer cancel()
	started := time.Now()
	if q.Recipe == localProfile {
		return recordedAttempt(
			child,
			prepared,
			read,
			q,
			started,
			func(ctx context.Context) (observation, error) { return corpus.local(ctx, read, q) },
		)
	}

	ports, err := factory(child)
	if err != nil {
		return observation{}, err
	}
	if ports.modelIdentity != "" && ports.modelIdentity != identity.model {
		return observation{}, errInvalid
	}
	if ports.configuration != "" {
		ports.configuration = bindTokenizerIdentity(ports.configuration, identity.tokenizer)
	}
	return recordedAttempt(
		child,
		prepared,
		read,
		q,
		started,
		func(ctx context.Context) (observation, error) { return prepared.summary(ctx, read, q, ports) },
	)
}

func captureBaseline(
	ctx context.Context,
	baseline hybridBaseline,
	prepared summarySources,
	read access.Binding,
	q query,
) (observation, error) {
	child, cancel := context.WithTimeout(ctx, attemptDuration)
	defer cancel()
	started := time.Now()
	return recordedAttempt(
		child,
		prepared,
		read,
		q,
		started,
		func(ctx context.Context) (observation, error) { return baseline.retrieve(ctx, q) },
	)
}

func recordedAttempt(
	ctx context.Context,
	prepared summarySources,
	read access.Binding,
	q query,
	started time.Time,
	execute func(context.Context) (observation, error),
) (observation, error) {
	result, err := prepared.recordObservation(ctx, read, q, evidence.Required, captureSink{}, execute)
	if err != nil {
		return result.Result, err
	}
	encoded, err := result.Receipt.Record.MarshalJSON()
	if err != nil {
		return observation{}, err
	}
	sample := result.Result
	sample.Evidence = encoded
	sample.Nanos = time.Since(started).Nanoseconds()
	if err = read.Check(ctx); err != nil {
		return observation{}, err
	}
	return sample, nil
}

func buildCaptureDenseCorpus(
	ctx context.Context,
	root string,
	f fixture,
	profile *codexHostProfile,
) (denseCorpus, error) {
	corpus, err := buildDenseCorpus(ctx, root, f)
	corpus.bindingLifetime = profileBindingLifetime(profile)
	return corpus, err
}
