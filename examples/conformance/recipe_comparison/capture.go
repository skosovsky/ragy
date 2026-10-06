package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"sync/atomic"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

const (
	captureDuration      = 2 * time.Minute
	referenceCostCap     = 100
	referenceInputCap    = 2048
	referenceOutputCap   = 512
	perCallInput         = 1024
	perCallOutput        = 256
	fusionK              = 60
	referencePolicyEpoch = 7
)

type captureCorpus struct {
	read    access.Binding
	index   *lexical.BM25Snapshot[comparisonMetadata]
	fixture fixture
}
type sampleBackend struct {
	index *lexical.BM25Snapshot[comparisonMetadata]
	calls uint64
}

func (b *sampleBackend) Schema() filter.Schema                 { return b.index.Schema() }
func (b *sampleBackend) ReadCapabilities() access.Capabilities { return b.index.ReadCapabilities() }

func (b *sampleBackend) Retrieve(
	ctx context.Context,
	request retrieval.Query[struct{}],
) (retrieval.ResultSet[comparisonMetadata], error) {
	b.calls++
	return b.index.Retrieve(ctx, request)
}

type observedModelTransport struct {
	next  http.RoundTripper
	calls atomic.Uint64
}

func (t *observedModelTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	t.calls.Add(1)
	return t.next.RoundTrip(request)
}

type modelUsage struct {
	Stage  string `json:"stage"`
	Known  bool   `json:"known"`
	Input  uint64 `json:"input_tokens"`
	Output uint64 `json:"output_tokens"`
	Cost   uint64 `json:"cost_units"`
}

func capturedUsage(stage string, usage recipe.Usage) modelUsage {
	return modelUsage{
		Stage:  stage,
		Known:  usage.Known,
		Input:  usage.Value.InputTokens,
		Output: usage.Value.OutputTokens,
		Cost:   usage.Value.Cost,
	}
}
func corpusReference(id string) source.Reference {
	return source.Reference{
		Namespace:         "experiment",
		Source:            "corpus",
		Revision:          "r1",
		Transformation:    "original",
		AccessFingerprint: "a-public",
		Artifact:          id,
		Representation:    "text",
	}
}
func corpusLocation(id string) source.Locator {
	return source.Locator{Reference: corpusReference(id), Kind: source.DocumentLocation}
}
func newCaptureCorpus(ctx context.Context) (captureCorpus, error) {
	return newCaptureCorpusLifetime(ctx, captureDuration)
}
func newCaptureCorpusLifetime(ctx context.Context, lifetime time.Duration) (captureCorpus, error) {
	var f fixture
	if err := decodeStrict(fixtureJSON, &f); err != nil {
		return captureCorpus{}, err
	}
	fields := filter.NewSchema()
	tenant, err := fields.String("tenant")
	if err != nil {
		return captureCorpus{}, err
	}
	schema, err := fields.Build()
	if err != nil {
		return captureCorpus{}, err
	}
	builder, err := filter.NewBuilder(schema)
	if err != nil {
		return captureCorpus{}, err
	}
	mandatory, err := filter.Eq(builder, tenant, "a").Build()
	if err != nil {
		return captureCorpus{}, err
	}
	sum := sha256.Sum256(fixtureJSON)
	publication, err := access.PinPublication(
		"corpus-"+hex.EncodeToString(sum[:]),
		[]access.TargetRevision{
			{
				Target:            "lexical",
				Namespace:         "experiment",
				Source:            "corpus",
				Revision:          "r1",
				Transformation:    "original",
				AccessFingerprint: "a-public",
			},
		},
	)
	if err != nil {
		return captureCorpus{}, err
	}
	now := time.Now()
	read, err := access.Scoped(
		access.ScopedConfig{
			Snapshot: access.Snapshot{
				Identity:    "synthetic-public-policy",
				PolicyEpoch: referencePolicyEpoch,
				IssuedAt:    now,
				ExpiresAt:   now.Add(lifetime),
			},
			Mandatory:   mandatory,
			Schema:      schema,
			Publication: publication,
			Now:         time.Now,
			Authority:   access.AuthorityFunc(func(ctx context.Context, _ access.Snapshot) error { return ctx.Err() }),
		},
	)
	if err != nil {
		return captureCorpus{}, err
	}
	docs := make([]retrieval.Document[comparisonMetadata], 0, len(f.Documents)+1)
	for _, doc := range f.Documents {
		docs = append(
			docs,
			retrieval.Document[comparisonMetadata]{
				ID:             doc.ID,
				Content:        doc.Text,
				Meta:           comparisonMetadata{Tenant: "a"},
				SourceSupports: []source.Locator{corpusLocation(doc.ID)},
			},
		)
	}
	// Adversarial foreign text must never reach planner assessment/capture.
	docs = append(
		docs,
		retrieval.Document[comparisonMetadata]{
			ID:      "foreign",
			Content: "возврат оплаты сброс пароля банковской картой private payload",
			Meta:    comparisonMetadata{Tenant: "b"},
		},
	)
	index, err := lexical.NewBM25Snapshot(
		ctx,
		schema,
		lexical.Config[comparisonMetadata]{SearchFields: []string{"content"}, K1: comparisonK1, B: comparisonB},
		read,
		docs,
		cloneComparisonMetadata,
	)
	if err != nil {
		return captureCorpus{}, err
	}
	return captureCorpus{read: read, index: index, fixture: f}, nil
}
func cloneComparisonMetadata(meta comparisonMetadata) (comparisonMetadata, error) { return meta, nil }

func admitComparisonSources(
	ctx context.Context,
	read access.Binding,
	doc retrieval.Document[comparisonMetadata],
) ([]source.Locator, error) {
	if err := read.Check(ctx); err != nil {
		return nil, err
	}
	if doc.Meta.Tenant != "a" {
		return nil, access.Protect(ragy.ErrUnavailable)
	}
	var f fixture
	if err := decodeStrict(fixtureJSON, &f); err != nil {
		return nil, err
	}
	for _, original := range f.Documents {
		if original.ID == doc.ID && original.Text == doc.Content {
			return []source.Locator{corpusLocation(doc.ID)}, nil
		}
	}
	return nil, access.Protect(ragy.ErrUnavailable)
}
func captureExperiment(ctx context.Context, cfg liveModelConfig, kind string) (capture, error) {
	if kind != liveExecution && kind != contractExecution {
		return capture{}, errInvalid
	}
	if cfg.apiKey == "" || cfg.counter.model == "" || cfg.counter.identity == "" || cfg.counter.program == "" {
		return capture{}, errCounter
	}
	whole, cancel := context.WithTimeout(ctx, captureDuration)
	defer cancel()
	corpus, err := newCaptureCorpus(whole)
	if err != nil {
		return capture{}, err
	}
	raw := capture{
		FixtureIdentity:   corpus.fixture.Identity,
		AdapterIdentity:   "ragy-scoped-readonly-bm25+structured-http",
		ModelIdentity:     cfg.counter.model,
		TokenizerIdentity: cfg.counter.identity,
		Configuration:     referenceConfiguration(),
		ConfigIdentity:    configurationIdentity(referenceConfiguration()),
		ExecutionKind:     kind,
	}
	for _, profile := range profiles() {
		for _, q := range corpus.fixture.Queries {
			if err = whole.Err(); err != nil {
				return raw, err
			}
			sample, sampleErr := captureSample(whole, cfg, corpus, profile, q)
			if sampleErr != nil {
				return raw, sampleErr
			}
			raw.Samples = append(raw.Samples, sample)
		}
	}
	return raw, nil
}

func captureSample(
	parent context.Context,
	cfg liveModelConfig,
	corpus captureCorpus,
	profile string,
	q query,
) (observation, error) {
	ctx, cancel := context.WithTimeout(parent, attemptDuration)
	defer cancel()
	request := retrieval.Query[struct{}]{
		Read:    corpus.read,
		Text:    q.Text,
		Options: retrieval.RetrieveOptions{TopK: topK},
	}
	started := time.Now()
	sample := observation{
		Query:       q.ID,
		Strategy:    profile,
		Scope:       corpus.read.Snapshot().Identity,
		Publication: corpus.read.Publication().Reference(),
	}
	backend := &sampleBackend{index: corpus.index}
	if profile == baselineProfile {
		return captureBaseline(ctx, backend, request, sample, started)
	}
	transport := &observedModelTransport{next: http.DefaultTransport}
	client := http.Client{}
	if cfg.client != nil {
		client = *cfg.client
		if client.Transport != nil {
			transport.next = client.Transport
		}
	}
	client.Transport = transport
	cfg.client = &client
	ports, err := newLiveModelPorts(ctx, cfg, recipe.Strategy(profile))
	if err != nil {
		return sample, err
	}
	config := comparisonRecipeConfig(backend, ports, recipe.Strategy(profile), &sample)
	instance, err := recipe.New(config)
	if err != nil {
		return sample, err
	}
	result, runErr := instance.RunOwnObserved(ctx, request)
	sample.RetrievalCalls, sample.ModelCalls = backend.calls, transport.calls.Load()
	sample.InputTokens, sample.OutputTokens, sample.Cost = result.Budget.Actual.InputTokens, result.Budget.Actual.OutputTokens, result.Budget.Actual.Cost
	sample.UsageKnown = runErr == nil && result.Budget.UnknownUsage == 0 && !result.Budget.UnknownCost
	sample.Outcome, sample.Stop = string(result.Outcome), string(result.Stop)
	sample.Nanos = time.Since(started).Nanoseconds()
	if runErr != nil {
		sample.Failed = true
		sample.Outcome = failedOutcome
		sample.Stop = executionErrorStop
		return sample, nil
	}
	if err = corpus.read.Check(ctx); err != nil {
		return observation{}, err
	}
	for _, selected := range result.Selected {
		sample.IDs = append(sample.IDs, selected.Document.ID)
		for _, loc := range selected.Document.SourceLocations() {
			sample.SourceRefs = append(sample.SourceRefs, loc.Reference)
		}
	}
	return sample, nil
}

func captureBaseline(
	ctx context.Context,
	backend *sampleBackend,
	request retrieval.Query[struct{}],
	sample observation,
	started time.Time,
) (observation, error) {
	result, err := backend.Retrieve(ctx, request)
	sample.RetrievalCalls = backend.calls
	sample.UsageKnown = true
	sample.Nanos = time.Since(started).Nanoseconds()
	if err == nil {
		if gateErr := request.Read.Check(ctx); gateErr != nil {
			return observation{}, gateErr
		}
		sample.Outcome, sample.Stop = completeOutcome, "retrieved"
		for _, doc := range result.Documents() {
			sample.IDs = append(sample.IDs, doc.ID)
			for _, loc := range doc.SourceLocations() {
				sample.SourceRefs = append(sample.SourceRefs, loc.Reference)
			}
		}
	} else {
		sample.Failed = true
		sample.Outcome = failedOutcome
		sample.Stop = executionErrorStop
	}
	return sample, nil
}

func comparisonRecipeConfig(
	backend *sampleBackend,
	ports liveModelPorts,
	strategy recipe.Strategy,
	sample *observation,
) recipe.Config[struct{}, retrieval.NoRequestMeta, comparisonMetadata] {
	queryLimit := maximumAssessmentQueries
	if strategy == recipe.SingleRewrite {
		queryLimit = 1
	}
	if strategy == recipe.MultiQuery {
		queryLimit = normalModelCalls
	}
	retrievalLimit := uint64(maximumRecipeCalls)
	if strategy == recipe.SingleRewrite {
		retrievalLimit = normalModelCalls
	}
	return recipe.Config[struct{}, retrieval.NoRequestMeta, comparisonMetadata]{
		BackendModelFree: true,
		Strategy:         strategy,
		Revision:         "task12-text-reference",
		Backend:          backend,
		Identity:         retrieval.DocumentIDResolver[comparisonMetadata]{},
		Admission: func(ctx context.Context, req retrieval.Query[struct{}]) (retrieval.ReadCoverage, error) {
			_, err := retrieval.PrepareRead(ctx, req, backend)
			return retrieval.CompleteReadCoverage(), err
		},
		Planner: func(ctx context.Context, req retrieval.Query[struct{}], limits recipe.ModelLimits) (recipe.Planning, error) {
			result, err := ports.planner.Plan(ctx, req, limits)
			sample.ModelUsage = append(sample.ModelUsage, capturedUsage("plan", result.Usage))
			return result, err
		},
		Assessor: func(ctx context.Context, input recipe.AssessmentInput[struct{}, retrieval.NoRequestMeta, comparisonMetadata], limits recipe.ModelLimits) (recipe.Assessment, error) {
			result, err := ports.assessor.Assess(ctx, input, limits)
			sample.ModelUsage = append(sample.ModelUsage, capturedUsage("assess", result.Usage))
			return result, err
		},
		Pricing: func(_ context.Context, op recipe.Operation) (recipe.Quote, error) {
			if op == recipe.Retrieve {
				return recipe.Quote{CostKnown: true}, nil
			}
			return recipe.Quote{
				Usage:     budget.Usage{InputTokens: perCallInput, OutputTokens: perCallOutput, Cost: fixtureCallCost},
				CostKnown: true,
			}, nil
		},
		CloneIntent:      func(input struct{}) (struct{}, error) { return input, nil },
		CloneRequestMeta: func(input retrieval.NoRequestMeta) (retrieval.NoRequestMeta, error) { return input, nil },
		CloneMeta:        cloneComparisonMetadata,
		Supports:         admitComparisonSources,
		Limits: budget.Limits{
			RetrievalCalls: retrievalLimit,
			ModelCalls:     normalModelCalls,
			Usage: budget.Usage{
				InputTokens:  referenceInputCap,
				OutputTokens: referenceOutputCap,
				Cost:         referenceCostCap,
			},
		},
		RequireKnownCost: true,
		Duration:         attemptDuration,
		Now:              time.Now,
		MaxQueries:       queryLimit,
		MaxDocuments:     maximumAssessmentDocuments,
		FusionK:          fusionK,
	}
}
