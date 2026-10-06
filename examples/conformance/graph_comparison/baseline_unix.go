//go:build darwin || linux

package main

import (
	"context"
	"encoding/json"
	"path/filepath"
	"slices"
	"time"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/dense"
	densefs "github.com/skosovsky/ragy/dense/persistent"
	"github.com/skosovsky/ragy/evidence"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

const (
	denseTarget          = "dense"
	foreignSource        = "foreign"
	referencePolicyEpoch = 7
)

type sourceIdentity struct{}

func (sourceIdentity) Resolve(doc retrieval.Document[baselineMetadata]) retrieval.Identity {
	return retrieval.Identity{DocumentID: doc.ID, MergeKey: doc.Meta.SourceID}
}

type denseCorpus struct {
	bindingLifetime time.Duration
	adapter         *densefs.Adapter[baselineMetadata]
	store           lifecycle.Store
	schema          filter.Schema
	tenant          filter.Field[string]
	fixture         fixture
}
type hybridBaseline struct {
	dense   *densefs.Adapter[baselineMetadata]
	lexical *lexical.BM25Snapshot[baselineMetadata]
	read    access.Binding
}

func baselineSpace() dense.Space {
	return dense.Space{
		Model:         "saved-graph-fixture",
		ModelRevision: "v1",
		Configuration: "hand-defined-normalized",
		VectorSpace:   "dot",
		Dimension:     baselineDimension,
	}
}
func buildDenseCorpus(ctx context.Context, root string, f fixture) (denseCorpus, error) {
	fields := filter.NewSchema()
	tenant, err := fields.String("tenant")
	if err != nil {
		return denseCorpus{}, err
	}
	if _, err = fields.String("source_key"); err != nil {
		return denseCorpus{}, err
	}
	schema, err := fields.Build()
	if err != nil {
		return denseCorpus{}, err
	}
	store, err := filestore.New(filepath.Join(root, "dense-ledger"), baselineManifestBytes)
	if err != nil {
		return denseCorpus{}, err
	}
	cfg := densefs.Config[baselineMetadata]{
		Root:            filepath.Join(root, denseTarget),
		Namespace:       "n",
		Target:          denseTarget,
		Store:           store,
		Schema:          schema,
		Space:           baselineSpace(),
		CloneMeta:       func(m baselineMetadata) (baselineMetadata, error) { return m, nil },
		MaxRecords:      baselineRecordCap,
		MaxScanRecords:  baselineRecordCap,
		MaxCatalogBytes: baselineFileBytes,
		MaxPayloadBytes: baselineFileBytes,
	}
	adapter, err := densefs.New(cfg)
	if err != nil {
		return denseCorpus{}, err
	}
	executor, err := lifecycle.NewExecutor(lifecycle.ExecutorConfig[[]densefs.Record[baselineMetadata]]{
		Store: store,
		Targets: []lifecycle.Registration[[]densefs.Record[baselineMetadata]]{
			{Name: denseTarget, Port: adapter},
		},
		ClonePayload: cloneDenseRecords,
		ValidatePayload: func(m lifecycle.Manifest, p []densefs.Record[baselineMetadata]) error {
			raw, e := json.Marshal(p)
			if e != nil {
				return e
			}
			if digest(raw) != m.Payload {
				return errInvalid
			}
			return nil
		},
		Now: time.Now,
	})
	if err != nil {
		return denseCorpus{}, err
	}
	for _, row := range f.Sources {
		if err = publishDenseSource(ctx, executor, row, "a"); err != nil {
			return denseCorpus{}, err
		}
	}
	foreign := sourceRow{
		ID:        foreignSource,
		Namespace: productionNamespace,
		Text:      "Billing Search LedgerDB private foreign payload",
		Dense:     []float32{1, 0, 0},
	}
	if err = publishDenseSource(ctx, executor, foreign, "b"); err != nil {
		return denseCorpus{}, err
	}
	// New adapter reads the actual durable catalogs and payloads.
	adapter, err = densefs.New(cfg)
	if err != nil {
		return denseCorpus{}, err
	}
	return denseCorpus{adapter: adapter, store: store, schema: schema, tenant: tenant, fixture: f}, nil
}
func cloneDenseRecords(input []densefs.Record[baselineMetadata]) ([]densefs.Record[baselineMetadata], error) {
	out := slices.Clone(input)
	for i := range out {
		out[i].Value.Vector = slices.Clone(out[i].Value.Vector)
	}
	return out, nil
}
func mappedSource(row sourceRow) (source.MappedText, error) {
	return source.OriginalText(
		source.Locator{
			Reference: originalReference(row.ID),
			Kind:      source.TextLocation,
			Span:      source.ByteSpan{End: len(row.Text)},
		},
		row.Text,
	)
}

func publishDenseSource(
	ctx context.Context,
	executor *lifecycle.Executor[[]densefs.Record[baselineMetadata]],
	row sourceRow,
	tenant string,
) error {
	original := originalReference(row.ID)
	indexed := original
	indexed.Transformation = "saved-dense"
	indexed.Artifact = row.ID
	indexed.Representation = "dense-vector"
	mapping, err := mappedSource(row)
	if err != nil {
		return err
	}
	payload := []densefs.Record[baselineMetadata]{
		{
			Reference:     indexed,
			SourceMapping: mapping,
			Space:         baselineSpace(),
			Value: dense.Record[baselineMetadata]{
				ID:      row.ID,
				Content: row.Text,
				Meta:    baselineMetadata{Tenant: tenant, SourceID: row.ID},
				Vector:  row.Dense,
			},
		},
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	manifest := lifecycle.Manifest{
		ID:      row.ID,
		Key:     row.ID,
		Payload: digest(encoded),
		State:   lifecycle.Planned,
		Identity: lifecycle.Identity{
			Namespace:      "n",
			Source:         row.ID,
			Revision:       "v1",
			Content:        digest([]byte(row.Text)),
			Transformation: "saved-dense",
			Access:         original.AccessFingerprint,
		},
		Targets: []lifecycle.Target{
			{
				Name:      denseTarget,
				Required:  true,
				State:     lifecycle.TargetPending,
				Artifacts: []lifecycle.Artifact{{Reference: indexed, Supports: []source.Reference{original}}},
			},
		},
	}
	if _, err = executor.Prepare(ctx, manifest); err != nil {
		return err
	}
	if _, err = executor.Stage(ctx, "n", row.ID, denseTarget, payload); err != nil {
		return err
	}
	_, err = executor.Publish(ctx, "n", row.ID)
	return err
}
func (c denseCorpus) bind(ctx context.Context, extra []access.TargetRevision) (access.Binding, error) {
	publication, err := lifecycle.CapturePublication(ctx, c.store, "n", []string{denseTarget})
	if err != nil {
		return access.Binding{}, err
	}
	targets := append(publication.Targets(), extra...)
	for _, row := range c.fixture.Sources {
		ref := originalReference(row.ID)
		targets = append(
			targets,
			access.TargetRevision{
				Target:            lexicalTarget,
				Namespace:         ref.Namespace,
				Source:            ref.Source,
				Revision:          ref.Revision,
				Transformation:    ref.Transformation,
				AccessFingerprint: ref.AccessFingerprint,
			},
		)
	}
	encoded, err := json.Marshal(targets)
	if err != nil {
		return access.Binding{}, err
	}
	publication, err = access.PinPublication(digest(encoded), targets)
	if err != nil {
		return access.Binding{}, err
	}
	builder, err := filter.NewBuilder(c.schema)
	if err != nil {
		return access.Binding{}, err
	}
	mandatory, err := filter.Eq(builder, c.tenant, "a").Build()
	if err != nil {
		return access.Binding{}, err
	}
	now := time.Now()
	lifetime := time.Minute
	if c.bindingLifetime > 0 {
		lifetime = c.bindingLifetime
	}
	return access.Scoped(
		access.ScopedConfig{
			Snapshot: access.Snapshot{
				Identity:    "graph-experiment-policy",
				PolicyEpoch: referencePolicyEpoch,
				IssuedAt:    now,
				ExpiresAt:   now.Add(lifetime),
			},
			Mandatory:   mandatory,
			Schema:      c.schema,
			Publication: publication,
			Now:         time.Now,
			Authority:   access.AuthorityFunc(func(ctx context.Context, _ access.Snapshot) error { return ctx.Err() }),
		},
	)
}
func (c denseCorpus) baseline(ctx context.Context, read access.Binding) (hybridBaseline, error) {
	docs := make([]retrieval.Document[baselineMetadata], 0, len(c.fixture.Sources)+1)
	for _, row := range c.fixture.Sources {
		mapping, err := mappedSource(row)
		if err != nil {
			return hybridBaseline{}, err
		}
		docs = append(
			docs,
			retrieval.Document[baselineMetadata]{
				ID:            row.ID,
				Content:       row.Text,
				Meta:          baselineMetadata{Tenant: "a", SourceID: row.ID},
				SourceMapping: mapping,
			},
		)
	}
	docs = append(
		docs,
		retrieval.Document[baselineMetadata]{
			ID:      foreignSource,
			Content: "private foreign payload",
			Meta:    baselineMetadata{Tenant: "b", SourceID: foreignSource},
		},
	)
	index, err := lexical.NewBM25Snapshot(
		ctx,
		c.schema,
		lexical.Config[baselineMetadata]{SearchFields: []string{"content"}, K1: baselineK1, B: baselineB},
		read,
		docs,
		func(m baselineMetadata) (baselineMetadata, error) { return m, nil },
	)
	if err != nil {
		return hybridBaseline{}, err
	}
	return hybridBaseline{dense: c.adapter, lexical: index, read: read}, nil
}
func (b hybridBaseline) retrieve(parent context.Context, q query) (observation, error) {
	ctx, cancel := context.WithTimeout(parent, attemptDuration)
	defer cancel()
	started := time.Now()
	sample := observation{
		Query:       q.ID,
		Profile:     baselineProfile,
		Scope:       b.read.Snapshot().Identity,
		Publication: b.read.Publication().Reference(),
		UsageKnown:  true, CallsKnown: true,
	}
	denseRequest := retrieval.Query[densefs.Intent]{
		Read:    b.read,
		Intent:  densefs.Intent{Embedding: dense.Embedding{Space: baselineSpace(), Vector: q.Dense}},
		Options: retrieval.RetrieveOptions{TopK: supportTopK},
	}
	lexicalRequest := retrieval.Query[struct{}]{
		Read:    b.read,
		Text:    q.Text,
		Options: retrieval.RetrieveOptions{TopK: supportTopK},
	}
	if _, err := retrieval.PrepareRead(ctx, denseRequest, b.dense); err != nil {
		return observation{}, err
	}
	if _, err := retrieval.PrepareRead(ctx, lexicalRequest, b.lexical); err != nil {
		return observation{}, err
	}
	sample.RetrievalCalls++
	denseResult, err := b.dense.Retrieve(ctx, denseRequest)
	if err != nil {
		return failedBaseline(sample, started), nil
	}
	sample.stages = append(sample.stages, observedResultStage(denseTarget, denseResult))
	sample.RetrievalCalls++
	lexicalResult, err := b.lexical.Retrieve(ctx, lexicalRequest)
	if err != nil {
		return failedBaseline(sample, started), nil
	}
	sample.stages = append(sample.stages, observedResultStage(lexicalTarget, lexicalResult))
	merger, err := retrieval.NewReciprocalRankFusion[baselineMetadata](baselineFusionK, sourceIdentity{})
	if err != nil {
		return observation{}, err
	}
	fused, err := merger.Merge(ctx, denseResult, lexicalResult)
	if err != nil {
		return failedBaseline(sample, started), nil
	}
	if err = b.read.Check(ctx); err != nil {
		return observation{}, err
	}
	sample.stages = append(sample.stages, observedResultStage("rrf", fused))
	for _, doc := range fused.Documents()[:min(supportTopK, fused.Len())] {
		for _, loc := range doc.SourceMapping.Supports() {
			if !slices.Contains(sample.Supports, loc.Reference) {
				sample.Supports = append(sample.Supports, loc.Reference)
			}
		}
	}
	sample.Nanos = time.Since(started).Nanoseconds()
	sample.Outcome = completeOutcome
	sample.Stop = "retrieved"
	return sample, b.read.Check(ctx)
}
func failedBaseline(sample observation, started time.Time) observation {
	sample.Nanos = time.Since(started).Nanoseconds()
	sample.Failed = true
	sample.Outcome = failedOutcome
	sample.Stop = executionErrorStop
	return sample
}

func observedResultStage(name string, result retrieval.ResultSet[baselineMetadata]) evidence.Stage[baselineMetadata] {
	stage := evidence.Stage[baselineMetadata]{
		Name:      name,
		Status:    evidence.StageObserved,
		Scores:    evidence.Observed,
		Sources:   evidence.Observed,
		Judgments: evidence.Unavailable,
	}
	for _, doc := range result.Documents() {
		doc.ScoreHistory = slices.Clone(doc.ScoreHistory)
		// This consumer exports original source associations, not indexed vector locators.
		// The document ID and numeric score still identify the actual adapter result.
		originalSupports := doc.SourceMapping.Supports()
		for _, loc := range doc.SourceSupports {
			if loc.Reference == originalReference(loc.Reference.Source) && !slices.Contains(originalSupports, loc) {
				originalSupports = append(originalSupports, loc)
			}
		}
		doc.SourceSupports = slices.Clone(originalSupports)
		refs := []source.Reference{}
		for _, loc := range originalSupports {
			if !slices.Contains(refs, loc.Reference) {
				refs = append(refs, loc.Reference)
			}
		}
		stage.Hits = append(stage.Hits, evidence.Hit[baselineMetadata]{Document: doc, Sources: refs})
	}
	return stage
}
