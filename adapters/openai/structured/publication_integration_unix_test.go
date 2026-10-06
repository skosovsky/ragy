//go:build darwin || linux

package structured_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"slices"
	"sync/atomic"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/graph/managed"
	"github.com/skosovsky/ragy/graphingest/extraction"
	"github.com/skosovsky/ragy/graphingest/materialization"
	"github.com/skosovsky/ragy/graphingest/resolution"
	"github.com/skosovsky/ragy/graphingest/resolution/history"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
	"github.com/skosovsky/ragy/source"
)

type publishedMetadata struct {
	Tenant string `json:"tenant"`
	Value  int64  `json:"value"`
}
type graphPublicationFixture struct {
	store    lifecycle.Store
	target   *managed.Adapter[publishedMetadata]
	executor *lifecycle.Executor[managed.Payload[publishedMetadata]]
	schema   graph.Schema
}

func TestHTTPExtractionHistoryPublicationAndRetainedCleanup(t *testing.T) {
	// Arrange: HTTP is a protocol fixture, not a live quality experiment.
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		_, _ = io.WriteString(
			w,
			envelope(
				`{"entities":[{"id":"e1","name":"Billing","kind":"Service","attributes":{"value":9007199254740993},"snippets":[0]}],"relations":[]}`,
				"stop",
			),
		)
	}))
	t.Cleanup(server.Close)
	core, ledger, read, mapping, location := extractionFixture(t, server.URL)
	admit := exactOriginalAdmission(location)
	fixture := newGraphPublicationFixture(t)
	archiveRoot := filepath.Join(t.TempDir(), "history")
	// Act: actual extraction, resolution, immutable durable history and publication.
	extracted, err := core.Extract(context.Background(), read, ledger,
		[]extraction.Snippet[string]{{Namespace: "prod", Mapping: mapping, Access: "a"}})
	if err != nil {
		t.Fatal(err)
	}
	resolved := resolvePublishedExtraction(t, read, extracted.Extraction, admit)
	reference := archivePublishedExtraction(t, archiveRoot, read, extracted.Extraction, resolved, admit)
	plan := materializePublishedExtraction(t, fixture.schema, read, resolved, location, admit)
	publishGraphPlan(t, fixture.executor, plan)
	pinned := pinnedGraphRead(t, fixture.store, fixture.schema, read)
	seed := resolved.Entities[0].ID
	before, err := fixture.target.FindByIDs(context.Background(), graphLookup(pinned, seed))
	if err != nil {
		t.Fatal(err)
	}
	retirePublishedGraph(t, fixture, plan.Manifest)
	_, staleErr := fixture.target.FindByIDs(context.Background(), graphLookup(pinned, seed))
	after, err := fixture.target.FindByIDs(
		context.Background(),
		graphLookup(pinnedGraphRead(t, fixture.store, fixture.schema, read), seed),
	)
	if err != nil {
		t.Fatal(err)
	}
	retained := readPublishedArchive(t, archiveRoot, read, reference, admit)
	// Assert: integer, original support and canonical identity survived every boundary.
	if calls.Load() != 1 || ledger.Snapshot().Actual.Cost != 45 || len(before.Snapshot.Nodes) != 1 ||
		before.Snapshot.Nodes[0].Meta.Value != 9007199254740993 || before.Snapshot.Nodes[0].ID != seed ||
		len(
			before.Supports,
		) != 1 || !slices.Equal(before.Supports[0].References, []source.Reference{location.Reference}) ||
		!errors.Is(staleErr, ragy.ErrUnavailable) || len(after.Snapshot.Nodes) != 0 ||
		retained.Input.Entities[0].Attributes.Value.String() != "9007199254740993" || retained.Result.Entities[0].ID != seed {
		t.Fatal(calls.Load(), ledger.Snapshot(), before, staleErr, after, retained)
	}
}
func exactOriginalAdmission(expected source.Locator) history.Admission {
	return func(ctx context.Context, read access.Binding, actual source.Locator) error {
		if err := read.Check(ctx); err != nil {
			return err
		}
		if actual != expected {
			return ragy.ErrUnavailable
		}
		return nil
	}
}

func resolvePublishedExtraction(
	t *testing.T,
	read access.Binding,
	input resolution.Extraction[string, string, payload],
	admit history.Admission,
) resolution.Result[string, string, payload] {
	t.Helper()
	resolver, err := resolution.New(resolution.Config[string, string, payload]{
		OntologyIdentity: "host-ontology",
		PolicyIdentity:   "host-identity-policy",
		MaxEntities:      2,
		MaxRelations:     2,
		MaxSupports:      4,
		ValidateEntity: func(kind string, p payload) error {
			if kind != "Service" {
				return ragy.ErrInvalidGraph
			}
			_, e := p.Value.Int64()
			return e
		},
		ValidateRelation: func(string, string, string, payload) error { return ragy.ErrInvalidGraph },
		Identity: func(e resolution.Entity[string, payload]) (resolution.Decision, error) {
			return resolution.Decision{
				State:     resolution.Resolved,
				Namespace: e.Namespace,
				Key:       e.Kind + ":" + e.Name,
				Name:      e.Name,
			}, nil
		},
		RelationKey:     func(e resolution.Relation[string, payload]) (string, error) { return e.Kind, nil },
		CloneAttributes: func(p payload) (payload, error) { return p, nil },
		Equivalent:      func(a, b payload) bool { return a.Value == b.Value },
		AdmitSupport:    admit,
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := resolver.Resolve(context.Background(), read, input)
	if err != nil || len(result.Entities) != 1 {
		t.Fatal(result, err)
	}
	return result
}

func archivePublishedExtraction(
	t *testing.T,
	root string,
	read access.Binding,
	input resolution.Extraction[string, string, payload],
	result resolution.Result[string, string, payload],
	admit history.Admission,
) history.Reference {
	t.Helper()
	snapshot, err := history.Capture(context.Background(), read, history.Record[string, string, payload]{
		Metadata: history.Metadata{
			Run:                   "http-fixture",
			ExtractionFingerprint: "host-extraction",
		},
		Input:  input,
		Result: result,
	}, admit, 1<<20, 4)
	if err != nil {
		t.Fatal(err)
	}
	store, err := history.NewFileStore[string, string, payload](root, 1<<20, 4, admit)
	if err != nil {
		t.Fatal(err)
	}
	if err = store.Append(context.Background(), read, snapshot); err != nil {
		t.Fatal(err)
	}
	return snapshot.Reference()
}

func readPublishedArchive(
	t *testing.T,
	root string,
	read access.Binding,
	reference history.Reference,
	admit history.Admission,
) history.Record[string, string, payload] {
	t.Helper()
	// Fresh storage object; retaining original source is an explicit host decision.
	reopened, err := history.NewFileStore[string, string, payload](root, 1<<20, 4, admit)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := reopened.Read(context.Background(), read, reference)
	if err != nil {
		t.Fatal(err)
	}
	record, err := snapshot.Record()
	if err != nil {
		t.Fatal(err)
	}
	return record
}
func publishedSchema(t *testing.T) graph.Schema {
	t.Helper()
	fields := filter.NewSchema()
	if _, err := fields.String("tenant"); err != nil {
		t.Fatal(err)
	}
	if _, err := fields.Int("value"); err != nil {
		t.Fatal(err)
	}
	attrs, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	schema, err := graph.NewSchema(attrs, attrs)
	if err != nil {
		t.Fatal(err)
	}
	return schema
}
func newGraphPublicationFixture(t *testing.T) graphPublicationFixture {
	t.Helper()
	store, err := filestore.New(filepath.Join(t.TempDir(), "ledger"), 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	schema := publishedSchema(t)
	target, err := managed.New(
		managed.Config[publishedMetadata]{Namespace: "n", Target: "graph", Store: store, Schema: schema,
			CloneMeta: func(m publishedMetadata) (publishedMetadata, error) { return m, nil }, MaxRecords: 4},
	)
	if err != nil {
		t.Fatal(err)
	}
	executor, err := lifecycle.NewExecutor(lifecycle.ExecutorConfig[managed.Payload[publishedMetadata]]{
		Store:   store,
		Targets: []lifecycle.Registration[managed.Payload[publishedMetadata]]{{Name: "graph", Port: target}},
		ClonePayload: func(p managed.Payload[publishedMetadata]) (managed.Payload[publishedMetadata], error) {
			p.Nodes = slices.Clone(p.Nodes)
			p.Edges = slices.Clone(p.Edges)
			for i := range p.Nodes {
				p.Nodes[i].Value.Labels = slices.Clone(p.Nodes[i].Value.Labels)
			}
			return p, nil
		},
		ValidatePayload: func(_ lifecycle.Manifest, p managed.Payload[publishedMetadata]) error {
			if len(p.Nodes) != 1 || len(p.Edges) != 0 || p.Nodes[0].Value.Meta.Tenant != "a" {
				return ragy.ErrInvalidArgument
			}
			return nil
		},
		Now: time.Now,
	})
	if err != nil {
		t.Fatal(err)
	}
	return graphPublicationFixture{store: store, target: target, executor: executor, schema: schema}
}

func materializePublishedExtraction(
	t *testing.T,
	schema graph.Schema,
	read access.Binding,
	result resolution.Result[string, string, payload],
	location source.Locator,
	admit history.Admission,
) materialization.Result[publishedMetadata] {
	t.Helper()
	builder, err := materialization.New(materialization.Config[string, string, payload, publishedMetadata]{
		OntologyIdentity: "host-ontology",
		PolicyIdentity:   "host-identity-policy",
		Schema:           schema,
		MaxFacts:         4,
		MaxSupports:      4,
		CloneAttributes:  func(p payload) (payload, error) { return p, nil },
		CloneMeta:        func(m publishedMetadata) (publishedMetadata, error) { return m, nil },
		Node: func(identity resolution.Decision, kind string, p payload) (materialization.NodeValue[publishedMetadata], error) {
			value, e := p.Value.Int64()
			return materialization.NodeValue[publishedMetadata]{
				Labels:  []string{kind},
				Content: identity.Name,
				Meta:    publishedMetadata{Tenant: "a", Value: value},
			}, e
		},
		Edge: func(string, payload) (materialization.EdgeValue[publishedMetadata], error) {
			return materialization.EdgeValue[publishedMetadata]{}, ragy.ErrInvalidGraph
		},
		AdmitSupport: admit,
	})
	if err != nil {
		t.Fatal(err)
	}
	ref := location.Reference
	plan, err := builder.Build(context.Background(), read, materialization.Request{
		Identity: lifecycle.Identity{
			Namespace:      ref.Namespace,
			Source:         ref.Source,
			Revision:       ref.Revision,
			Content:        "fixture-content",
			Transformation: "graph-extraction",
			Access:         ref.AccessFingerprint,
		},
		Target:             "graph",
		ManifestID:         "extracted-source",
		Key:                "extracted-source",
		PayloadFingerprint: "fixture-payload",
	}, result)
	if err != nil {
		t.Fatal(err)
	}
	return plan
}

func publishGraphPlan(
	t *testing.T,
	executor *lifecycle.Executor[managed.Payload[publishedMetadata]],
	plan materialization.Result[publishedMetadata],
) {
	t.Helper()
	ctx := context.Background()
	if _, err := executor.Prepare(ctx, plan.Manifest); err != nil {
		t.Fatal(err)
	}
	if _, err := executor.Stage(ctx, "n", plan.Manifest.ID, "graph", plan.Payload); err != nil {
		t.Fatal(err)
	}
	if _, err := executor.Publish(ctx, "n", plan.Manifest.ID); err != nil {
		t.Fatal(err)
	}
}
func pinnedGraphRead(t *testing.T, store lifecycle.Store, schema graph.Schema, original access.Binding) access.Binding {
	t.Helper()
	ctx := context.Background()
	publication, err := lifecycle.CapturePublication(ctx, store, "n", []string{"graph"})
	if err != nil {
		t.Fatal(err)
	}
	mandatory, err := original.Prepare(
		ctx,
		schema.NodeAttributes,
		filter.Condition{},
		access.Capabilities{ScopeProfile: true},
	)
	if err != nil {
		t.Fatal(err)
	}
	read, err := access.Scoped(
		access.ScopedConfig{
			Snapshot:    original.Snapshot(),
			Mandatory:   mandatory,
			Schema:      schema.NodeAttributes,
			Publication: publication,
			Now:         time.Now,
			Authority: access.AuthorityFunc(
				func(ctx context.Context, _ access.Snapshot) error { return original.Check(ctx) },
			),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return read
}
func graphLookup(read access.Binding, seed string) managed.Request {
	return managed.Request{
		Read:      read,
		Traversal: graph.TraversalRequest{Seeds: []string{seed}, Direction: graph.DirectionOutbound, Depth: 1},
		MaxNodes:  4,
		MaxEdges:  4,
	}
}
func retirePublishedGraph(t *testing.T, fixture graphPublicationFixture, original lifecycle.Manifest) {
	t.Helper()
	ctx := context.Background()
	identity := original.Identity
	identity.Revision = "deleted"
	tombstone := lifecycle.Manifest{
		ID:                  "deleted-source",
		Identity:            identity,
		Key:                 "deleted-source",
		Payload:             "deleted-source",
		ExpectedPublication: original.ID,
		Tombstone:           true,
		State:               lifecycle.Planned,
	}
	if _, err := fixture.executor.Prepare(ctx, tombstone); err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.executor.Publish(ctx, "n", tombstone.ID); err != nil {
		t.Fatal(err)
	}
	cleaner, err := lifecycle.NewCleaner(
		lifecycle.CleanerConfig{
			Store:   fixture.store,
			Now:     time.Now,
			Targets: []lifecycle.CleanupRegistration{{Name: "graph", Port: fixture.target}},
			Policy:  lifecycle.CleanupPolicy{Deadline: time.Minute, Backoff: []time.Duration{time.Second}},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = cleaner.Begin(ctx, "n", tombstone.ID); err != nil {
		t.Fatal(err)
	}
	if _, err = cleaner.Attempt(ctx, "n", tombstone.ID, original.ID, "graph", false); err != nil {
		t.Fatal(err)
	}
}
