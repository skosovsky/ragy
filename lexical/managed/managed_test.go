//go:build darwin || linux

package managed_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"slices"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/lexical/managed"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

type metadata struct {
	Tenant   string   `json:"tenant"`
	Artifact string   `json:"artifact"`
	Tags     []string `json:"-"`
}

type fixture struct {
	store        *filestore.Store
	adapter      *managed.Adapter[metadata]
	executor     *lifecycle.Executor[[]managed.Record[metadata]]
	schema       filter.Schema
	copies       []string
	epoch        int64
	revokeOnCopy bool
	now          time.Time
}

func newFixture(t *testing.T) *fixture {
	t.Helper()
	fields := filter.NewSchema()
	for _, name := range []string{"tenant", "artifact"} {
		if _, err := fields.String(name); err != nil {
			t.Fatal(err)
		}
	}
	schema, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	store, err := filestore.New(t.TempDir(), 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	f := &fixture{store: store, schema: schema, epoch: 7, now: time.Unix(100, 0).UTC()}
	adapter, err := managed.New(managed.Config[metadata]{
		Namespace: "n", Target: "lexical", Store: store, Schema: schema,
		BM25: lexical.Config[metadata]{SearchFields: []string{"content"}},
		CloneMeta: func(meta metadata) (metadata, error) {
			f.copies = append(f.copies, meta.Artifact)
			if f.revokeOnCopy {
				f.epoch = 8
			}
			meta.Tags = slices.Clone(meta.Tags)
			return meta, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	f.adapter = adapter
	executor, err := lifecycle.NewExecutor(lifecycle.ExecutorConfig[[]managed.Record[metadata]]{
		Store: store, Now: func() time.Time { return f.now },
		Targets: []lifecycle.Registration[[]managed.Record[metadata]]{{Name: "lexical", Port: adapter}},
		ClonePayload: func(records []managed.Record[metadata]) ([]managed.Record[metadata], error) {
			captured := slices.Clone(records)
			for i := range captured {
				captured[i].Document.Meta.Tags = slices.Clone(captured[i].Document.Meta.Tags)
			}
			return captured, nil
		},
		ValidatePayload: func(manifest lifecycle.Manifest, records []managed.Record[metadata]) error {
			if manifest.Payload != payloadFingerprint(records) {
				return ragy.ErrInvalidArgument
			}
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	f.executor = executor
	return f
}

func sourcePlan(id, sourceID, revision, expected string) (lifecycle.Manifest, []managed.Record[metadata]) {
	var artifacts []lifecycle.Artifact
	var records []managed.Record[metadata]
	for _, tenant := range []string{"a", "b"} {
		reference := source.Reference{
			Namespace:         "n",
			Source:            sourceID,
			Revision:          revision,
			Transformation:    "chunk",
			AccessFingerprint: "acl",
			Artifact:          tenant,
			Representation:    "text",
		}
		artifacts = append(artifacts, lifecycle.Artifact{Reference: reference, Supports: []source.Reference{reference}})
		records = append(records, managed.Record[metadata]{Reference: reference, Document: retrieval.Document[metadata]{
			ID:      tenant,
			Content: sourceID + " searchable " + revision,
			Meta:    metadata{Tenant: tenant, Artifact: sourceID + "-" + tenant, Tags: []string{"owned"}},
		}})
	}
	return lifecycle.Manifest{
		ID:                  id,
		Key:                 id + "-key",
		Payload:             payloadFingerprint(records),
		ExpectedPublication: expected,
		State:               lifecycle.Planned,
		Identity: lifecycle.Identity{
			Namespace:      "n",
			Source:         sourceID,
			Revision:       revision,
			Content:        revision,
			Transformation: "chunk",
			Access:         "acl",
		},
		Targets: []lifecycle.Target{
			{Name: "lexical", Required: true, State: lifecycle.TargetPending, Artifacts: artifacts},
		},
	}, records
}
func (f *fixture) ingest(t *testing.T, plan lifecycle.Manifest, records []managed.Record[metadata], publish bool) {
	t.Helper()
	ctx := context.Background()
	if _, err := f.executor.Prepare(ctx, plan); err != nil {
		t.Fatal(err)
	}
	if _, err := f.executor.Stage(ctx, "n", plan.ID, "lexical", records); err != nil {
		t.Fatal(err)
	}
	if publish {
		if _, err := f.executor.Publish(ctx, "n", plan.ID); err != nil {
			t.Fatal(err)
		}
	}
}
func (f *fixture) pin(t *testing.T) access.Binding {
	t.Helper()
	publication, err := lifecycle.CapturePublication(context.Background(), f.store, "n", []string{"lexical"})
	if err != nil {
		t.Fatal(err)
	}
	fields := filter.NewSchema()
	tenant, err := fields.String("tenant")
	if err != nil {
		t.Fatal(err)
	}
	// Field identity is schema-name based; finalized schema belongs to this adapter.
	builder, err := filter.NewBuilder(f.schema)
	if err != nil {
		t.Fatal(err)
	}
	mandatory, err := filter.Eq(builder, tenant, "a").Build()
	if err != nil {
		t.Fatal(err)
	}
	read, err := access.Scoped(access.ScopedConfig{
		Snapshot: access.Snapshot{
			Identity:    "policy",
			PolicyEpoch: 7,
			IssuedAt:    f.now,
			ExpiresAt:   f.now.Add(time.Minute),
		},
		Mandatory:   mandatory,
		Schema:      f.schema,
		Publication: publication,
		Now:         func() time.Time { return f.now },
		Authority: access.AuthorityFunc(func(context.Context, access.Snapshot) error {
			if f.epoch != 7 {
				return ragy.ErrUnavailable
			}
			return nil
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	return read
}
func query(read access.Binding) retrieval.Query[struct{}] {
	return retrieval.Query[struct{}]{Read: read, Text: "searchable", Options: retrieval.RetrieveOptions{TopK: 10}}
}

func TestManuallyPinnedStagedRevisionCannotReadBeforePublication(t *testing.T) {
	// Arrange: durable staging exists, but publication has not occurred.
	f := newFixture(t)
	plan, records := sourcePlan("p1", "policy", "r1", "")
	f.ingest(t, plan, records, false)
	publication, err := access.PinPublication("unconfirmed", []access.TargetRevision{{
		Target: "lexical", Namespace: "n", Source: "policy", Revision: "r1",
		Transformation: "chunk", AccessFingerprint: "acl",
	}})
	if err != nil {
		t.Fatal(err)
	}
	read, err := access.UnrestrictedAt(publication)
	if err != nil {
		t.Fatal(err)
	}
	f.copies = nil
	// Act.
	result, err := f.adapter.Retrieve(context.Background(), query(read))
	// Assert: rejection precedes payload clone/codec execution.
	if !errors.Is(err, ragy.ErrUnavailable) || result.Len() != 0 || len(f.copies) != 0 {
		t.Fatal("unpublished staged revision became readable", err)
	}
	if _, err = f.executor.Publish(context.Background(), "n", plan.ID); err != nil {
		t.Fatal(err)
	}
	result, err = f.adapter.Retrieve(context.Background(), query(read))
	if err != nil || result.Len() != 2 {
		t.Fatal("confirmed same inventory could not be read", err)
	}
}

func TestActualBM25StagingPublicationScopeAndRetainedSnapshots(t *testing.T) {
	// Arrange: actual BM25 target and durable publication; two source-local IDs collide.
	f := newFixture(t)
	policy, records := sourcePlan("p1", "policy", "r1", "")
	f.ingest(t, policy, records, true)
	faq, faqRecords := sourcePlan("f1", "faq", "r1", "")
	f.ingest(t, faq, faqRecords, true)
	old := f.pin(t)
	replacement, newRecords := sourcePlan("p2", "policy", "r2", "p1")
	f.ingest(t, replacement, newRecords, false)
	f.copies = nil
	// Act.
	stagedRead := f.pin(t)
	before, err := f.adapter.Retrieve(context.Background(), query(stagedRead))
	// Assert: unpublished r2 excluded; mandatory metadata gate precedes private payload clone.
	if err != nil || before.Len() != 2 {
		t.Fatal("actual pinned BM25 failed", err)
	}
	checkScopeAndStaging(t, before, f.copies)
	if before.Documents()[0].ID == before.Documents()[1].ID {
		t.Fatal("source-local IDs collided")
	}
	if _, err = f.executor.Publish(context.Background(), "n", "p2"); err != nil {
		t.Fatal(err)
	}
	current, err := f.adapter.Retrieve(context.Background(), query(f.pin(t)))
	if err != nil || current.Len() != 2 {
		t.Fatal(err)
	}
	for _, doc := range current.Documents() {
		if doc.Meta.Artifact == "policy-a" && doc.Content != "policy searchable r2" {
			t.Fatal("new publication did not select r2")
		}
	}
	retained, err := f.adapter.Retrieve(context.Background(), query(old))
	if err != nil || retained.Len() != 2 {
		t.Fatal("old pin substituted or unavailable prematurely", err)
	}
	for _, doc := range retained.Documents() {
		if doc.Content == "policy searchable r2" {
			t.Fatal("old pin mixed revisions")
		}
	}
	current.Documents()[0].Meta.Tags[0] = "changed-output"
	again, err := f.adapter.Retrieve(context.Background(), query(f.pin(t)))
	if err != nil || again.Documents()[0].Meta.Tags[0] != "owned" {
		t.Fatal("metadata output aliased staged state", err)
	}
}

func TestActualBM25TombstoneExactCleanupAndMissingSnapshot(t *testing.T) {
	// Arrange: unrelated faq must survive policy tombstone/cleanup.
	f := newFixture(t)
	for _, sourceID := range []string{"policy", "faq"} {
		plan, records := sourcePlan(sourceID, sourceID, "r1", "")
		f.ingest(t, plan, records, true)
	}
	old := f.pin(t)
	tombstone, _ := sourcePlan("deleted", "policy", "r2", "policy")
	tombstone.Tombstone = true
	tombstone.Targets = nil
	if _, err := f.executor.Prepare(context.Background(), tombstone); err != nil {
		t.Fatal(err)
	}
	if _, err := f.executor.Publish(context.Background(), "n", "deleted"); err != nil {
		t.Fatal(err)
	}
	// Act/Assert: read barrier is active before physical cleanup.
	visible, err := f.adapter.Retrieve(context.Background(), query(f.pin(t)))
	if err != nil || visible.Len() != 1 || visible.Documents()[0].Meta.Artifact != "faq-a" {
		t.Fatal("tombstone failed or deleted unrelated source", err)
	}
	cleaner, err := lifecycle.NewCleaner(lifecycle.CleanerConfig{
		Store: f.store, Now: func() time.Time { return f.now },
		Targets: []lifecycle.CleanupRegistration{{Name: "lexical", Port: f.adapter}},
		Policy:  lifecycle.CleanupPolicy{Deadline: time.Minute, Backoff: []time.Duration{time.Second}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = cleaner.Begin(context.Background(), "n", "deleted"); err != nil {
		t.Fatal(err)
	}
	if _, err = cleaner.Attempt(context.Background(), "n", "deleted", "policy", "lexical", false); err != nil {
		t.Fatal(err)
	}
	unavailable, err := f.adapter.Retrieve(context.Background(), query(old))
	if !errors.Is(err, ragy.ErrUnavailable) || unavailable.Len() != 0 {
		t.Fatal("cleaned old snapshot substituted latest or leaked partial faq")
	}
	visible, err = f.adapter.Retrieve(context.Background(), query(f.pin(t)))
	if err != nil || visible.Len() != 1 {
		t.Fatal("cleanup removed faq", err)
	}
}

func TestActualBM25RevocationDuringOwnedProjectionFailsClosed(t *testing.T) {
	// Arrange.
	f := newFixture(t)
	plan, records := sourcePlan("p1", "policy", "r1", "")
	f.ingest(t, plan, records, true)
	read := f.pin(t)
	f.revokeOnCopy = true
	// Act.
	result, err := f.adapter.Retrieve(context.Background(), query(read))
	// Assert.
	if !errors.Is(err, ragy.ErrUnavailable) || result.Len() != 0 {
		t.Fatal("revoked projection delivered evidence", err)
	}
}

func payloadFingerprint(records []managed.Record[metadata]) string {
	type entry struct {
		Reference source.Reference `json:"reference"`
		ID        string           `json:"id"`
		Content   string           `json:"content"`
		Tenant    string           `json:"tenant"`
		Artifact  string           `json:"artifact"`
		Tags      []string         `json:"tags"`
	}
	var payload []entry
	for _, record := range records {
		payload = append(
			payload,
			entry{
				Reference: record.Reference,
				ID:        record.Document.ID,
				Content:   record.Document.Content,
				Tenant:    record.Document.Meta.Tenant,
				Artifact:  record.Document.Meta.Artifact,
				Tags:      record.Document.Meta.Tags,
			},
		)
	}
	data, _ := json.Marshal(payload)
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:])
}

func TestEmptyPinAndVolatileMissingSnapshotRemainExplicit(t *testing.T) {
	// Arrange/Act/Assert: empty publication must not silently become live.
	f := newFixture(t)
	result, err := f.adapter.Retrieve(context.Background(), query(f.pin(t)))
	if err != nil || result.Len() != 0 {
		t.Fatal("empty pinned inventory failed", err)
	}
	if _, err = retrieval.PrepareRead(
		context.Background(),
		query(access.Unrestricted()),
		f.adapter,
	); !access.IsUnsupportedCapability(
		err,
	) {
		t.Fatal("live profile admitted on pinned-only target", err)
	}
	plan, records := sourcePlan("p1", "policy", "r1", "")
	f.ingest(t, plan, records, true)
	read := f.pin(t)
	restarted, err := managed.New(managed.Config[metadata]{
		Namespace: "n", Target: "lexical", Store: f.store, Schema: f.schema,
		BM25:      lexical.Config[metadata]{SearchFields: []string{"content"}},
		CloneMeta: func(meta metadata) (metadata, error) { meta.Tags = slices.Clone(meta.Tags); return meta, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err = restarted.Retrieve(context.Background(), query(read))
	if !errors.Is(err, ragy.ErrUnavailable) || result.Len() != 0 {
		t.Fatal("volatile restart pretended to retain snapshot", err)
	}
}

func checkScopeAndStaging(t *testing.T, results retrieval.ResultSet[metadata], copies []string) {
	t.Helper()
	for _, doc := range results.Documents() {
		if doc.Meta.Tenant != "a" || doc.Content == "policy searchable r2" {
			t.Fatal("scope/staging isolation failed")
		}
	}
	for _, copied := range copies {
		if copied == "policy-b" || copied == "faq-b" {
			t.Fatal("private payload projection before metadata admission")
		}
	}
}
