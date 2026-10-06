//go:build darwin || linux

package persistent_test

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/dense/persistent"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

func pin(t *testing.T, config persistent.Config[metadata]) access.Binding {
	t.Helper()
	publication, err := lifecycle.CapturePublication(context.Background(), config.Store, "n", []string{"dense"})
	if err != nil {
		t.Fatal(err)
	}
	fields := filter.NewSchema()
	tenant, err := fields.String("tenant")
	if err != nil {
		t.Fatal(err)
	}
	builder, err := filter.NewBuilder(config.Schema)
	if err != nil {
		t.Fatal(err)
	}
	mandatory, err := filter.Eq(builder, tenant, "a").Build()
	if err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	read, err := access.Scoped(access.ScopedConfig{
		Snapshot:  access.Snapshot{Identity: "policy", PolicyEpoch: 7, IssuedAt: now, ExpiresAt: now.Add(time.Minute)},
		Mandatory: mandatory, Schema: config.Schema, Publication: publication, Now: func() time.Time { return now },
		Authority: access.AuthorityFunc(func(context.Context, access.Snapshot) error { return nil }),
	})
	if err != nil {
		t.Fatal(err)
	}
	return read
}

func query(read access.Binding, _ []persistent.Record[metadata]) retrieval.Query[persistent.Intent] {
	return retrieval.Query[persistent.Intent]{
		Read:    read,
		Intent:  persistent.Intent{Embedding: dense.Embedding{Space: space(), Vector: []float32{1, 0}}},
		Options: retrieval.RetrieveOptions{TopK: 10},
	}
}

func published(
	t *testing.T,
	config persistent.Config[metadata],
	input []persistent.Record[metadata],
) *persistent.Adapter[metadata] {
	t.Helper()
	adapter, err := persistent.New(config)
	if err != nil {
		t.Fatal(err)
	}
	executor := newExecutor(t, config, adapter)
	manifest := plan(input)
	ctx := context.Background()
	if _, err = executor.Prepare(ctx, manifest); err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Stage(ctx, "n", manifest.ID, "dense", input); err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Publish(ctx, "n", manifest.ID); err != nil {
		t.Fatal(err)
	}
	return adapter
}

func TestPersistentDenseNativeScoresAfterRestart(t *testing.T) {
	// Arrange: actual persistent vectors and durable publication.
	config := newConfig(t)
	input := records()
	_ = published(t, config, input)
	adapter, err := persistent.New(config)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	result, err := adapter.Retrieve(context.Background(), query(pin(t, config), input))
	// Assert: native normalized dot includes zero and negative scores.
	if err != nil {
		t.Fatal(err)
	}
	docs := result.Documents()
	if len(docs) != 3 || docs[0].Score != 1 || docs[1].Score != 0 || docs[2].Score != -1 || docs[0].Content != "t1" {
		t.Fatal("native dense ranking mismatch", docs)
	}
	caps := adapter.QueryCapabilities()
	if !caps.Exact || caps.Space != space() || caps.ScoreSemantics != docs[0].ScoreSemantics {
		t.Fatal("score capability mismatch")
	}
}

func TestPersistentDensePrivatePayloadNeverLoaded(t *testing.T) {
	// Arrange: private t3 payload is corrupt; admitted t1/t2 remain valid.
	config := newConfig(t)
	input := records()
	input[2].Value.Meta.Tenant = "b"
	adapter := published(t, config, input)
	corruptPrivate(t, config)
	// Act.
	result, err := adapter.Retrieve(context.Background(), query(pin(t, config), input))
	// Assert.
	if err != nil || result.Len() != 2 {
		t.Fatal("private vector/content read", err)
	}
}

func corruptPrivate(t *testing.T, config persistent.Config[metadata]) {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(config.Root, "*", "*", "*.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, file := range files {
		if filepath.Base(file) == "catalog.json" {
			continue
		}
		data, readErr := os.ReadFile(file)
		if readErr != nil {
			t.Fatal(readErr)
		}
		var stored struct {
			Reference source.Reference `json:"reference"`
		}
		if err = json.Unmarshal(data, &stored); err != nil {
			t.Fatal(err)
		}
		if stored.Reference.Artifact != "t3" {
			continue
		}
		if err = os.WriteFile(file, []byte("corrupt"), 0o600); err != nil {
			t.Fatal(err)
		}
		return
	}
	t.Fatal("private fixture payload missing")
}

func malformedEmbeddings() []dense.Embedding {
	vectors := [][]float32{
		nil,
		{},
		{1},
		{2, 0},
		{float32(math.NaN()), 0},
		{float32(math.Inf(1)), 0},
		{float32(math.Inf(-1)), 0},
	}
	out := make([]dense.Embedding, 0, len(vectors)+1)
	for _, vector := range vectors {
		out = append(out, dense.Embedding{Space: space(), Vector: vector})
	}
	incompatible := space()
	incompatible.ModelRevision = "different"
	return append(out, dense.Embedding{Space: incompatible, Vector: []float32{1, 0}})
}

func TestPersistentDenseMalformedQueryBeforeIO(t *testing.T) {
	// Arrange.
	config := newConfig(t)
	input := records()
	adapter := published(t, config, input)
	request := query(pin(t, config), input)
	locks, err := filepath.Glob(filepath.Join(config.Root, "*", "target.lock"))
	if err != nil || len(locks) != 1 {
		t.Fatal("lock fixture", err)
	}
	if err = os.Remove(locks[0]); err != nil {
		t.Fatal(err)
	}
	// Act/Assert.
	for _, embedding := range malformedEmbeddings() {
		request.Intent.Embedding = embedding
		result, queryErr := adapter.Retrieve(context.Background(), request)
		if queryErr == nil || result.Len() != 0 {
			t.Fatal("invalid embedding accepted")
		}
		if _, queryErr = os.Stat(locks[0]); !errors.Is(queryErr, os.ErrNotExist) {
			t.Fatal("invalid embedding accessed target", queryErr)
		}
	}
}

func TestPersistentDenseScanLimitBeforePayloadCallbacks(t *testing.T) {
	// Arrange: allow only two admitted records; actual corpus contains three.
	config := newConfig(t)
	input := records()
	_ = published(t, config, input)
	config.MaxScanRecords = 2
	calls := 0
	config.CloneMeta = func(meta metadata) (metadata, error) { calls++; return meta, nil }
	adapter, err := persistent.New(config)
	if err != nil {
		t.Fatal(err)
	}
	request := query(pin(t, config), input)
	request.Options.TopK = 1
	// Act.
	result, err := adapter.Retrieve(context.Background(), request)
	// Assert: no truncation and no admitted payload callback before budget rejection.
	if !errors.Is(err, ragy.ErrInvalidArgument) || result.Len() != 0 || calls != 0 {
		t.Fatal("scan budget failed", err)
	}
}

func TestPersistentDenseSeparateProcessRestart(t *testing.T) {
	// Arrange: subprocess reopens actual files without an in-memory adapter or staging.
	config := newConfig(t)
	if root := os.Getenv("RAGY_DENSE_RESTART_ROOT"); root != "" {
		store, err := filestore.New(os.Getenv("RAGY_DENSE_RESTART_MANIFESTS"), 8<<20)
		if err != nil {
			t.Fatal(err)
		}
		config.Root, config.Store = root, store
		adapter, err := persistent.New(config)
		if err != nil {
			t.Fatal(err)
		}
		result, err := adapter.Retrieve(context.Background(), query(pin(t, config), records()))
		if err != nil {
			t.Fatal(err)
		}
		docs := result.Documents()
		if len(docs) != 3 || docs[0].Score != 1 || docs[1].Score != 0 || docs[2].Score != -1 {
			t.Fatal("restart ranking mismatch")
		}
		return
	}
	manifests := filepath.Join(t.TempDir(), "manifests")
	store, err := filestore.New(manifests, 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	config.Store = store
	_ = published(t, config, records())
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	child := exec.CommandContext(t.Context(), binary, "-test.run=^TestPersistentDenseSeparateProcessRestart$")
	child.Env = append(os.Environ(), "RAGY_DENSE_RESTART_ROOT="+config.Root, "RAGY_DENSE_RESTART_MANIFESTS="+manifests)
	output, err := child.CombinedOutput()
	// Assert.
	if err != nil {
		t.Fatalf("restart failed: %v\n%s", err, output)
	}
}

func TestPersistentDenseTombstoneThenPhysicalCleanup(t *testing.T) {
	// Arrange.
	config := newConfig(t)
	input := records()
	adapter := published(t, config, input)
	old := pin(t, config)
	executor := newExecutor(t, config, adapter)
	deleted := plan(input)
	deleted.ID, deleted.Key, deleted.ExpectedPublication = "deleted", "delete", "operation"
	deleted.Identity.Revision = "r2"
	deleted.Tombstone, deleted.Targets = true, nil
	ctx := context.Background()
	if _, err := executor.Prepare(ctx, deleted); err != nil {
		t.Fatal(err)
	}
	if _, err := executor.Publish(ctx, "n", "deleted"); err != nil {
		t.Fatal(err)
	}
	// Act/Assert: read barrier is immediate; retained snapshot still owns its revision.
	result, err := adapter.Retrieve(ctx, query(pin(t, config), input))
	if err != nil || result.Len() != 0 {
		t.Fatal("tombstone exposed dense records", err)
	}
	result, err = adapter.Retrieve(ctx, query(old, input))
	if err != nil || result.Len() != 3 {
		t.Fatal("retained snapshot lost early", err)
	}
	cleaner, err := lifecycle.NewCleaner(
		lifecycle.CleanerConfig{
			Store:   config.Store,
			Now:     time.Now,
			Targets: []lifecycle.CleanupRegistration{{Name: "dense", Port: adapter}},
			Policy:  lifecycle.CleanupPolicy{Deadline: time.Minute, Backoff: []time.Duration{time.Second}},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = cleaner.Begin(ctx, "n", "deleted"); err != nil {
		t.Fatal(err)
	}
	job, err := cleaner.Attempt(ctx, "n", "deleted", "operation", "dense", false)
	if err != nil || !job.Complete {
		t.Fatal("cleanup incomplete", err)
	}
	result, err = adapter.Retrieve(ctx, query(old, input))
	if !errors.Is(err, ragy.ErrUnavailable) || result.Len() != 0 {
		t.Fatal("cleaned revision replaced or exposed", err)
	}
}

func TestPersistentOriginalMappingSurvivesRestart(t *testing.T) {
	// Arrange: original representation differs from indexed transformation.
	config := newConfig(t)
	input := records()
	original := input[0].Reference
	original.Transformation = "original"
	original.Representation = "retained-text"
	loc := source.Locator{
		Kind:      source.TextLocation,
		Reference: original,
		Span:      source.ByteSpan{Start: 0, End: len(input[0].Value.Content)},
	}
	mapping, err := source.OriginalText(loc, input[0].Value.Content)
	if err != nil {
		t.Fatal(err)
	}
	input[0].SourceMapping = mapping
	_ = published(t, config, input)
	restarted, err := persistent.New(config)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	result, err := restarted.Retrieve(context.Background(), query(pin(t, config), input))
	// Assert: original coordinates and index document-level support both survive.
	if err != nil {
		t.Fatal(err)
	}
	doc := result.Documents()[0]
	if doc.SourceMapping.Text() != input[0].Value.Content || len(doc.SourceLocations()) != 2 ||
		doc.SourceMapping.Supports()[0] != loc {
		t.Fatal(doc)
	}
}

func TestPersistentMappingCannotClaimAnotherRevision(t *testing.T) {
	// Arrange: mapping is structurally valid but claims a foreign revision.
	config := newConfig(t)
	input := records()
	foreign := input[0].Reference
	foreign.Revision = "other"
	loc := source.Locator{
		Kind:      source.TextLocation,
		Reference: foreign,
		Span:      source.ByteSpan{Start: 0, End: len(input[0].Value.Content)},
	}
	mapping, err := source.OriginalText(loc, input[0].Value.Content)
	if err != nil {
		t.Fatal(err)
	}
	input[0].SourceMapping = mapping
	adapter, err := persistent.New(config)
	if err != nil {
		t.Fatal(err)
	}
	executor := newExecutor(t, config, adapter)
	manifest := plan(input)
	if _, err = executor.Prepare(context.Background(), manifest); err != nil {
		t.Fatal(err)
	}
	// Act.
	_, err = executor.Stage(context.Background(), "n", manifest.ID, "dense", input)
	// Assert.
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatal("foreign mapping admitted", err)
	}
}
