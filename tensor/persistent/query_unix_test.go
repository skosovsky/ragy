//go:build darwin || linux

package persistent_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
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
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
	"github.com/skosovsky/ragy/tensor"
	"github.com/skosovsky/ragy/tensor/persistent"
	tensorquery "github.com/skosovsky/ragy/tensor/query"
)

func pin(t *testing.T, config persistent.Config[metadata]) access.Binding {
	t.Helper()
	publication, err := lifecycle.CapturePublication(context.Background(), config.Store, "n", []string{"tensor"})
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

func query(read access.Binding, input []persistent.Record[metadata]) retrieval.Query[tensorquery.Intent] {
	refs := make([]source.Reference, 0, len(input))
	for _, record := range input {
		refs = append(refs, record.Reference)
	}
	return retrieval.Query[tensorquery.Intent]{Read: read, Intent: tensorquery.Intent{
		Embedding: tensor.Embedding{
			Space:  space(),
			Tokens: tensor.Tensor{{1, 0}, {0, 1}},
		},
		Candidates:      refs,
		CandidateBudget: 100,
	}, Options: retrieval.RetrieveOptions{TopK: 10}}
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
	if _, err = executor.Stage(ctx, "n", manifest.ID, "tensor", input); err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Publish(ctx, "n", manifest.ID); err != nil {
		t.Fatal(err)
	}
	return adapter
}

func TestPersistentQueryNativeOracleAndNegativeCandidates(t *testing.T) {
	// Arrange: actual persistent matrices, durable publication and independent adapter instance.
	config := newConfig(t)
	input := records()
	_ = published(t, config, input)
	adapter, err := persistent.New(config)
	if err != nil {
		t.Fatal(err)
	}
	request := query(pin(t, config), input)
	// Act.
	result, err := adapter.Query(context.Background(), request)
	// Assert: exact native scores; no clamp of >1 or negative values.
	if err != nil {
		t.Fatal(err)
	}
	docs := result.Documents.Documents()
	if len(docs) != 3 || docs[0].Score != 2 || docs[1].Score != 1 || docs[2].Score != -1 ||
		len(result.Evidence.CandidateIDs) != 3 ||
		result.Evidence.CandidateBudget != 100 {
		t.Fatal("oracle mismatch", docs, result.Evidence)
	}
	request.Intent.Candidates = request.Intent.Candidates[1:]
	result, err = adapter.Query(context.Background(), request)
	if err != nil || len(result.Evidence.CandidateIDs) != 2 || result.Documents.Documents()[0].Content != "t2" {
		t.Fatal("negative candidate fixture concealed missing t1", err)
	}
}

func TestPersistentQueryFiltersBeforePrivatePayloadRead(t *testing.T) {
	// Arrange: t3 belongs to another tenant; its payload is corrupt on disk.
	config := newConfig(t)
	input := records()
	input[2].Value.Meta.Tenant = "b"
	adapter := published(t, config, input)
	corruptPrivatePayload(t, config)
	// Act.
	result, err := adapter.Query(context.Background(), query(pin(t, config), input))
	// Assert: a private payload read would fail checksum; it must never happen.
	if err != nil || len(result.Documents.Documents()) != 2 || len(result.Evidence.CandidateIDs) != 2 {
		t.Fatal("scope failed before payload read", err)
	}
	builder, err := filter.NewBuilder(config.Schema)
	if err != nil {
		t.Fatal(err)
	}
	fields := filter.NewSchema()
	tenant, err := fields.String("tenant")
	if err != nil {
		t.Fatal(err)
	}
	contradiction, err := filter.Eq(builder, tenant, "b").Build()
	if err != nil {
		t.Fatal(err)
	}
	request := query(pin(t, config), input)
	request.Plan = &retrieval.PlannedQuery[tensorquery.Intent]{Filters: contradiction}
	result, err = adapter.Query(context.Background(), request)
	if err != nil || result.Documents.Len() != 0 {
		t.Fatal("planner replaced mandatory tenant", err)
	}
}

func TestPersistentQueryRejectsInvalidMatrixBeforeFilesystemIO(t *testing.T) {
	// Arrange: published fixture, then remove the lock to expose any target access.
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
	// Act/Assert: every malformed profile fails before the shared-lock file is created.
	for _, embedding := range malformedEmbeddings() {
		request.Intent.Embedding = embedding
		result, queryErr := adapter.Query(context.Background(), request)
		if queryErr == nil || result.Documents.Len() != 0 {
			t.Fatal("invalid query accepted")
		}
		if _, queryErr = os.Stat(locks[0]); !errors.Is(queryErr, os.ErrNotExist) {
			t.Fatal("invalid query touched backend", queryErr)
		}
	}
}

func TestPersistentQuerySeparateProcessRestart(t *testing.T) {
	// Arrange: child starts with no adapter memory and opens the parent's durable paths.
	config := newConfig(t)
	if root := os.Getenv("RAGY_TENSOR_RESTART_ROOT"); root != "" {
		store, err := filestore.New(os.Getenv("RAGY_TENSOR_RESTART_MANIFESTS"), 8<<20)
		if err != nil {
			t.Fatal(err)
		}
		config.Root, config.Store = root, store
		adapter, err := persistent.New(config)
		if err != nil {
			t.Fatal(err)
		}
		result, err := adapter.Query(context.Background(), query(pin(t, config), records()))
		if err != nil {
			t.Fatal(err)
		}
		docs := result.Documents.Documents()
		if len(docs) != 3 || docs[0].Score != 2 || docs[1].Score != 1 || docs[2].Score != -1 {
			t.Fatal("restart lost native ranking")
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
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	// Act: real process boundary; the child never writes/stages test records.
	child := exec.CommandContext(t.Context(), executable, "-test.run=^TestPersistentQuerySeparateProcessRestart$")
	child.Env = append(
		os.Environ(),
		"RAGY_TENSOR_RESTART_ROOT="+config.Root,
		"RAGY_TENSOR_RESTART_MANIFESTS="+manifests,
	)
	output, err := child.CombinedOutput()
	// Assert.
	if err != nil {
		t.Fatalf("restart query failed: %v\n%s", err, output)
	}
}

func TestPersistentTombstoneCleanupRetainedReadUnavailable(t *testing.T) {
	// Arrange: publish actual tensor files and capture the pre-deletion revision.
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
	// Act/Assert: tombstone read barrier precedes physical cleanup.
	result, err := adapter.Query(ctx, query(pin(t, config), input))
	if err != nil || result.Documents.Len() != 0 {
		t.Fatal("tombstone exposed tensors", err)
	}
	result, err = adapter.Query(ctx, query(old, input))
	if err != nil || result.Documents.Len() != 3 {
		t.Fatal("retained revision lost before cleanup", err)
	}
	staging, unknown := interruptedPaths(t, config)
	cleaner, err := lifecycle.NewCleaner(lifecycle.CleanerConfig{
		Store: config.Store, Now: time.Now, Targets: []lifecycle.CleanupRegistration{{Name: "tensor", Port: adapter}},
		Policy: lifecycle.CleanupPolicy{Deadline: time.Minute, Backoff: []time.Duration{time.Second}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = cleaner.Begin(ctx, "n", "deleted"); err != nil {
		t.Fatal(err)
	}
	job, err := cleaner.Attempt(ctx, "n", "deleted", "operation", "tensor", false)
	if err != nil || !job.Complete {
		t.Fatal("persistent cleanup failed", err)
	}
	restarted, err := persistent.New(config)
	if err != nil {
		t.Fatal(err)
	}
	result, err = restarted.Query(ctx, query(old, input))
	if !errors.Is(err, ragy.ErrUnavailable) || result.Documents.Len() != 0 {
		t.Fatal("cleaned snapshot was substituted", err)
	}
	if _, err = os.Stat(staging); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("known interrupted staging remains", err)
	}
	if _, err = os.Stat(filepath.Join(unknown, "opaque")); err != nil {
		t.Fatal("unknown inventory was deleted by guess", err)
	}

	files, err := filepath.Glob(filepath.Join(config.Root, "*", "*", "*.json"))
	if err != nil || len(files) != 0 {
		t.Fatal("physical tensors remain after cleanup", err)
	}
}

func corruptPrivatePayload(t *testing.T, config persistent.Config[metadata]) {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(config.Root, "*", "*", "*.json"))
	if err != nil {
		t.Fatal(err)
	}
	found := false
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
		// Read only fixture data to identify the private payload, independently of adapter execution.
		if err = json.Unmarshal(data, &stored); err != nil {
			t.Fatal(err)
		}
		if stored.Reference.Artifact == "t3" {
			if err = os.WriteFile(file, []byte("corrupt"), 0o600); err != nil {
				t.Fatal(err)
			}
			found = true
		}
	}
	if !found {
		t.Fatal("private fixture payload missing")
	}
}

func TestActualSparseCandidatesToPersistentTensor(t *testing.T) {
	for _, omitBest := range []bool{false, true} {
		name := "all-candidates"
		if omitBest {
			name = "missing-t1"
		}
		t.Run(name, func(t *testing.T) { sparseCase(t, omitBest) })
	}
}

func interruptedPaths(t *testing.T, config persistent.Config[metadata]) (string, string) {
	t.Helper()
	// Simulate an interrupted write left in the reserved path for this known operation.
	collections, err := filepath.Glob(filepath.Join(config.Root, "*"))
	if err != nil || len(collections) != 1 {
		t.Fatal("collection fixture", err)
	}
	hash := sha256.Sum256([]byte("operation"))
	staging := filepath.Join(collections[0], ".stage-"+hex.EncodeToString(hash[:]))
	if err = os.Mkdir(staging, 0o700); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(staging, "partial"), []byte("partial tensor"), 0o600); err != nil {
		t.Fatal(err)
	}
	unknown := filepath.Join(collections[0], ".stage-unmanaged")
	if err = os.Mkdir(unknown, 0o700); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(unknown, "opaque"), []byte("unknown owner"), 0o600); err != nil {
		t.Fatal(err)
	}
	return staging, unknown
}

func sparseCase(t *testing.T, omitBest bool) {
	t.Helper()

	// Arrange: actual BM25 corpus and actual persistent tensor storage.
	config := newConfig(t)
	input := records()
	adapter := published(t, config, input)
	read := pin(t, config)
	docs := make([]retrieval.Document[metadata], 0, len(input))
	refs := make(map[string]source.Reference, len(input))
	for _, record := range input {
		content := "needle " + record.Value.ID
		if omitBest && record.Value.ID == "t1" {
			content = "irrelevant"
		}
		docs = append(
			docs,
			retrieval.Document[metadata]{ID: record.Value.ID, Content: content, Meta: record.Value.Meta},
		)
		refs[record.Value.ID] = record.Reference
	}
	clone := func(meta metadata) (metadata, error) { return meta, nil }
	sparse, err := lexical.NewBM25Snapshot(
		context.Background(),
		config.Schema,
		lexical.Config[metadata]{SearchFields: []string{"content"}},
		read,
		docs,
		clone,
	)
	if err != nil {
		t.Fatal(err)
	}
	projected := retrieval.ProjectedBackend[tensorquery.Intent, retrieval.NoRequestMeta, struct{}, retrieval.NoRequestMeta, metadata]{
		Next: sparse,
		AdmissionProject: func(req retrieval.Query[tensorquery.Intent]) retrieval.Query[struct{}] {
			return retrieval.Query[struct{}]{Read: req.Read, Text: req.Text, Options: req.Options}
		},
		Project: func(req retrieval.Query[tensorquery.Intent]) retrieval.Query[struct{}] {
			return retrieval.Query[struct{}]{Read: req.Read, Text: req.Text, Options: req.Options}
		},
	}
	search, err := tensorquery.New(tensorquery.Config[metadata, metadata]{
		Candidates:         projected,
		Target:             adapter,
		CloneCandidateMeta: clone,
		Reference:          func(doc retrieval.Document[metadata]) (source.Reference, error) { return refs[doc.ID], nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	request := query(read, input)
	request.Intent.Candidates = nil
	request.Text = "needle"
	// Act: Search performs real candidate retrieval, projection and tensor query.
	result, err := search.Query(context.Background(), request)
	// Assert: candidate recall limitations remain visible in the final evidence.
	if err != nil {
		t.Fatal(err)
	}
	expected := 3
	best := "t1"
	if omitBest {
		expected = 2
		best = "t2"
	}
	if result.Documents.Len() != expected || len(result.Evidence.CandidateIDs) != expected ||
		result.Documents.Documents()[0].Content != best ||
		result.Evidence.CandidateBudget != 100 {
		t.Fatal("real candidate composition mismatch", result.Evidence)
	}
	if len(request.Intent.Candidates) != 0 {
		t.Fatal("composition mutated caller request")
	}
}

func malformedEmbeddings() []tensor.Embedding {
	matrices := []tensor.Tensor{
		nil,
		{},
		{{}},
		{{1}},
		{{1, 0}, {1}},
		{{float32(math.NaN()), 0}},
		{{float32(math.Inf(1)), 0}},
		{{float32(math.Inf(-1)), 0}},
		{{2, 0}},
	}
	out := make([]tensor.Embedding, 0, len(matrices)+1)
	for _, matrix := range matrices {
		out = append(out, tensor.Embedding{Space: space(), Tokens: matrix})
	}
	incompatible := space()
	incompatible.ModelRevision = "different"
	out = append(out, tensor.Embedding{Space: incompatible, Tokens: tensor.Tensor{{1, 0}}})
	return out
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
	_, err = executor.Stage(context.Background(), "n", manifest.ID, "tensor", input)
	// Assert.
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatal("foreign mapping admitted", err)
	}
}
