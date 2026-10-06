//go:build darwin || linux

package final_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/dense"
	densefs "github.com/skosovsky/ragy/dense/persistent"
	"github.com/skosovsky/ragy/embedding"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

type spaceMeta struct {
	Tenant string `json:"tenant"`
	Label  string `json:"label"`
}

// The observer reads the actual admitted file and adds no space validation.
type spacePayloadReader struct{ calls atomic.Int64 }

func (r *spacePayloadReader) ReadPayload(ctx context.Context, input lifecycle.PayloadRead) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	r.calls.Add(1)
	file, err := os.Open(input.Path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()
	data, err := io.ReadAll(io.LimitReader(file, input.MaxBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > input.MaxBytes {
		return nil, ragy.ErrProtocol
	}
	return data, ctx.Err()
}

func TestActualPersistentDenseRejectsSameDimensionForeignSpacesBeforePayload(t *testing.T) {
	// Arrange: publish a real dense payload with host-owned typed metadata, then reopen the target.
	space := embedding.Space{
		Model: "host-model", ModelRevision: "r1", Configuration: "normalize-v1",
		VectorSpace: "semantic", Dimension: 2, Metric: embedding.NormalizedDot,
	}
	backend, read, probe, expectedID := spacePublishedDense(t, space)
	for _, test := range []struct {
		name string
		edit func(*embedding.Space)
	}{
		{"model", func(s *embedding.Space) { s.Model = "another-model" }},
		{"revision", func(s *embedding.Space) { s.ModelRevision = "r2" }},
		{"configuration", func(s *embedding.Space) { s.Configuration = "normalize-v2" }},
		{"vector-space", func(s *embedding.Space) { s.VectorSpace = "another-space" }},
		{"metric", func(s *embedding.Space) { s.Metric = embedding.Cosine }},
	} {
		t.Run(test.name, func(t *testing.T) {
			// Arrange: the vector is valid and dimension-compatible in both spaces.
			foreign := space
			test.edit(&foreign)
			query := retrieval.Query[densefs.Intent]{
				Read: read, Intent: densefs.Intent{Embedding: dense.Embedding{Space: foreign, Vector: []float32{1, 0}}},
				Options: retrieval.RetrieveOptions{TopK: 1},
			}
			if err := query.Intent.Embedding.Validate(); err != nil {
				t.Fatal("foreign embedding must be valid on its own", err)
			}
			before := probe.calls.Load()
			// Act: dispatch the public query directly to the actual storage adapter.
			result, err := backend.Retrieve(t.Context(), query)
			// Assert: identity mismatch is fatal before payload materialization.
			if !errors.Is(err, ragy.ErrInvalidArgument) || !access.IsProtectionFailure(err) || !result.IsEmpty() {
				t.Fatalf("foreign space returned payload: result=%v err=%v", result.Documents(), err)
			}
			if probe.calls.Load() != before {
				t.Fatal("foreign space reached the actual payload reader")
			}
		})
	}
	t.Run("compatible-space", func(t *testing.T) {
		// Arrange: use precisely the persisted identity with the same normalized vector.
		query := retrieval.Query[densefs.Intent]{
			Read: read, Intent: densefs.Intent{Embedding: dense.Embedding{Space: space, Vector: []float32{1, 0}}},
			Options: retrieval.RetrieveOptions{TopK: 1},
		}
		before := probe.calls.Load()
		// Act.
		result, err := backend.Retrieve(t.Context(), query)
		// Assert: a functioning persisted payload and typed metadata are delivered.
		if err != nil || result.Len() != 1 {
			t.Fatal("compatible embedding rejected", err)
		}
		doc := result.Documents()[0]
		if doc.ID != expectedID || doc.Content != "persisted document" || doc.Score != 1 ||
			doc.Meta != (spaceMeta{Tenant: "a", Label: "host label"}) ||
			doc.ScoreSemantics != backend.QueryCapabilities().ScoreSemantics || probe.calls.Load() != before+1 {
			t.Fatalf("actual payload, metadata or native score mismatch: %+v", doc)
		}
	})
}

func spacePublishedDense(
	t *testing.T,
	space embedding.Space,
) (*densefs.Adapter[spaceMeta], access.Binding, *spacePayloadReader, string) {
	t.Helper()
	fields := filter.NewSchema()
	tenant, err := fields.String("tenant")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = fields.String("label"); err != nil {
		t.Fatal(err)
	}
	schema, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	store, err := filestore.New(filepath.Join(root, "manifests"), 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	probe := &spacePayloadReader{}
	config := densefs.Config[spaceMeta]{
		Root: filepath.Join(root, "dense"), Namespace: "n", Target: "dense", Store: store,
		Schema: schema, Space: space, PayloadReader: probe,
		CloneMeta:       func(meta spaceMeta) (spaceMeta, error) { return meta, nil },
		MaxCatalogBytes: 1 << 20, MaxPayloadBytes: 1 << 20, MaxRecords: 10, MaxScanRecords: 10,
	}
	backend, err := densefs.New(config)
	if err != nil {
		t.Fatal(err)
	}
	ref := source.Reference{Namespace: "n", Source: "corpus", Revision: "r1", Transformation: "embedding",
		AccessFingerprint: "acl", Artifact: "document", Representation: "dense-vector"}
	records := []densefs.Record[spaceMeta]{{Reference: ref, Value: dense.Record[spaceMeta]{
		ID: "document", Content: "persisted document", Meta: spaceMeta{Tenant: "a", Label: "host label"},
		Space: space, Vector: []float32{1, 0},
	}}}
	spacePublishRecords(t, store, backend, ref, records)
	backend, err = densefs.New(config)
	if err != nil {
		t.Fatal(err)
	}
	publication, err := lifecycle.CapturePublication(t.Context(), store, "n", []string{"dense"})
	if err != nil {
		t.Fatal(err)
	}
	builder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	mandatory, err := filter.Eq(builder, tenant, "a").Build()
	if err != nil {
		t.Fatal(err)
	}
	now := time.Unix(100, 0)
	read, err := access.Scoped(access.ScopedConfig{
		Schema:      schema,
		Mandatory:   mandatory,
		Publication: publication,
		Now:         func() time.Time { return now },
		Snapshot: access.Snapshot{
			Identity:    "host-policy",
			PolicyEpoch: 1,
			IssuedAt:    now,
			ExpiresAt:   now.Add(time.Minute),
		},
		Authority: access.AuthorityFunc(func(context.Context, access.Snapshot) error { return nil }),
	})
	if err != nil {
		t.Fatal(err)
	}
	expectedID, err := (source.Locator{Reference: ref, Kind: source.DocumentLocation}).Identity()
	if err != nil {
		t.Fatal(err)
	}
	if probe.calls.Load() != 0 {
		t.Fatal("lifecycle publishing used the query payload reader")
	}
	return backend, read, probe, expectedID
}

func spacePublishRecords(
	t *testing.T,
	store lifecycle.Store,
	backend *densefs.Adapter[spaceMeta],
	ref source.Reference,
	records []densefs.Record[spaceMeta],
) {
	t.Helper()
	fingerprint := func(value []densefs.Record[spaceMeta]) string {
		data, marshalErr := json.Marshal(value)
		if marshalErr != nil {
			t.Fatal(marshalErr)
		}
		sum := sha256.Sum256(data)
		return hex.EncodeToString(sum[:])
	}
	manifest := lifecycle.Manifest{
		ID: "operation", Key: "request", Payload: fingerprint(records), State: lifecycle.Planned,
		Identity: lifecycle.Identity{Namespace: "n", Source: "corpus", Revision: "r1", Content: "content",
			Transformation: "embedding", Access: "acl"},
		Targets: []lifecycle.Target{{Name: "dense", Required: true, State: lifecycle.TargetPending,
			Artifacts: []lifecycle.Artifact{{Reference: ref, Supports: []source.Reference{ref}}}}},
	}
	executor, err := lifecycle.NewExecutor(lifecycle.ExecutorConfig[[]densefs.Record[spaceMeta]]{
		Store: store, Targets: []lifecycle.Registration[[]densefs.Record[spaceMeta]]{{Name: "dense", Port: backend}},
		ClonePayload: func(value []densefs.Record[spaceMeta]) ([]densefs.Record[spaceMeta], error) {
			var cloned []densefs.Record[spaceMeta]
			data, cloneErr := json.Marshal(value)
			if cloneErr == nil {
				cloneErr = json.Unmarshal(data, &cloned)
			}
			return cloned, cloneErr
		},
		ValidatePayload: func(m lifecycle.Manifest, value []densefs.Record[spaceMeta]) error {
			if fingerprint(value) != m.Payload {
				return ragy.ErrInvalidArgument
			}
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Prepare(t.Context(), manifest); err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Stage(t.Context(), "n", manifest.ID, "dense", records); err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Publish(t.Context(), "n", manifest.ID); err != nil {
		t.Fatal(err)
	}
}
