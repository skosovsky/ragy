//go:build darwin || linux

package bridge_test

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	bridge "example.com/ragy-context-bridge"
	"github.com/skosovsky/memy"

	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lexical"
	managed "github.com/skosovsky/ragy/lexical/managed"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

type indexMeta struct {
	Tag string `json:"tag"`
}

func TestForgetDrivesActualIndexLifecycleCleanup(t *testing.T) {
	// Arrange: a real managed lexical inventory in a separate lifecycle store.
	f := newFixture(t)
	store, err := filestore.New(filepath.Join(t.TempDir(), "index-ledger"), 1<<20)
	check(t, err)
	fields := filter.NewSchema()
	_, err = fields.String("tag")
	check(t, err)
	schema, err := fields.Build()
	check(t, err)
	adapter, err := managed.New(
		managed.Config[indexMeta]{
			Namespace: f.b.Scope.Key(),
			Target:    "lexical",
			Store:     store,
			Schema:    schema,
			BM25: lexical.Config[indexMeta]{
				SearchFields: []string{"content"},
			},
			CloneMeta:          func(m indexMeta) (indexMeta, error) { return m, nil },
			MaxCachedSnapshots: 4,
		},
	)
	check(t, err)
	ref := source.Reference{
		Namespace:         f.b.Scope.Key(),
		Source:            "record-0",
		Revision:          "indexed-1",
		Transformation:    "identity",
		AccessFingerprint: "host-policy",
		Artifact:          "chunk",
		Representation:    "utf8",
	}
	records := []managed.Record[indexMeta]{
		{
			Reference: ref,
			Document: retrieval.Document[indexMeta]{
				ID:      "chunk",
				Content: "retained index text",
				Meta:    indexMeta{Tag: "public"},
			},
		},
	}
	executor, err := lifecycle.NewExecutor(lifecycle.ExecutorConfig[[]managed.Record[indexMeta]]{
		Store: store,
		Targets: []lifecycle.Registration[[]managed.Record[indexMeta]]{
			{Name: "lexical", Port: adapter},
		},
		Now: f.clock.Now,
		ClonePayload: func(r []managed.Record[indexMeta]) ([]managed.Record[indexMeta], error) {
			return append([]managed.Record[indexMeta](nil), r...), nil
		},
		ValidatePayload: func(m lifecycle.Manifest, r []managed.Record[indexMeta]) error {
			if len(r) != 1 || r[0].Reference.Source != m.Identity.Source {
				return memy.ErrInvalid
			}
			return nil
		},
	})
	check(t, err)
	original := lifecycle.Manifest{
		ID:      "indexed",
		Key:     "indexed",
		Payload: "host-payload-digest",
		State:   lifecycle.Planned,
		Identity: lifecycle.Identity{
			Namespace:      ref.Namespace,
			Source:         ref.Source,
			Revision:       ref.Revision,
			Content:        "content-digest",
			Transformation: ref.Transformation,
			Access:         ref.AccessFingerprint,
		},
		Targets: []lifecycle.Target{
			{
				Name:      "lexical",
				Required:  true,
				State:     lifecycle.TargetPending,
				Artifacts: []lifecycle.Artifact{{Reference: ref, Supports: []source.Reference{ref}}},
			},
		},
	}
	_, err = executor.Prepare(t.Context(), original)
	check(t, err)
	_, err = executor.Stage(t.Context(), ref.Namespace, original.ID, "lexical", records)
	check(t, err)
	_, err = executor.Publish(t.Context(), ref.Namespace, original.ID)
	check(t, err)
	cleaner, err := lifecycle.NewCleaner(
		lifecycle.CleanerConfig{
			Store:   store,
			Now:     f.clock.Now,
			Targets: []lifecycle.CleanupRegistration{{Name: "lexical", Port: adapter}},
			Policy:  lifecycle.CleanupPolicy{Deadline: time.Minute, Backoff: []time.Duration{time.Second}},
		},
	)
	check(t, err)
	tombstone := lifecycle.Manifest{
		ID:                  "forget-index",
		Key:                 "forget-index",
		Payload:             "delete-digest",
		ExpectedPublication: "indexed",
		State:               lifecycle.Planned,
		Tombstone:           true,
		Identity:            original.Identity,
	}
	tombstone.Identity.Revision = "deleted"
	cleanup := bridge.CleanupSink{ID: "index-lifecycle", Apply: func(ctx context.Context, batch memy.PurgeBatch) error {
		if batch.Scope != f.b.Scope {
			return memy.ErrScopeViolation
		}
		_, prepareErr := executor.Prepare(ctx, tombstone)
		if prepareErr != nil {
			return prepareErr
		}
		_, publishErr := executor.Publish(ctx, ref.Namespace, tombstone.ID)
		if publishErr != nil {
			return publishErr
		}
		_, beginErr := cleaner.Begin(ctx, ref.Namespace, tombstone.ID)
		if beginErr != nil {
			return beginErr
		}
		job, attemptErr := cleaner.Attempt(ctx, ref.Namespace, tombstone.ID, original.ID, "lexical", false)
		if attemptErr != nil {
			return attemptErr
		}
		if !job.Complete {
			return memy.ErrUnavailable
		}
		return nil
	}}
	f.sink.cleanup = func(ctx context.Context, batch memy.PurgeBatch) error {
		_, cleanupErr := cleanup.Purge(ctx, batch)
		return cleanupErr
	}
	// Act: canonical Forget invokes the actual tombstone and Cleaner path.
	_, err = f.b.Run(t.Context())
	check(t, err)
	receipt, err := f.forget(t.Context())
	check(t, err)
	again, err := f.forget(t.Context())
	check(t, err)
	_, lateErr := adapter.Stage(t.Context(), lifecycle.StageRequest{Manifest: original, Target: "lexical"}, records)
	snapshot, err := store.Load(t.Context(), ref.Namespace)
	check(t, err)
	// Assert: cleanup completed, repeated deletion is safe and old upsert cannot resurrect inventory.
	if receipt.State != memy.PurgeComplete || again.State != memy.PurgeComplete || lateErr == nil ||
		len(snapshot.Cleanups) != 1 ||
		!snapshot.Cleanups[0].Complete ||
		len(f.sink.data) != 0 {
		t.Fatalf("receipt=%+v late=%v snapshot=%+v", receipt, lateErr, snapshot.Cleanups)
	}
}
