//go:build darwin || linux

package persistent_test

import (
	"context"
	"fmt"
	"path/filepath"
	"slices"
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

// BenchmarkTask18DenseExactScan measures durable catalog and payload reads, scoring,
// and TopK materialization. Publication, corpus creation, and query construction
// are excluded. Filesystem page caches are warm; no cache eviction is attempted.
func BenchmarkTask18DenseExactScan(b *testing.B) {
	for _, n := range []int{100, 1000, 10000} {
		b.Run(fmt.Sprintf("N%d", n), func(b *testing.B) {
			config := task18NewConfig(b)
			config.Space.Dimension = 32
			config.MaxRecords = n
			config.MaxCatalogBytes = 64 << 20
			config.MaxScanRecords = n
			input := make([]persistent.Record[metadata], n)
			for i := range input {
				id := fmt.Sprintf("doc-%06d", i)
				ref := source.Reference{
					Namespace:         "n",
					Source:            "policy",
					Revision:          "r1",
					Transformation:    "embedding",
					AccessFingerprint: "acl",
					Artifact:          id,
					Representation:    "dense-vector",
				}
				vector := task18UnitVector(i)
				input[i] = persistent.Record[metadata]{
					Reference: ref,
					Value: dense.Record[metadata]{
						ID:      id,
						Content: id,
						Meta:    metadata{Tenant: "a"},
						Space:   config.Space,
						Vector:  vector,
					},
				}
			}
			adapter := task18Published(b, config, input)
			request := retrieval.Query[persistent.Intent]{
				Read: task18Pin(b, config),
				Intent: persistent.Intent{
					Embedding: dense.Embedding{Space: config.Space, Vector: task18UnitVector(0)},
				},
				Options: retrieval.RetrieveOptions{TopK: 10},
			}
			if result, err := adapter.Retrieve(context.Background(), request); err != nil || result.Len() != 10 {
				b.Fatalf("warmup: len=%d err=%v", result.Len(), err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				result, err := adapter.Retrieve(context.Background(), request)
				if err != nil || result.Len() != 10 {
					b.Fatalf("query: len=%d err=%v", result.Len(), err)
				}
			}
		})
	}
}

// Exact signed basis vectors avoid normalization drift and random fixture seeds.
func task18UnitVector(index int) []float32 {
	vector := make([]float32, 32)
	vector[index%32] = 1
	if (index/32)%2 != 0 {
		vector[index%32] = -1
	}
	return vector
}

func task18NewConfig(t *testing.B) persistent.Config[metadata] {
	t.Helper()
	fields := filter.NewSchema()
	if _, err := fields.String("tenant"); err != nil {
		t.Fatal(err)
	}
	schema, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	store, err := filestore.New(filepath.Join(t.TempDir(), "manifests"), 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	return persistent.Config[metadata]{
		Root:            t.TempDir(),
		Namespace:       "n",
		Target:          "dense",
		Store:           store,
		Schema:          schema,
		Space:           space(),
		CloneMeta:       func(meta metadata) (metadata, error) { return meta, nil },
		MaxCatalogBytes: 1 << 20,
		MaxPayloadBytes: 1 << 20,
		MaxRecords:      100,
		MaxScanRecords:  100,
	}
}

func task18NewExecutor(
	t *testing.B,
	config persistent.Config[metadata],
	adapter *persistent.Adapter[metadata],
) *lifecycle.Executor[[]persistent.Record[metadata]] {
	t.Helper()
	executor, err := lifecycle.NewExecutor(lifecycle.ExecutorConfig[[]persistent.Record[metadata]]{
		Store:   config.Store,
		Targets: []lifecycle.Registration[[]persistent.Record[metadata]]{{Name: "dense", Port: adapter}},
		ClonePayload: func(input []persistent.Record[metadata]) ([]persistent.Record[metadata], error) {
			out := slices.Clone(input)
			for i := range out {
				out[i].Value.Vector = slices.Clone(out[i].Value.Vector)
			}
			return out, nil
		},
		ValidatePayload: func(manifest lifecycle.Manifest, input []persistent.Record[metadata]) error {
			if fingerprint(input) != manifest.Payload {
				return ragy.ErrInvalidArgument
			}
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return executor
}

func task18Published(
	t *testing.B,
	config persistent.Config[metadata],
	input []persistent.Record[metadata],
) *persistent.Adapter[metadata] {
	t.Helper()
	adapter, err := persistent.New(config)
	if err != nil {
		t.Fatal(err)
	}
	executor := task18NewExecutor(t, config, adapter)
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

func task18Pin(t *testing.B, config persistent.Config[metadata]) access.Binding {
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
