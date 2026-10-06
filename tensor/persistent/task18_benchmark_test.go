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
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
	"github.com/skosovsky/ragy/tensor"
	"github.com/skosovsky/ragy/tensor/persistent"
	tensorquery "github.com/skosovsky/ragy/tensor/query"
)

// BenchmarkTask18TensorCandidateMaxSim measures durable catalog and payload reads, scoring,
// and TopK materialization. Publication, corpus creation, and query construction
// are excluded. Filesystem page caches are warm; no cache eviction is attempted.
//
//nolint:gocognit // Fixed corpus setup and candidate profiles share one explicit timed boundary.
func BenchmarkTask18TensorCandidateMaxSim(b *testing.B) {
	for _, n := range []int{100, 1000, 10000} {
		b.Run(fmt.Sprintf("N%d", n), func(b *testing.B) {
			config := task18NewConfig(b)
			config.Space.Dimension = 32
			config.MaxRecords = n
			config.MaxCatalogBytes = 64 << 20
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
					Representation:    "token-matrix",
				}
				tokens := make(tensor.Tensor, 4)
				for j := range tokens {
					tokens[j] = task18UnitVector(i*4 + j)
				}
				input[i] = persistent.Record[metadata]{
					Reference: ref,
					Value: tensor.Record[metadata]{
						ID:      id,
						Content: id,
						Meta:    metadata{Tenant: "a"},
						Space:   config.Space,
						Tensor:  tokens,
					},
				}
			}
			adapter := task18Published(b, config, input)
			read := task18Pin(b, config)
			for _, percent := range []int{10, 100} {
				b.Run(fmt.Sprintf("Candidates%dPct", percent), func(b *testing.B) {
					count := n * percent / 100
					refs := make([]source.Reference, count)
					for i := range refs {
						refs[i] = input[i*n/count].Reference
					}
					request := retrieval.Query[tensorquery.Intent]{
						Read: read,
						Intent: tensorquery.Intent{
							Embedding: tensor.Embedding{
								Space:  config.Space,
								Tokens: tensor.Tensor{task18UnitVector(0), task18UnitVector(1)},
							},
							Candidates:      refs,
							CandidateBudget: count,
						},
						Options: retrieval.RetrieveOptions{TopK: 10},
					}
					if result, err := adapter.Query(context.Background(), request); err != nil ||
						result.Documents.Len() != 10 ||
						len(result.Evidence.CandidateIDs) != count {
						b.Fatalf(
							"warmup: len=%d candidates=%d err=%v",
							result.Documents.Len(),
							len(result.Evidence.CandidateIDs),
							err,
						)
					}
					b.ReportAllocs()
					b.ResetTimer()
					for b.Loop() {
						result, err := adapter.Query(context.Background(), request)
						if err != nil || result.Documents.Len() != 10 || len(result.Evidence.CandidateIDs) != count {
							b.Fatalf(
								"query: len=%d candidates=%d err=%v",
								result.Documents.Len(),
								len(result.Evidence.CandidateIDs),
								err,
							)
						}
					}
				})
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
		Target:          "tensor",
		Store:           store,
		Schema:          schema,
		Space:           space(),
		CloneMeta:       func(meta metadata) (metadata, error) { return meta, nil },
		MaxCatalogBytes: 1 << 20,
		MaxPayloadBytes: 1 << 20,
		MaxRecords:      100,
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
		Targets: []lifecycle.Registration[[]persistent.Record[metadata]]{{Name: "tensor", Port: adapter}},
		ClonePayload: func(input []persistent.Record[metadata]) ([]persistent.Record[metadata], error) {
			out := slices.Clone(input)
			for i := range out {
				out[i].Value.Tensor = slices.Clone(out[i].Value.Tensor)
				for j := range out[i].Value.Tensor {
					out[i].Value.Tensor[j] = slices.Clone(out[i].Value.Tensor[j])
				}
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
	if _, err = executor.Stage(ctx, "n", manifest.ID, "tensor", input); err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Publish(ctx, "n", manifest.ID); err != nil {
		t.Fatal(err)
	}
	return adapter
}

func task18Pin(t *testing.B, config persistent.Config[metadata]) access.Binding {
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
