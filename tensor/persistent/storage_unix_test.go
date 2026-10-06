//go:build darwin || linux

package persistent_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
	"github.com/skosovsky/ragy/source"
	"github.com/skosovsky/ragy/tensor"
	"github.com/skosovsky/ragy/tensor/persistent"
)

type metadata struct {
	Tenant string `json:"tenant"`
}

func space() tensor.Space {
	return tensor.Space{
		Model:         "fixture",
		ModelRevision: "r1",
		Configuration: "normalized",
		VectorSpace:   "dot",
		Dimension:     2,
	}
}
func records() []persistent.Record[metadata] {
	var out []persistent.Record[metadata]
	for _, row := range []struct {
		id     string
		tokens tensor.Tensor
	}{
		{id: "t1", tokens: tensor.Tensor{{1, 0}, {0, 1}}},
		{id: "t2", tokens: tensor.Tensor{{1, 0}}},
		{id: "t3", tokens: tensor.Tensor{{-1, 0}}},
	} {
		ref := source.Reference{
			Namespace:         "n",
			Source:            "policy",
			Revision:          "r1",
			Transformation:    "embedding",
			AccessFingerprint: "acl",
			Artifact:          row.id,
			Representation:    "token-matrix",
		}
		out = append(
			out,
			persistent.Record[metadata]{
				Reference: ref,
				Value: tensor.Record[metadata]{
					ID:      row.id,
					Content: row.id,
					Meta:    metadata{Tenant: "a"},
					Tensor:  row.tokens,
					Space:   space(),
				},
			},
		)
	}
	return out
}
func fingerprint(records []persistent.Record[metadata]) string {
	type entry struct {
		Reference source.Reference `json:"reference"`
		Tokens    tensor.Tensor    `json:"tokens"`
		Space     tensor.Space     `json:"space"`
		Meta      metadata         `json:"meta"`
		Content   string           `json:"content"`
	}
	var input []entry
	for _, record := range records {
		input = append(
			input,
			entry{
				Reference: record.Reference,
				Tokens:    record.Value.Tensor,
				Space:     record.Value.Space,
				Meta:      record.Value.Meta,
				Content:   record.Value.Content,
			},
		)
	}
	data, _ := json.Marshal(input)
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])
}
func plan(input []persistent.Record[metadata]) lifecycle.Manifest {
	var artifacts []lifecycle.Artifact
	for _, record := range input {
		artifacts = append(
			artifacts,
			lifecycle.Artifact{Reference: record.Reference, Supports: []source.Reference{record.Reference}},
		)
	}
	return lifecycle.Manifest{
		ID:      "operation",
		Key:     "request",
		Payload: fingerprint(input),
		State:   lifecycle.Planned,
		Identity: lifecycle.Identity{
			Namespace:      "n",
			Source:         "policy",
			Revision:       "r1",
			Content:        "content",
			Transformation: "embedding",
			Access:         "acl",
		},
		Targets: []lifecycle.Target{
			{Name: "tensor", Required: true, State: lifecycle.TargetPending, Artifacts: artifacts},
		},
	}
}
func newConfig(t *testing.T) persistent.Config[metadata] {
	t.Helper()
	fields := filter.NewSchema()
	if _, err := fields.String("tenant"); err != nil {
		t.Fatal(err)
	}
	schema, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	store, err := filestore.New(filepath.Join(t.TempDir(), "manifests"), 8<<20)
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

func newExecutor(
	t *testing.T,
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

func TestPersistentTensorStageSurvivesFreshAdapterAndDetectsCorruption(t *testing.T) {
	// Arrange: actual filesystem target + durable lifecycle state, not an in-memory fake.
	config := newConfig(t)
	adapter, err := persistent.New(config)
	if err != nil {
		t.Fatal(err)
	}
	executor := newExecutor(t, config, adapter)
	input := records()
	manifest := plan(input)
	ctx := context.Background()
	if _, err = executor.Prepare(ctx, manifest); err != nil {
		t.Fatal(err)
	}
	// Act.
	if _, err = executor.Stage(ctx, "n", manifest.ID, "tensor", input); err != nil {
		t.Fatal(err)
	}
	input[0].Value.Tensor[0][0] = -1
	restarted, err := persistent.New(config)
	if err != nil {
		t.Fatal(err)
	}
	outcome, err := restarted.Inspect(ctx, lifecycle.StageRequest{Manifest: manifest, Target: "tensor"})
	// Assert: files and native matrices remain valid independently of the original adapter/input.
	if err != nil || outcome.State != lifecycle.TargetReady || outcome.Revision != "r1" {
		t.Fatal("persistent ready state lost", err)
	}
	files, err := filepath.Glob(filepath.Join(config.Root, "*", "*", "*.json"))
	if err != nil || len(files) != 4 {
		t.Fatal("expected catalog plus three payloads", err)
	}
	for _, file := range files {
		if filepath.Base(file) == "catalog.json" {
			continue
		}
		if err = os.WriteFile(file, []byte(`{"schema":"corrupt"}`), 0o600); err != nil {
			t.Fatal(err)
		}
		break
	}
	if _, err = restarted.Inspect(
		ctx,
		lifecycle.StageRequest{Manifest: manifest, Target: "tensor"},
	); !errors.Is(
		err,
		ragy.ErrProtocol,
	) {
		t.Fatal("corrupt payload reported ready", err)
	}
}

func TestPersistentTensorInvalidMatrixFailsBeforeTargetWrites(t *testing.T) {
	// Arrange: a valid prepared inventory, followed by invalid stage data.
	config := newConfig(t)
	adapter, err := persistent.New(config)
	if err != nil {
		t.Fatal(err)
	}
	input := records()
	manifest := plan(input)
	// Act/Assert: direct target contract validates every matrix before any write/lock.
	for _, embedding := range malformedEmbeddings() {
		input[0].Value.Tensor, input[0].Value.Space = embedding.Tokens, embedding.Space
		if _, err = adapter.Stage(
			context.Background(),
			lifecycle.StageRequest{Manifest: manifest, Target: "tensor"},
			input,
		); err == nil {
			t.Fatal("invalid record accepted")
		}
	}

	files, err := filepath.Glob(filepath.Join(config.Root, "*", "*"))
	if err != nil || len(files) != 0 {
		t.Fatal("invalid matrix wrote target state", err)
	}
}
