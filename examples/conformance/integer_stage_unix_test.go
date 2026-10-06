//go:build darwin || linux

package consumer_test

import (
	"encoding/json"
	"errors"
	"io/fs"
	"path/filepath"
	"strings"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
	densefs "github.com/skosovsky/ragy/dense/persistent"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
	"github.com/skosovsky/ragy/tensor"
	tensorfs "github.com/skosovsky/ragy/tensor/persistent"
)

type invalidIntegerCodec struct{ number string }

func (c invalidIntegerCodec) Encode(integerStorageMeta) (filter.RawAttributes, error) {
	return filter.RawAttributes{"tenant": json.Number(c.number)}, nil
}
func (invalidIntegerCodec) Decode(filter.RawAttributes) (integerStorageMeta, error) {
	return integerStorageMeta{}, ragy.ErrProtocol
}
func invalidStoredIntegers() []string {
	return []string{
		"1.5",
		"9223372036854775808",
		"-9223372036854775809",
		"1e999",
		"1." + strings.Repeat("0", 100) + "1",
	}
}

func assertInvalidIntegerStage[T any](
	t *testing.T,
	state persistentScopeState,
	target, root string,
	port lifecycle.StagePort[T],
	payload T,
	ref source.Reference,
) {
	t.Helper()
	// Arrange: a durable valid operation owns the intended inventory; the host codec
	// violates its integer schema without changing the source or payload identity.
	executor, plan := preparePersistentFixture(t, state, target, port, payload, []source.Reference{ref})
	// Act.
	_, err := executor.Stage(t.Context(), "n", plan.ID, target, payload)
	// Assert: invalid wire metadata fails before target lock/payload/catalog writes.
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatal("invalid integer encoded to target", err)
	}
	files := 0
	walkErr := filepath.WalkDir(root, func(_ string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !entry.IsDir() {
			files++
		}
		return nil
	})
	if walkErr != nil || files != 0 || state.probe.ioCount() != 0 {
		t.Fatal("invalid metadata wrote/read index files", walkErr, files)
	}
	publication, captureErr := lifecycle.CapturePublication(t.Context(), state.store, "n", []string{target})
	if captureErr != nil || len(publication.Targets()) != 0 {
		t.Fatal("failed stage became published", captureErr)
	}
}
func TestExternalPersistentDenseRejectsInvalidIntegerBeforeWrite(t *testing.T) {
	for _, number := range invalidStoredIntegers() {
		t.Run(number, func(t *testing.T) {
			state, _ := integerScopeState(t)
			space := dense.Space{Metric: "normalized-dot",
				Model:         "saved-fixture",
				ModelRevision: "r1",
				Configuration: "normalized",
				VectorSpace:   "dot",
				Dimension:     2,
			}
			root := filepath.Join(state.root, "dense")
			backend, err := densefs.New(
				densefs.Config[integerStorageMeta]{
					Root:            root,
					Namespace:       "n",
					Target:          "dense",
					Store:           state.store,
					Schema:          state.schema,
					Space:           space,
					Codec:           invalidIntegerCodec{number: number},
					CloneMeta:       func(value integerStorageMeta) (integerStorageMeta, error) { return value, nil },
					MaxCatalogBytes: 1 << 20,
					MaxPayloadBytes: 1 << 20,
					MaxRecords:      100,
					MaxScanRecords:  100,
				},
			)
			if err != nil {
				t.Fatal(err)
			}
			ref := persistentReference("invalid-integer", "dense-vector")
			input := []densefs.Record[integerStorageMeta]{
				{
					Reference: ref,
					Value: dense.Record[integerStorageMeta]{Space: space,
						ID:      ref.Artifact,
						Content: "policy",
						Meta:    integerStorageMeta{Tenant: integerTenantB},
						Vector:  []float32{1, 0},
					},
				},
			}
			assertInvalidIntegerStage(t, state, "dense", root, backend, input, ref)
		})
	}
}
func TestExternalPersistentTensorRejectsInvalidIntegerBeforeWrite(t *testing.T) {
	for _, number := range invalidStoredIntegers() {
		t.Run(number, func(t *testing.T) {
			state, _ := integerScopeState(t)
			space := tensor.Space{Metric: "normalized-dot",
				Model:         "saved-fixture",
				ModelRevision: "r1",
				Configuration: "normalized",
				VectorSpace:   "dot",
				Dimension:     2,
			}
			root := filepath.Join(state.root, "tensor")
			backend, err := tensorfs.New(
				tensorfs.Config[integerStorageMeta]{
					Root:            root,
					Namespace:       "n",
					Target:          "tensor",
					Store:           state.store,
					Schema:          state.schema,
					Space:           space,
					Codec:           invalidIntegerCodec{number: number},
					CloneMeta:       func(value integerStorageMeta) (integerStorageMeta, error) { return value, nil },
					MaxCatalogBytes: 1 << 20,
					MaxPayloadBytes: 1 << 20,
					MaxRecords:      100,
				},
			)
			if err != nil {
				t.Fatal(err)
			}
			ref := persistentReference("invalid-integer", "token-matrix")
			input := []tensorfs.Record[integerStorageMeta]{
				{
					Reference: ref,
					Value: tensor.Record[integerStorageMeta]{
						ID:      ref.Artifact,
						Content: "policy",
						Meta:    integerStorageMeta{Tenant: integerTenantB},
						Tensor:  tensor.Tensor{{1, 0}, {0, 1}},
						Space:   space,
					},
				},
			}
			assertInvalidIntegerStage(t, state, "tensor", root, backend, input, ref)
		})
	}
}

var _ retrieval.MetadataCodec[integerStorageMeta] = invalidIntegerCodec{}
