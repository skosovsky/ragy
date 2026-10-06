//go:build darwin || linux

package consumer_test

import (
	"context"
	"errors"
	"math"
	"path/filepath"
	"slices"
	"strconv"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/dense"
	densefs "github.com/skosovsky/ragy/dense/persistent"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
	"github.com/skosovsky/ragy/tensor"
	tensorfs "github.com/skosovsky/ragy/tensor/persistent"
	tensorquery "github.com/skosovsky/ragy/tensor/query"
)

const integerTenantA int64 = 9007199254740992
const integerTenantB int64 = 9007199254740993

type integerStorageMeta struct {
	Tenant int64 `json:"tenant"`
}
type integerQueryTarget[TIntent any] interface {
	Retrieve(context.Context, retrieval.Query[TIntent]) (retrieval.ResultSet[integerStorageMeta], error)
}

func integerScopeState(t *testing.T) (persistentScopeState, filter.Field[int64]) {
	t.Helper()
	fields := filter.NewSchema()
	tenant, err := fields.Int("tenant")
	if err != nil {
		t.Fatal(err)
	}
	schema, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	store, err := filestore.New(filepath.Join(root, "state"), 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	return persistentScopeState{
		schema: schema,
		store:  store,
		root:   root,
		probe:  &payloadPort{clock: time.Unix(100, 0), epoch: 7},
	}, tenant
}
func integerTenants() []int64 {
	return []int64{integerTenantA, integerTenantB, math.MinInt64, math.MaxInt64}
}
func TestExternalPersistentDenseIntegerMetadataRoundTrip(t *testing.T) {
	// Arrange: exact BYOT integers cross actual JSON files, publication and fresh adapter.
	state, tenant := integerScopeState(t)
	space := dense.Space{
		Model:         "saved-fixture",
		ModelRevision: "r1",
		Configuration: "normalized",
		VectorSpace:   "dot",
		Dimension:     2,
	}
	config := densefs.Config[integerStorageMeta]{
		Root:            filepath.Join(state.root, "dense"),
		Namespace:       "n",
		Target:          "dense",
		Store:           state.store,
		Schema:          state.schema,
		Space:           space,
		PayloadReader:   observedFileReader{probe: state.probe},
		CloneMeta:       func(value integerStorageMeta) (integerStorageMeta, error) { return value, nil },
		MaxCatalogBytes: 1 << 20,
		MaxPayloadBytes: 1 << 20,
		MaxRecords:      100,
		MaxScanRecords:  100,
	}
	backend, err := densefs.New(config)
	if err != nil {
		t.Fatal(err)
	}
	var records []densefs.Record[integerStorageMeta]
	var refs []source.Reference
	for _, value := range integerTenants() {
		ref := persistentReference(strconv.FormatInt(value, 10), "dense-vector")
		refs = append(refs, ref)
		records = append(
			records,
			densefs.Record[integerStorageMeta]{
				Reference: ref,
				Space:     space,
				Value: dense.Record[integerStorageMeta]{
					ID:      ref.Artifact,
					Content: "policy",
					Meta:    integerStorageMeta{Tenant: value},
					Vector:  []float32{1, 0},
				},
			},
		)
	}
	publishPersistentFixture(t, state, "dense", backend, records, refs)
	backend, err = densefs.New(config)
	if err != nil {
		t.Fatal(err)
	}
	// Act/Assert: independent filter cases also observe physical payload materialization.
	checkIntegerStorage(
		t,
		state,
		"dense",
		tenant,
		backend,
		densefs.Intent{Embedding: dense.Embedding{Space: space, Vector: []float32{1, 0}}},
	)
}
func TestExternalPersistentTensorIntegerMetadataRoundTrip(t *testing.T) {
	// Arrange.
	state, tenant := integerScopeState(t)
	space := tensor.Space{
		Model:         "saved-fixture",
		ModelRevision: "r1",
		Configuration: "normalized",
		VectorSpace:   "dot",
		Dimension:     2,
	}
	config := tensorfs.Config[integerStorageMeta]{
		Root:            filepath.Join(state.root, "tensor"),
		Namespace:       "n",
		Target:          "tensor",
		Store:           state.store,
		Schema:          state.schema,
		Space:           space,
		PayloadReader:   observedFileReader{probe: state.probe},
		CloneMeta:       func(value integerStorageMeta) (integerStorageMeta, error) { return value, nil },
		MaxCatalogBytes: 1 << 20,
		MaxPayloadBytes: 1 << 20,
		MaxRecords:      100,
	}
	backend, err := tensorfs.New(config)
	if err != nil {
		t.Fatal(err)
	}
	var records []tensorfs.Record[integerStorageMeta]
	var refs []source.Reference
	for _, value := range integerTenants() {
		ref := persistentReference(strconv.FormatInt(value, 10), "token-matrix")
		refs = append(refs, ref)
		records = append(
			records,
			tensorfs.Record[integerStorageMeta]{
				Reference: ref,
				Value: tensor.Record[integerStorageMeta]{
					ID:      ref.Artifact,
					Content: "policy",
					Meta:    integerStorageMeta{Tenant: value},
					Tensor:  tensor.Tensor{{1, 0}, {0, 1}},
					Space:   space,
				},
			},
		)
	}
	publishPersistentFixture(t, state, "tensor", backend, records, refs)
	backend, err = tensorfs.New(config)
	if err != nil {
		t.Fatal(err)
	}
	// Act/Assert.
	checkIntegerStorage(
		t,
		state,
		"tensor",
		tenant,
		backend,
		tensorquery.Intent{
			Embedding:       tensor.Embedding{Space: space, Tokens: tensor.Tensor{{1, 0}, {0, 1}}},
			Candidates:      refs,
			CandidateBudget: 100,
		},
	)
}

type integerStorageCase struct {
	name      string
	condition filter.Condition
	scoped    bool
	expected  []int64
}

func integerStorageCases(t *testing.T, schema filter.Schema, tenant filter.Field[int64]) []integerStorageCase {
	t.Helper()
	condition := func(operator string, values ...int64) filter.Condition {
		builder, err := filter.NewBuilder(schema)
		if err != nil {
			t.Fatal(err)
		}
		switch operator {
		case "eq":
			builder = filter.Eq(builder, tenant, values[0])
		case "in":
			builder = filter.In(builder, tenant, values...)
		case "neq":
			builder = filter.NotEq(builder, tenant, values[0])
		}
		out, err := builder.Build()
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	return []integerStorageCase{
		{name: "all-roundtrip", expected: integerTenants()},
		{name: "eq-adjacent-a", condition: condition("eq", integerTenantA), expected: []int64{integerTenantA}},
		{name: "eq-adjacent-b", condition: condition("eq", integerTenantB), expected: []int64{integerTenantB}},
		{name: "eq-min", condition: condition("eq", math.MinInt64), expected: []int64{math.MinInt64}},
		{name: "eq-max", condition: condition("eq", math.MaxInt64), expected: []int64{math.MaxInt64}},
		{
			name:      "membership",
			condition: condition("in", integerTenantA, integerTenantB),
			expected:  []int64{integerTenantA, integerTenantB},
		},
		{
			name:      "not-equal",
			condition: condition("neq", integerTenantA),
			expected:  []int64{integerTenantB, math.MinInt64, math.MaxInt64},
		},
		{name: "mandatory-adjacent-b", scoped: true, expected: []int64{integerTenantB}},
		{name: "mandatory-conflicting-a", scoped: true, condition: condition("eq", integerTenantA), expected: nil},
	}
}

func integerStorageBinding(
	t *testing.T,
	state persistentScopeState,
	tenant filter.Field[int64],
	publication access.Publication,
	scoped bool,
) access.Binding {
	t.Helper()
	if !scoped {
		binding, err := access.UnrestrictedAt(publication)
		if err != nil {
			t.Fatal(err)
		}
		return binding
	}
	builder, err := filter.NewBuilder(state.schema)
	if err != nil {
		t.Fatal(err)
	}
	builder = filter.Eq(builder, tenant, integerTenantB)
	mandatory, err := builder.Build()
	if err != nil {
		t.Fatal(err)
	}
	binding, err := access.Scoped(
		access.ScopedConfig{
			Snapshot: access.Snapshot{
				Identity:    "integer-policy",
				PolicyEpoch: 7,
				IssuedAt:    state.probe.now(),
				ExpiresAt:   state.probe.now().Add(30 * time.Second),
			},
			Mandatory:   mandatory,
			Schema:      state.schema,
			Publication: publication,
			Authority:   access.AuthorityFunc(state.probe.validate),
			Now:         state.probe.now,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return binding
}

func checkIntegerStorage[TIntent any](
	t *testing.T,
	state persistentScopeState,
	target string,
	tenant filter.Field[int64],
	backend integerQueryTarget[TIntent],
	intent TIntent,
) {
	t.Helper()
	publication, err := lifecycle.CapturePublication(t.Context(), state.store, "n", []string{target})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range integerStorageCases(t, state.schema, tenant) {
		t.Run(tc.name, func(t *testing.T) {
			// Arrange: keep the same persisted publication; change only declared read policy.
			binding := integerStorageBinding(t, state, tenant, publication, tc.scoped)
			state.probe.mu.Lock()
			state.probe.calls = 0
			state.probe.ids = nil
			state.probe.mu.Unlock()
			// Act.
			result, readErr := backend.Retrieve(
				t.Context(),
				retrieval.Query[TIntent]{
					Read:    binding,
					Intent:  intent,
					Options: retrieval.RetrieveOptions{Filters: tc.condition, TopK: 10},
				},
			)
			// Assert: both metadata and physically loaded artifact identities remain exact.
			if readErr != nil {
				t.Fatal(readErr)
			}
			got := make([]int64, 0, result.Len())
			for _, doc := range result.Documents() {
				got = append(got, doc.Meta.Tenant)
			}
			slices.Sort(got)
			expected := slices.Clone(tc.expected)
			slices.Sort(expected)
			if !slices.Equal(got, expected) {
				t.Fatal("integer metadata changed", got, expected)
			}
			assertIntegerPayloads(t, state.probe, expected)
		})
	}
}
func assertIntegerPayloads(t *testing.T, probe *payloadPort, expected []int64) {
	t.Helper()
	ids := probe.payloadIDs()
	slices.Sort(ids)
	wanted := make([]string, 0, len(expected))
	for _, value := range expected {
		wanted = append(wanted, strconv.FormatInt(value, 10))
	}
	slices.Sort(wanted)
	if !slices.Equal(ids, wanted) || probe.ioCount() != len(expected) {
		t.Fatal("wrong integer tenant payload loaded", ids, wanted)
	}
}

func TestExternalIntegerScopeRejectsUnsupportedMandatoryNotEq(t *testing.T) {
	// Arrange: exclusion is a query operator; the mandatory profile is Eq/In/And.
	state, tenant := integerScopeState(t)
	builder, err := filter.NewBuilder(state.schema)
	if err != nil {
		t.Fatal(err)
	}
	mandatory, err := filter.NotEq(builder, tenant, integerTenantA).Build()
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	binding, err := access.Scoped(
		access.ScopedConfig{
			Snapshot: access.Snapshot{
				Identity:    "integer-policy",
				PolicyEpoch: 7,
				IssuedAt:    state.probe.now(),
				ExpiresAt:   state.probe.now().Add(30 * time.Second),
			},
			Mandatory:   mandatory,
			Schema:      state.schema,
			Publication: access.CurrentPublication(),
			Authority:   access.AuthorityFunc(state.probe.validate),
			Now:         state.probe.now,
		},
	)
	// Assert: the unsupported mandatory profile cannot create an authorization binding.
	if !errors.Is(err, ragy.ErrUnsupported) || !access.IsProtectionFailure(err) || binding.IsScoped() {
		t.Fatal("unsupported integer scope admitted", err)
	}
}
