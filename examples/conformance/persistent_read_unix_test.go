//go:build darwin || linux

package consumer_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/contracttest"
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

// This module cannot import ragy's internal packages. Observation wraps real file
// materialization; query projection supplies no extra enforcement that masks a leaf defect.
type observedFileReader struct{ probe *payloadPort }

func (r observedFileReader) ReadPayload(ctx context.Context, input lifecycle.PayloadRead) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	r.probe.mu.Lock()
	r.probe.calls++
	r.probe.ids = append(r.probe.ids, input.Reference.Artifact)
	r.probe.deadline, r.probe.hasDeadline = ctx.Deadline()
	r.probe.mu.Unlock()
	file, err := os.Open(input.Path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()
	data, err := io.ReadAll(io.LimitReader(file, input.MaxBytes+1))
	if err != nil {
		return nil, err
	}
	r.probe.mu.Lock()
	if r.probe.revokeDuring {
		r.probe.epoch = 8
	}
	r.probe.mu.Unlock()
	if int64(len(data)) > input.MaxBytes {
		return nil, ragy.ErrProtocol
	}
	return data, ctx.Err()
}

type persistentQueryTarget[TIntent any] interface {
	Schema() filter.Schema
	ReadCapabilities() access.Capabilities
	Retrieve(context.Context, retrieval.Query[TIntent]) (retrieval.ResultSet[sourceMeta], error)
}
type externalPersistentBackend[TIntent any] struct {
	target persistentQueryTarget[TIntent]
}

func (b externalPersistentBackend[TIntent]) Schema() filter.Schema { return b.target.Schema() }
func (b externalPersistentBackend[TIntent]) ReadCapabilities() access.Capabilities {
	return b.target.ReadCapabilities()
}

func (b externalPersistentBackend[TIntent]) Retrieve(
	ctx context.Context,
	request retrieval.Request[TIntent, requestMeta],
) (retrieval.ResultSet[sourceMeta], error) {
	return b.target.Retrieve(
		ctx,
		retrieval.Query[TIntent]{
			Read:    request.Read,
			Text:    request.Text,
			Intent:  request.Intent,
			Options: request.Options,
			Plan:    request.Plan,
		},
	)
}

type persistentScopeState struct {
	schema              filter.Schema
	mandatory, conflict filter.Condition
	probe               *payloadPort
	store               lifecycle.Store
	root                string
}

func newPersistentScope(t *testing.T) persistentScopeState {
	t.Helper()
	fields := filter.NewSchema()
	org, err := fields.String("organization")
	if err != nil {
		t.Fatal(err)
	}
	visibility, err := fields.String("access")
	if err != nil {
		t.Fatal(err)
	}
	schema, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	builder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	mandatory, err := filter.In(filter.Eq(builder, org, "a"), visibility, "public").Build()
	if err != nil {
		t.Fatal(err)
	}
	builder, err = filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	conflict, err := filter.Eq(builder, org, "b").Build()
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	store, err := filestore.New(filepath.Join(root, "state"), 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	return persistentScopeState{
		schema:    schema,
		mandatory: mandatory,
		conflict:  conflict,
		probe:     &payloadPort{clock: time.Unix(100, 0), epoch: 7},
		store:     store,
		root:      root,
	}
}
func persistentReference(id, representation string) source.Reference {
	return source.Reference{
		Namespace:         "n",
		Source:            "corpus",
		Revision:          "r1",
		Transformation:    "saved-embedding",
		AccessFingerprint: "acl",
		Artifact:          id,
		Representation:    representation,
	}
}
func scopeMetadata(id string) sourceMeta {
	switch id {
	case privateID:
		return sourceMeta{Organization: "a", Access: "private"}
	case foreignID:
		return sourceMeta{Organization: "b", Access: "public"}
	default:
		return sourceMeta{Organization: "a", Access: "public"}
	}
}
func persistentFingerprint[T any](value T) (string, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

func preparePersistentFixture[T any](
	t *testing.T,
	state persistentScopeState,
	target string,
	port lifecycle.StagePort[T],
	payload T,
	refs []source.Reference,
) (*lifecycle.Executor[T], lifecycle.Manifest) {
	t.Helper()
	fingerprint, err := persistentFingerprint(payload)
	if err != nil {
		t.Fatal(err)
	}
	artifacts := make([]lifecycle.Artifact, 0, len(refs))
	for _, ref := range refs {
		artifacts = append(artifacts, lifecycle.Artifact{Reference: ref, Supports: []source.Reference{ref}})
	}
	plan := lifecycle.Manifest{
		ID:      "operation",
		Key:     "request",
		Payload: fingerprint,
		State:   lifecycle.Planned,
		Identity: lifecycle.Identity{
			Namespace:      "n",
			Source:         "corpus",
			Revision:       "r1",
			Content:        fingerprint,
			Transformation: "saved-embedding",
			Access:         "acl",
		},
		Targets: []lifecycle.Target{
			{Name: target, Required: true, State: lifecycle.TargetPending, Artifacts: artifacts},
		},
	}
	executor, err := lifecycle.NewExecutor(
		lifecycle.ExecutorConfig[T]{
			Store:   state.store,
			Targets: []lifecycle.Registration[T]{{Name: target, Port: port}},
			ClonePayload: func(value T) (T, error) {
				var copyValue T
				data, e := json.Marshal(value)
				if e != nil {
					return copyValue, e
				}
				e = json.Unmarshal(data, &copyValue)
				return copyValue, e
			},
			ValidatePayload: func(manifest lifecycle.Manifest, value T) error {
				got, e := persistentFingerprint(value)
				if e != nil {
					return e
				}
				if got != manifest.Payload {
					return ragy.ErrInvalidArgument
				}
				return nil
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Prepare(t.Context(), plan); err != nil {
		t.Fatal(err)
	}
	return executor, plan
}

func publishPersistentFixture[T any](
	t *testing.T,
	state persistentScopeState,
	target string,
	port lifecycle.StagePort[T],
	payload T,
	refs []source.Reference,
) {
	t.Helper()
	executor, plan := preparePersistentFixture(t, state, target, port, payload, refs)
	var err error
	if _, err = executor.Stage(t.Context(), "n", plan.ID, target, payload); err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Publish(t.Context(), "n", plan.ID); err != nil {
		t.Fatal(err)
	}
	if state.probe.ioCount() != 0 {
		t.Fatal("query reader used during lifecycle stage/verification")
	}
}

func persistentReadFixture[TIntent any](
	t *testing.T,
	state persistentScopeState,
	target string,
	backend persistentQueryTarget[TIntent],
	intent TIntent,
	ref source.Reference,
) contracttest.ScopedReadFixture[TIntent, requestMeta, sourceMeta] {
	t.Helper()
	publication, err := lifecycle.CapturePublication(t.Context(), state.store, "n", []string{target})
	if err != nil {
		t.Fatal(err)
	}
	binding, err := access.Scoped(
		access.ScopedConfig{
			Snapshot: access.Snapshot{
				Identity:    "host-policy-7",
				PolicyEpoch: 7,
				IssuedAt:    state.probe.now(),
				ExpiresAt:   state.probe.now().Add(30 * time.Second),
			},
			Mandatory:   state.mandatory,
			Schema:      state.schema,
			Publication: publication,
			Authority:   access.AuthorityFunc(state.probe.validate),
			Now:         state.probe.now,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	id, err := (source.Locator{Reference: ref, Kind: source.DocumentLocation}).Identity()
	if err != nil {
		t.Fatal(err)
	}
	return contracttest.ScopedReadFixture[TIntent, requestMeta, sourceMeta]{
		Backend: externalPersistentBackend[TIntent]{target: backend},
		Request: retrieval.Request[TIntent, requestMeta]{
			Read:    binding,
			Text:    "policy",
			Intent:  intent,
			Meta:    requestMeta{Correlation: "qa"},
			Options: retrieval.RetrieveOptions{TopK: 10},
		},
		Conflict:            state.conflict,
		Unsupported:         foreignPredicate(t),
		ExpectedIDs:         []string{id},
		ForbiddenPayloadIDs: []string{privateID, foreignID},
		IOCount:             state.probe.ioCount,
		PayloadIDs:          state.probe.payloadIDs,
		Revoke:              state.probe.revoke,
		Expire:              state.probe.expire,
		RevokeDuringIO:      state.probe.revokeOnRead,
		ObservedDeadline:    state.probe.observedDeadline,
	}
}
func newPersistentDenseFixture(t *testing.T) contracttest.ScopedReadFixture[densefs.Intent, requestMeta, sourceMeta] {
	t.Helper()
	state := newPersistentScope(t)
	space := dense.Space{Metric: "normalized-dot",
		Model:         "saved-fixture",
		ModelRevision: "r1",
		Configuration: "normalized",
		VectorSpace:   "dot",
		Dimension:     2,
	}
	config := densefs.Config[sourceMeta]{
		Root:            filepath.Join(state.root, "dense"),
		Namespace:       "n",
		Target:          "dense",
		Store:           state.store,
		Schema:          state.schema,
		Space:           space,
		PayloadReader:   observedFileReader{probe: state.probe},
		CloneMeta:       func(value sourceMeta) (sourceMeta, error) { return value, nil },
		MaxCatalogBytes: 1 << 20,
		MaxPayloadBytes: 1 << 20,
		MaxRecords:      100,
		MaxScanRecords:  100,
	}
	backend, err := densefs.New(config)
	if err != nil {
		t.Fatal(err)
	}
	var records []densefs.Record[sourceMeta]
	var refs []source.Reference
	for _, id := range []string{publicID, privateID, foreignID} {
		ref := persistentReference(id, "dense-vector")
		refs = append(refs, ref)
		records = append(
			records,
			densefs.Record[sourceMeta]{
				Reference: ref,
				Value: dense.Record[sourceMeta]{Space: space,
					ID:      id,
					Content: "policy",
					Meta:    scopeMetadata(id),
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
	return persistentReadFixture(
		t,
		state,
		"dense",
		backend,
		densefs.Intent{Embedding: dense.Embedding{Space: space, Vector: []float32{1, 0}}},
		refs[0],
	)
}

func newPersistentTensorFixture(
	t *testing.T,
) contracttest.ScopedReadFixture[tensorquery.Intent, requestMeta, sourceMeta] {
	t.Helper()
	state := newPersistentScope(t)
	space := tensor.Space{Metric: "normalized-dot",
		Model:         "saved-fixture",
		ModelRevision: "r1",
		Configuration: "normalized",
		VectorSpace:   "dot",
		Dimension:     2,
	}
	config := tensorfs.Config[sourceMeta]{
		Root:            filepath.Join(state.root, "tensor"),
		Namespace:       "n",
		Target:          "tensor",
		Store:           state.store,
		Schema:          state.schema,
		Space:           space,
		PayloadReader:   observedFileReader{probe: state.probe},
		CloneMeta:       func(value sourceMeta) (sourceMeta, error) { return value, nil },
		MaxCatalogBytes: 1 << 20,
		MaxPayloadBytes: 1 << 20,
		MaxRecords:      100,
	}
	backend, err := tensorfs.New(config)
	if err != nil {
		t.Fatal(err)
	}
	var records []tensorfs.Record[sourceMeta]
	var refs []source.Reference
	for _, id := range []string{publicID, privateID, foreignID} {
		ref := persistentReference(id, "token-matrix")
		refs = append(refs, ref)
		records = append(
			records,
			tensorfs.Record[sourceMeta]{
				Reference: ref,
				Value: tensor.Record[sourceMeta]{
					ID:      id,
					Content: "policy",
					Meta:    scopeMetadata(id),
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
	return persistentReadFixture(
		t,
		state,
		"tensor",
		backend,
		tensorquery.Intent{
			Embedding:       tensor.Embedding{Space: space, Tokens: tensor.Tensor{{1, 0}, {0, 1}}},
			Candidates:      refs,
			CandidateBudget: 100,
		},
		refs[0],
	)
}
func TestExternalPersistentDenseScopedConformance(t *testing.T) {
	contracttest.RunScopedReadSuite(t, newPersistentDenseFixture)
}
func TestExternalPersistentTensorScopedConformance(t *testing.T) {
	contracttest.RunScopedReadSuite(t, newPersistentTensorFixture)
}

func checkPersistentPlannerScope[TIntent any](
	t *testing.T,
	factory func(*testing.T) contracttest.ScopedReadFixture[TIntent, requestMeta, sourceMeta],
) {
	t.Helper()
	for _, name := range []string{"empty-planner-filter", "conflicting-planner-filter", "unsupported-planner-filter"} {
		t.Run(name, func(t *testing.T) {
			// Arrange: forward the full plan to the real leaf, without host-side filters.
			fixture := factory(t)
			plan := retrieval.PlannedQuery[TIntent]{Intent: fixture.Request.Intent, Text: "policy"}
			switch name {
			case "conflicting-planner-filter":
				plan.Filters = fixture.Conflict
			case "unsupported-planner-filter":
				plan.Filters = fixture.Unsupported
			}
			request := fixture.Request.WithPlan(plan)
			// Act.
			result, err := fixture.Backend.Retrieve(t.Context(), request)
			// Assert.
			assertPersistentPlanner(t, fixture, name, result, err)
		})
	}
}

func assertPersistentPlanner[TIntent any](
	t *testing.T,
	fixture contracttest.ScopedReadFixture[TIntent, requestMeta, sourceMeta],
	name string,
	result retrieval.ResultSet[sourceMeta],
	err error,
) {
	t.Helper()
	if name == "unsupported-planner-filter" {
		if !errors.Is(err, ragy.ErrUnsupported) || !access.IsProtectionFailure(err) || result.Len() != 0 ||
			fixture.IOCount() != 0 {
			t.Fatal("unsupported plan dispatched", err)
		}
		return
	}
	if err != nil {
		t.Fatal(err)
	}
	if name == "conflicting-planner-filter" {
		if result.Len() != 0 || fixture.IOCount() != 0 {
			t.Fatal("contradiction read payload")
		}
		return
	}
	if result.Len() != 1 || result.Documents()[0].ID != fixture.ExpectedIDs[0] ||
		!slices.Equal(fixture.PayloadIDs(), []string{publicID}) {
		t.Fatal("empty plan removed mandatory scope")
	}
}
func TestExternalPersistentDensePlannerScope(t *testing.T) {
	checkPersistentPlannerScope(t, newPersistentDenseFixture)
}
func TestExternalPersistentTensorPlannerScope(t *testing.T) {
	checkPersistentPlannerScope(t, newPersistentTensorFixture)
}
