package lifecycle

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"slices"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/source"
)

var (
	ErrIdempotencyConflict = errors.New("lifecycle idempotency conflict")
	ErrOutcomeUnknown      = errors.New("lifecycle target outcome unknown")
)

// StageResult is a target's confirmed observation, not a guess based on timeout.
// Ready requires the exact planned revision. Pending is allowed only from Inspect.
type StageResult struct {
	State    TargetState
	Revision string
}

// StageRequest holds an owned full manifest and the named target's planned inventory.
type StageRequest struct {
	Manifest Manifest
	Target   string
}

// StagePort stages exact revision-bound records outside published read visibility.
// Inspect must observe actual backend outcome; it never silently repeats Stage.
// Target cleanup and read capabilities are separate from this staging contract.
type StagePort[TPayload any] interface {
	Stage(context.Context, StageRequest, TPayload) (StageResult, error)
	Inspect(context.Context, StageRequest) (StageResult, error)
}

type Registration[TPayload any] struct {
	Name string
	Port StagePort[TPayload]
}

type ExecutorConfig[TPayload any] struct {
	Now             func() time.Time
	Store           Store
	Targets         []Registration[TPayload]
	ClonePayload    func(TPayload) (TPayload, error)
	ValidatePayload func(Manifest, TPayload) error
}

// Executor performs bounded explicit operations; scheduling belongs to the host.
type Executor[TPayload any] struct{ config ExecutorConfig[TPayload] }

func NewExecutor[TPayload any](config ExecutorConfig[TPayload]) (*Executor[TPayload], error) {
	if nilPort(config.Store) || config.ClonePayload == nil || config.ValidatePayload == nil {
		return nil, ragy.ErrInvalidArgument
	}
	if config.Now == nil {
		config.Now = time.Now
	}
	config.Targets = slices.Clone(config.Targets)
	names := make(map[string]struct{}, len(config.Targets))
	for _, target := range config.Targets {
		if !identities(target.Name) || nilPort(target.Port) {
			return nil, ragy.ErrInvalidArgument
		}
		if _, exists := names[target.Name]; exists {
			return nil, ragy.ErrInvalidArgument
		}
		names[target.Name] = struct{}{}
	}
	return &Executor[TPayload]{config: config}, nil
}

// Prepare is idempotent for the entire unchanged operation plan. It never stages.
func (e *Executor[TPayload]) Prepare(ctx context.Context, plan Manifest) (Manifest, error) {
	if e == nil {
		return Manifest{}, ragy.ErrInvalidArgument
	}
	plan = cloneManifest(plan)
	if err := e.validatePlan(plan); err != nil {
		return Manifest{}, err
	}
	snapshot, err := e.load(ctx, plan.Identity.Namespace)
	if err != nil {
		return Manifest{}, err
	}
	for _, previous := range snapshot.Manifests {
		if previous.Key == plan.Key {
			if !samePlan(previous, plan) {
				return Manifest{}, ErrIdempotencyConflict
			}
			return cloneManifest(previous), nil
		}
		if previous.ID == plan.ID {
			return Manifest{}, ErrIdempotencyConflict
		}
	}
	if overlappingInventory(snapshot, plan) {
		return Manifest{}, ErrConflict
	}
	if activePublication(snapshot, plan.Identity.Source) != plan.ExpectedPublication {
		return Manifest{}, ErrConflict
	}
	snapshot.Manifests = append(snapshot.Manifests, plan)
	if _, err = e.config.Store.CompareSwap(ctx, snapshot.Generation, snapshot); err != nil {
		return Manifest{}, mutationError(err)
	}
	return cloneManifest(plan), nil
}

// Stage marks dispatch unknown durably first; retries of unknown require Reconcile.
func (e *Executor[TPayload]) Stage(
	ctx context.Context, namespace, id, targetName string, payload TPayload,
) (Manifest, error) {
	snapshot, index, targetIndex, port, err := e.operation(ctx, namespace, id, targetName)
	if err != nil {
		return Manifest{}, err
	}
	manifest := snapshot.Manifests[index]
	target := manifest.Targets[targetIndex]
	if activePublication(snapshot, manifest.Identity.Source) != manifest.ExpectedPublication {
		return Manifest{}, ErrConflict
	}
	captured, err := e.config.ClonePayload(payload)
	if err != nil {
		return Manifest{}, err
	}
	if err = e.config.ValidatePayload(cloneManifest(manifest), captured); err != nil {
		return Manifest{}, err
	}
	if err = ctx.Err(); err != nil {
		return Manifest{}, err
	}
	if target.State == TargetReady {
		return cloneManifest(manifest), nil
	}
	if target.State == TargetUnknown {
		return cloneManifest(manifest), ErrOutcomeUnknown
	}
	manifest.State, manifest.Checkpoint = Unknown, Staging
	manifest.Targets[targetIndex].State = TargetUnknown
	manifest.Targets[targetIndex].Revision = ""
	snapshot.Manifests[index] = manifest
	if _, err = e.config.Store.CompareSwap(ctx, snapshot.Generation, snapshot); err != nil {
		return Manifest{}, mutationError(err)
	}
	if err = ctx.Err(); err != nil {
		return cloneManifest(manifest), errors.Join(ErrOutcomeUnknown, err)
	}
	result, dispatchErr := port.Stage(
		ctx,
		StageRequest{Manifest: cloneManifest(manifest), Target: targetName},
		captured,
	)
	if dispatchErr != nil {
		return cloneManifest(manifest), errors.Join(ErrOutcomeUnknown, dispatchErr)
	}
	return e.confirm(ctx, namespace, id, targetName, result, false)
}

// Reconcile observes a previously uncertain target exactly once, without Stage.
func (e *Executor[TPayload]) Reconcile(ctx context.Context, namespace, id, targetName string) (Manifest, error) {
	snapshot, index, targetIndex, port, err := e.operation(ctx, namespace, id, targetName)
	if err != nil {
		return Manifest{}, err
	}
	manifest := snapshot.Manifests[index]
	if manifest.Targets[targetIndex].State != TargetUnknown {
		return cloneManifest(manifest), nil
	}
	result, inspectErr := port.Inspect(ctx, StageRequest{Manifest: cloneManifest(manifest), Target: targetName})
	if inspectErr != nil {
		return cloneManifest(manifest), errors.Join(ErrOutcomeUnknown, inspectErr)
	}
	return e.confirm(ctx, namespace, id, targetName, result, true)
}

// Publish performs source expected-publication and namespace generation CAS together.
// Cancellation/commit error requires Load reconciliation, never rollback inference.
func (e *Executor[TPayload]) Publish(ctx context.Context, namespace, id string) (Manifest, error) {
	snapshot, err := e.load(ctx, namespace)
	if err != nil {
		return Manifest{}, err
	}
	index := manifestIndex(snapshot, id)
	if index < 0 {
		return Manifest{}, ragy.ErrUnavailable
	}
	manifest := snapshot.Manifests[index]
	effective, err := confirmedState(manifest.State, manifest.Checkpoint)
	if err != nil {
		return Manifest{}, err
	}
	if afterPublished(effective) {
		return cloneManifest(manifest), nil
	}
	if activePublication(snapshot, manifest.Identity.Source) != manifest.ExpectedPublication {
		return Manifest{}, ErrConflict
	}
	candidate := cloneManifest(manifest)
	candidate.State, candidate.Checkpoint = Published, ""
	candidate.PublishedAt = e.config.Now().UTC()
	if !candidate.Tombstone && !candidate.Partial {
		for _, target := range candidate.Targets {
			if target.State == TargetUnknown {
				return Manifest{}, ErrOutcomeUnknown
			}
		}
	}
	if err = candidate.Validate(); err != nil {
		return Manifest{}, err
	}
	snapshot.Manifests[index] = candidate
	replaced := false
	for i, publication := range snapshot.Publications {
		if publication.Source == candidate.Identity.Source {
			snapshot.Publications[i].Manifest = candidate.ID
			replaced = true
		}
	}
	if !replaced {
		snapshot.Publications = append(
			snapshot.Publications,
			Publication{Source: candidate.Identity.Source, Manifest: candidate.ID},
		)
	}
	if _, err = e.config.Store.CompareSwap(ctx, snapshot.Generation, snapshot); err != nil {
		return Manifest{}, mutationError(err)
	}
	if err = ctx.Err(); err != nil {
		return cloneManifest(candidate), errors.Join(ErrOutcomeUnknown, err)
	}
	return cloneManifest(candidate), nil
}

func (e *Executor[TPayload]) load(ctx context.Context, namespace string) (Snapshot, error) {
	if e == nil || !identities(namespace) {
		return Snapshot{}, ragy.ErrInvalidArgument
	}
	if err := ctx.Err(); err != nil {
		return Snapshot{}, err
	}
	snapshot, err := e.config.Store.Load(ctx, namespace)
	if err != nil {
		return Snapshot{}, err
	}
	if snapshot.Namespace != namespace {
		return Snapshot{}, ragy.ErrProtocol
	}
	if err = snapshot.Validate(); err != nil {
		return Snapshot{}, ragy.ErrProtocol
	}
	if err = ctx.Err(); err != nil {
		return Snapshot{}, err
	}
	return snapshot, nil
}

func (e *Executor[TPayload]) validatePlan(plan Manifest) error {
	if plan.State != Planned || plan.Checkpoint != "" {
		return ragy.ErrInvalidArgument
	}
	if err := plan.Validate(); err != nil {
		return err
	}
	for _, target := range plan.Targets {
		if target.State != TargetPending || target.Revision != "" {
			return ragy.ErrInvalidArgument
		}
		if e.port(target.Name) == nil {
			return ragy.ErrUnsupported
		}
	}
	return nil
}

func (e *Executor[TPayload]) port(name string) StagePort[TPayload] {
	for _, registered := range e.config.Targets {
		if registered.Name == name {
			return registered.Port
		}
	}
	return nil
}

func (e *Executor[TPayload]) operation(
	ctx context.Context, namespace, id, name string,
) (Snapshot, int, int, StagePort[TPayload], error) {
	snapshot, err := e.load(ctx, namespace)
	if err != nil {
		return Snapshot{}, 0, 0, nil, err
	}
	index := manifestIndex(snapshot, id)
	if index < 0 {
		return Snapshot{}, 0, 0, nil, ragy.ErrUnavailable
	}
	manifest := snapshot.Manifests[index]
	state, err := confirmedState(manifest.State, manifest.Checkpoint)
	if err != nil {
		return Snapshot{}, 0, 0, nil, err
	}
	if afterPublished(state) || manifest.Tombstone {
		return Snapshot{}, 0, 0, nil, ErrConflict
	}
	for targetIndex, target := range manifest.Targets {
		if target.Name == name {
			port := e.port(name)
			if port == nil {
				return Snapshot{}, 0, 0, nil, ragy.ErrUnsupported
			}
			return snapshot, index, targetIndex, port, nil
		}
	}
	return Snapshot{}, 0, 0, nil, ragy.ErrInvalidArgument
}

func (e *Executor[TPayload]) confirm(
	ctx context.Context, namespace, id, name string, result StageResult, inspect bool,
) (Manifest, error) {
	snapshot, index, targetIndex, _, err := e.operation(ctx, namespace, id, name)
	if err != nil {
		return Manifest{}, errors.Join(ErrOutcomeUnknown, err)
	}
	manifest := snapshot.Manifests[index]
	if manifest.Targets[targetIndex].State != TargetUnknown {
		return Manifest{}, ErrConflict
	}
	switch result.State {
	case TargetReady:
		if result.Revision != manifest.Identity.Revision {
			return cloneManifest(manifest), errors.Join(ErrOutcomeUnknown, ragy.ErrProtocol)
		}
	case TargetPending:
		if !inspect || result.Revision != "" {
			return cloneManifest(manifest), errors.Join(ErrOutcomeUnknown, ragy.ErrProtocol)
		}
	case TargetFailed, TargetUnknown:
		if result.Revision != "" {
			return cloneManifest(manifest), errors.Join(ErrOutcomeUnknown, ragy.ErrProtocol)
		}
	default:
		return cloneManifest(manifest), errors.Join(ErrOutcomeUnknown, ragy.ErrProtocol)
	}
	manifest.Targets[targetIndex].State = result.State
	manifest.Targets[targetIndex].Revision = result.Revision
	manifest.State, manifest.Checkpoint = Staging, ""
	readyCandidate := cloneManifest(manifest)
	readyCandidate.State = Ready
	if readyCandidate.Validate() == nil {
		manifest.State = Ready
	}
	if result.State == TargetUnknown {
		manifest.State, manifest.Checkpoint = Unknown, Staging
	}
	if result.State == TargetFailed {
		manifest.State, manifest.Checkpoint = Failed, Staging
	}
	snapshot.Manifests[index] = manifest
	if _, err = e.config.Store.CompareSwap(ctx, snapshot.Generation, snapshot); err != nil {
		return cloneManifest(manifest), errors.Join(ErrOutcomeUnknown, err)
	}
	if err = ctx.Err(); err != nil {
		return cloneManifest(manifest), errors.Join(ErrOutcomeUnknown, err)
	}
	switch result.State {
	case TargetUnknown:
		return cloneManifest(manifest), ErrOutcomeUnknown
	case TargetFailed:
		return cloneManifest(manifest), ragy.ErrUnavailable
	case TargetPending, TargetReady:
		return cloneManifest(manifest), nil
	default:
		return Manifest{}, ragy.ErrProtocol
	}
}

func manifestIndex(snapshot Snapshot, id string) int {
	for i, manifest := range snapshot.Manifests {
		if manifest.ID == id {
			return i
		}
	}
	return -1
}
func activePublication(snapshot Snapshot, sourceID string) string {
	for _, publication := range snapshot.Publications {
		if publication.Source == sourceID {
			return publication.Manifest
		}
	}
	return ""
}
func cloneManifest(manifest Manifest) Manifest {
	manifest.Targets = slices.Clone(manifest.Targets)
	for i := range manifest.Targets {
		manifest.Targets[i].Artifacts = slices.Clone(manifest.Targets[i].Artifacts)
		for j := range manifest.Targets[i].Artifacts {
			manifest.Targets[i].Artifacts[j].Supports = slices.Clone(manifest.Targets[i].Artifacts[j].Supports)
		}
	}
	return manifest
}
func samePlan(first, second Manifest) bool {
	normalize := func(manifest Manifest) []byte {
		manifest = cloneManifest(manifest)
		manifest.State, manifest.Checkpoint = Planned, ""
		manifest.PublishedAt = time.Time{}
		for i := range manifest.Targets {
			manifest.Targets[i].State, manifest.Targets[i].Revision = TargetPending, ""
		}
		data, _ := json.Marshal(manifest)
		return data
	}
	return slices.Equal(normalize(first), normalize(second))
}
func nilPort(port any) bool {
	if port == nil {
		return true
	}
	value := reflect.ValueOf(port)
	switch value.Kind() {
	case reflect.Pointer, reflect.Interface, reflect.Map, reflect.Slice, reflect.Func, reflect.Chan:
		return value.IsNil()
	case reflect.Invalid, reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128, reflect.Array, reflect.String,
		reflect.Struct, reflect.UnsafePointer:
		return false
	}
	return false
}

func mutationError(err error) error {
	if errors.Is(err, ErrConflict) {
		return err
	}
	return errors.Join(ErrOutcomeUnknown, err)
}

func overlappingInventory(snapshot Snapshot, plan Manifest) bool {
	type key struct {
		target    string
		reference source.Reference
	}
	occupied := make(map[key]struct{})
	for _, previous := range snapshot.Manifests {
		for _, existing := range previous.Targets {
			for _, artifact := range existing.Artifacts {
				occupied[key{target: existing.Name, reference: artifact.Reference}] = struct{}{}
			}
		}
	}
	for _, target := range plan.Targets {
		for _, artifact := range target.Artifacts {
			if _, exists := occupied[key{target: target.Name, reference: artifact.Reference}]; exists {
				return true
			}
		}
	}
	return false
}
