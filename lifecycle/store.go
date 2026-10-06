package lifecycle

import (
	"context"
	"errors"
)

// ErrConflict means the expected generation no longer matches or a writer is busy.
var ErrConflict = errors.New("lifecycle compare-and-swap conflict")

// Store atomically replaces an owned namespace snapshot. CompareSwap increments
// generation once; callers must reconcile uncertain I/O outcomes by Load, not infer
// rollback. CompareSwap must apply ValidateReplacement to preserve retired and pin
// reservations. Store does not execute target operations or retry a lifecycle workflow.
type Store interface {
	Load(context.Context, string) (Snapshot, error)
	CompareSwap(context.Context, uint64, Snapshot) (Snapshot, error)
}

// ErrCapacity means the explicit local storage byte budget cannot admit this state.
var ErrCapacity = errors.New("lifecycle storage capacity exceeded")

// ErrRetired means a reserved operation or pin handle has been explicitly retired.
var ErrRetired = errors.New("lifecycle handle retired")

// ErrProtected means requested metadata retirement intersects protected state.
var ErrProtected = errors.New("lifecycle history protected")

// RetirementRequest is an exact host-selected list, not a retention policy.
// Maintenance never invokes target cleanup or deletes source payload.
type RetirementRequest struct {
	Namespace string
	Manifests []string
}

// MaintenanceStore is an optional storage-local compaction capability. It shares
// generation CAS, durability and uncertain-outcome semantics with Store.
type MaintenanceStore interface {
	Store
	Maintain(context.Context, uint64, RetirementRequest) (Snapshot, error)
}
