package lifecycle

import (
	"context"
	"errors"
)

// ErrConflict means the expected generation no longer matches or a writer is busy.
var ErrConflict = errors.New("lifecycle compare-and-swap conflict")

// Store atomically replaces an owned namespace snapshot. CompareSwap increments
// generation once; callers must reconcile uncertain I/O outcomes by Load, not infer
// rollback. Store does not execute target operations or retry a lifecycle workflow.
type Store interface {
	Load(context.Context, string) (Snapshot, error)
	CompareSwap(context.Context, uint64, Snapshot) (Snapshot, error)
}
