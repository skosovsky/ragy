package bridge

import (
	"context"
	"slices"
	"sync"

	"github.com/skosovsky/memy"
)

// SnapshotSink is a managed in-process host sink, not a durable storage service.
// It demonstrates exact-scope canonical Forget cleanup and owned serialized state.
type SnapshotSink[U any] struct {
	mu              sync.Mutex
	name            string
	uncertaintyType string
	snapshots       map[string]Published[U]
	fences          map[memy.Scope]memy.Version
}

// NewSnapshotSink returns an empty sink with an explicit codec identity.
func NewSnapshotSink[U any](name, uncertaintyType string) (*SnapshotSink[U], error) {
	if name == "" || uncertaintyType == "" {
		return nil, memy.ErrInvalid
	}
	return &SnapshotSink[U]{
		name:            name,
		uncertaintyType: uncertaintyType,
		snapshots:       make(map[string]Published[U]),
		fences:          make(map[memy.Scope]memy.Version),
	}, nil
}

// Name identifies the participant registered with the canonical engine.
func (s *SnapshotSink[U]) Name() string { return s.name }

// Publish owns the validated bytes and rejects cross-scope message identity reuse.
func (s *SnapshotSink[U]) Publish(ctx context.Context, p Published[U]) error {
	decoded, err := Decode[U](ctx, p.Durable, Registry[U](s.uncertaintyType))
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(decoded.Evidence.References) == 0 {
		return memy.ErrMissingEvidence
	}
	if decoded.Evidence.Epoch < s.fences[decoded.Evidence.Scope] {
		return memy.ErrRevoked
	}
	if previous, ok := s.snapshots[decoded.Message.ID]; ok && previous.Evidence.Scope != decoded.Evidence.Scope {
		return memy.ErrScopeViolation
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.snapshots[decoded.Message.ID] = decoded
	return nil
}

// Purge deletes snapshots supported by any record in this exact canonical batch.
func (s *SnapshotSink[U]) Purge(ctx context.Context, b memy.PurgeBatch) (memy.PurgeAck, error) {
	if b.Scope.Validate() != nil || b.Epoch == 0 || b.Epoch > memy.MaxVersion || b.OperationID == "" {
		return memy.PurgeAck{}, memy.ErrInvalid
	}
	if err := ctx.Err(); err != nil {
		return memy.PurgeAck{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if b.Epoch > s.fences[b.Scope] {
		s.fences[b.Scope] = b.Epoch
	}
	for id, p := range s.snapshots {
		if p.Evidence.Scope != b.Scope || p.Evidence.Epoch >= b.Epoch {
			continue
		}
		for _, ref := range p.Evidence.References {
			if slices.Contains(b.Records, ref.RecordID) {
				delete(s.snapshots, id)
				break
			}
		}
	}
	return memy.PurgeAck{Sink: s.name, OperationID: b.OperationID, Epoch: b.Epoch, Chunk: b.Chunk}, nil
}

// Snapshots returns owned durable bytes, sorted by message identity.
func (s *SnapshotSink[U]) Snapshots() [][]byte {
	s.mu.Lock()
	defer s.mu.Unlock()
	ids := make([]string, 0, len(s.snapshots))
	for id := range s.snapshots {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	out := make([][]byte, 0, len(ids))
	for _, id := range ids {
		out = append(out, slices.Clone(s.snapshots[id].Durable))
	}
	return out
}

// CleanupSink binds canonical Forget to a finite host-owned index lifecycle operation.
// Apply must tombstone/clean the complete exact inventory and be idempotent for the batch.
// Return an error when cleanup is pending; this prevents a false purge acknowledgement.
type CleanupSink struct {
	ID    string
	Apply func(context.Context, memy.PurgeBatch) error
}

func (s CleanupSink) Name() string { return s.ID }
func (s CleanupSink) Purge(ctx context.Context, b memy.PurgeBatch) (memy.PurgeAck, error) {
	if s.ID == "" || s.Apply == nil {
		return memy.PurgeAck{}, memy.ErrInvalid
	}
	if err := ctx.Err(); err != nil {
		return memy.PurgeAck{}, err
	}
	if err := s.Apply(ctx, b); err != nil {
		return memy.PurgeAck{}, err
	}
	if err := ctx.Err(); err != nil {
		return memy.PurgeAck{}, err
	}
	return memy.PurgeAck{Sink: s.ID, OperationID: b.OperationID, Epoch: b.Epoch, Chunk: b.Chunk}, nil
}
