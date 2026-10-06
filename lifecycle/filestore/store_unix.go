//go:build darwin || linux

// Package filestore supplies an optional durable filesystem lifecycle store.
package filestore

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"syscall"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/internal/durablefs"
	"github.com/skosovsky/ragy/lifecycle"
)

// Store persists a namespace snapshot under an exclusive cross-process file lock.
// It requires a local filesystem supporting flock, atomic rename and directory
// fsync. Root permissions/backup/retention are host responsibilities.
type Store struct {
	root             string
	maxSnapshotBytes int64
}

// New requires an explicit host storage budget, shared by reads and writes.
func New(root string, maxSnapshotBytes int64) (*Store, error) {
	if root == "" || maxSnapshotBytes <= 0 || maxSnapshotBytes == math.MaxInt64 {
		return nil, ragy.ErrInvalidArgument
	}
	absolute, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	if err = os.MkdirAll(absolute, 0o700); err != nil {
		return nil, err
	}
	return &Store{root: absolute, maxSnapshotBytes: maxSnapshotBytes}, nil
}

func (s *Store) Load(ctx context.Context, namespace string) (lifecycle.Snapshot, error) {
	if err := ctx.Err(); err != nil {
		return lifecycle.Snapshot{}, err
	}
	if s == nil || !validNamespace(namespace) {
		return lifecycle.Snapshot{}, ragy.ErrInvalidArgument
	}
	snapshot, err := s.load(ctx, namespace)
	if err != nil {
		return lifecycle.Snapshot{}, err
	}
	if err = ctx.Err(); err != nil {
		return lifecycle.Snapshot{}, err
	}
	return snapshot, nil
}

func (s *Store) load(ctx context.Context, namespace string) (lifecycle.Snapshot, error) {
	data, err := durablefs.ReadBounded(ctx, s.path(namespace)+".json", s.maxSnapshotBytes)
	if errors.Is(err, os.ErrNotExist) {
		return lifecycle.Snapshot{
			Schema:       lifecycle.SchemaIdentity,
			Namespace:    namespace,
			Generation:   0,
			Manifests:    nil,
			Publications: nil,
			Cleanups:     nil,
			Inventories:  nil,
		}, nil
	}
	if err != nil {
		return lifecycle.Snapshot{}, err
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var snapshot lifecycle.Snapshot
	if err = decoder.Decode(&snapshot); err != nil {
		return lifecycle.Snapshot{}, ragy.ErrProtocol
	}
	var trailing any
	if err = decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return lifecycle.Snapshot{}, ragy.ErrProtocol
	}
	if snapshot.Namespace != namespace || snapshot.Generation == 0 {
		return lifecycle.Snapshot{}, ragy.ErrProtocol
	}
	if err = snapshot.Validate(); err != nil {
		return lifecycle.Snapshot{}, ragy.ErrProtocol
	}
	return snapshot, nil
}

func (s *Store) CompareSwap(
	ctx context.Context, expected uint64, next lifecycle.Snapshot,
) (lifecycle.Snapshot, error) {
	if err := ctx.Err(); err != nil {
		return lifecycle.Snapshot{}, err
	}
	if s == nil || expected == math.MaxUint64 || next.Generation != expected {
		return lifecycle.Snapshot{}, ragy.ErrInvalidArgument
	}
	if err := next.Validate(); err != nil {
		return lifecycle.Snapshot{}, err
	}
	lock, err := os.OpenFile(s.path(next.Namespace)+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return lifecycle.Snapshot{}, err
	}
	// Closing releases the advisory lock, including process death.
	defer func() { _ = lock.Close() }()
	if err = syscall.Flock(int(lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		if errors.Is(err, syscall.EWOULDBLOCK) {
			return lifecycle.Snapshot{}, lifecycle.ErrConflict
		}
		return lifecycle.Snapshot{}, err
	}
	current, err := s.load(ctx, next.Namespace)
	if err != nil {
		return lifecycle.Snapshot{}, err
	}
	if current.Generation != expected {
		return lifecycle.Snapshot{}, lifecycle.ErrConflict
	}
	next.Generation = expected + 1
	data, err := json.Marshal(next)
	if err != nil {
		return lifecycle.Snapshot{}, err
	}
	if int64(len(data)) > s.maxSnapshotBytes {
		return lifecycle.Snapshot{}, ragy.ErrInvalidArgument
	}
	if err = s.persist(ctx, next.Namespace, data); err != nil {
		return lifecycle.Snapshot{}, err
	}
	// Decode the captured bytes to return an owned snapshot, never caller slices.
	var owned lifecycle.Snapshot
	if err = json.Unmarshal(data, &owned); err != nil {
		return lifecycle.Snapshot{}, ragy.ErrProtocol
	}
	return owned, nil
}

func (s *Store) persist(ctx context.Context, namespace string, data []byte) error {
	temporary, err := os.CreateTemp(s.root, ".lifecycle-*")
	if err != nil {
		return err
	}
	temporaryPath := temporary.Name()
	defer func() { _ = temporary.Close(); _ = os.Remove(temporaryPath) }()
	if _, err = temporary.Write(data); err != nil {
		return err
	}
	if err = temporary.Sync(); err != nil {
		return err
	}
	if err = temporary.Close(); err != nil {
		return err
	}
	if err = ctx.Err(); err != nil {
		return err
	}
	if err = os.Rename(temporaryPath, s.path(namespace)+".json"); err != nil {
		return err
	}
	directory, err := os.Open(s.root)
	if err != nil {
		return err
	}
	defer func() { _ = directory.Close() }()
	if err = directory.Sync(); err != nil {
		return err
	}
	// An error here means commit may have succeeded; reconcile using Load.
	return ctx.Err()
}
func (s *Store) path(namespace string) string {
	digest := sha256.Sum256([]byte(namespace))
	return filepath.Join(s.root, hex.EncodeToString(digest[:]))
}
func validNamespace(namespace string) bool { return namespace != "" && utf8.ValidString(namespace) }

var _ lifecycle.Store = (*Store)(nil)
