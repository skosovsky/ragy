//go:build darwin || linux

package history

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/internal/durablefs"
)

// FileStore appends immutable, content-addressed records in a host-owned directory.
// It has no latest pointer, implicit retention policy, overwrite or delete method.
type FileStore[TKind, TRel comparable, TAttr any] struct {
	root        string
	maxBytes    int
	maxSupports int
	admit       Admission
}

func NewFileStore[TKind, TRel comparable, TAttr any](
	root string,
	maxBytes, maxSupports int,
	admit Admission,
) (*FileStore[TKind, TRel, TAttr], error) {
	if root == "" || maxBytes <= 0 || maxBytes == math.MaxInt || maxSupports <= 0 || admit == nil {
		return nil, ragy.ErrInvalidArgument
	}
	absolute, err := filepath.Abs(root)
	if err != nil {
		return nil, ragy.ErrInvalidArgument
	}
	return &FileStore[TKind, TRel, TAttr]{
		root:        absolute,
		maxBytes:    maxBytes,
		maxSupports: maxSupports,
		admit:       admit,
	}, nil
}

func (s *FileStore[TKind, TRel, TAttr]) filename(reference Reference) (string, error) {
	if s == nil || !validID(reference.ID) || len(reference.Supports) == 0 || len(reference.Supports) > s.maxSupports {
		return "", ragy.ErrInvalidArgument
	}
	metadata, err := json.Marshal(reference)
	if err != nil {
		return "", ragy.ErrInvalidArgument
	}
	digest := sha256.Sum256(metadata)
	return filepath.Join(s.root, hex.EncodeToString(digest[:])+".json"), nil
}

// Append returns only after payload and directory synchronization. Retrying the
// identical snapshot is idempotent. Failure/cancellation after the atomic link may
// mean it already exists; inspect Read explicitly rather than creating a new run.
func (s *FileStore[TKind, TRel, TAttr]) Append(
	ctx context.Context,
	read access.Binding,
	snapshot Snapshot[TKind, TRel, TAttr],
) error {
	reference := snapshot.Reference()
	path, err := s.filename(reference)
	if err != nil {
		return err
	}
	if len(snapshot.data) == 0 || len(snapshot.data) > s.maxBytes {
		return ragy.ErrInvalidArgument
	}
	if err = authorize(ctx, read, reference.Supports, s.admit); err != nil {
		return err
	}
	if err = os.MkdirAll(s.root, 0o700); err != nil {
		return err
	}
	temporary, err := s.stage(ctx, snapshot.data)
	if err != nil {
		return err
	}
	defer func() { _ = os.Remove(temporary) }()
	if err = read.Check(ctx); err != nil {
		return err
	}
	if err = os.Link(temporary, path); err != nil {
		if !errors.Is(err, os.ErrExist) {
			return err
		}
		current, readErr := durablefs.ReadBounded(ctx, path, int64(s.maxBytes))
		if readErr != nil {
			return readErr
		}
		if !bytes.Equal(current, snapshot.data) {
			return ragy.ErrProtocol
		}
	}
	if err = durablefs.SyncDirectory(s.root); err != nil {
		return err
	}
	return read.Check(ctx)
}

func (s *FileStore[TKind, TRel, TAttr]) stage(ctx context.Context, data []byte) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	file, err := os.CreateTemp(s.root, ".resolution-stage-")
	if err != nil {
		return "", err
	}
	name := file.Name()
	defer func() { _ = file.Close() }()
	if _, err = file.Write(data); err == nil {
		err = file.Sync()
	}
	if err != nil {
		_ = os.Remove(name)
		return "", err
	}
	if err = ctx.Err(); err != nil {
		_ = os.Remove(name)
		return "", err
	}
	return name, nil
}

// Read authorizes the complete locator inventory before any payload I/O and
// verifies the content digest and the exact decoded support inventory. Host may
// allow retained historical revisions explicitly; current authorization is never
// inferred from an old successful write.
func (s *FileStore[TKind, TRel, TAttr]) Read(
	ctx context.Context,
	read access.Binding,
	reference Reference,
) (Snapshot[TKind, TRel, TAttr], error) {
	var empty Snapshot[TKind, TRel, TAttr]
	path, err := s.filename(reference)
	if err != nil {
		return empty, err
	}
	reference.Supports = slices.Clone(reference.Supports)
	if err = authorize(ctx, read, reference.Supports, s.admit); err != nil {
		return empty, err
	}
	data, err := durablefs.ReadBounded(ctx, path, int64(s.maxBytes))
	if err != nil {
		return empty, err
	}
	if err = read.Check(ctx); err != nil {
		return empty, err
	}
	digest := sha256.Sum256(data)
	if reference.ID != hex.EncodeToString(digest[:]) {
		return empty, ragy.ErrProtocol
	}
	var record Record[TKind, TRel, TAttr]
	if err = decode(data, &record); err != nil {
		return empty, err
	}
	supports, err := locations(record, s.maxSupports)
	if err != nil || !slices.Equal(supports, reference.Supports) {
		return empty, ragy.ErrProtocol
	}
	if err = read.Check(ctx); err != nil {
		return empty, err
	}
	return Snapshot[TKind, TRel, TAttr]{
		data:      data,
		reference: Reference{ID: reference.ID, Supports: slices.Clone(reference.Supports)},
	}, nil
}
