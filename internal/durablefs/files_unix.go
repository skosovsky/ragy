//go:build darwin || linux

// Package durablefs contains local filesystem durability primitives.
package durablefs

import (
	"context"
	"errors"
	"io"
	"os"
	"syscall"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/lifecycle"
)

// Lock is nonblocking and reports contention without retrying.
// The caller must not duplicate the returned handle; closing that handle releases
// its local filesystem lock. On Linux flock is tied to an open file description.
func Lock(ctx context.Context, path string, exclusive bool) (*os.File, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	mode := syscall.LOCK_SH | syscall.LOCK_NB
	if exclusive {
		mode = syscall.LOCK_EX | syscall.LOCK_NB
	}
	if err = syscall.Flock(int(file.Fd()), mode); err != nil {
		_ = file.Close()
		if errors.Is(err, syscall.EWOULDBLOCK) {
			return nil, lifecycle.ErrConflict
		}
		return nil, err
	}
	return file, nil
}
func Write(ctx context.Context, path string, data []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	defer func() { _ = file.Close() }()
	if _, err = file.Write(data); err != nil {
		return err
	}
	if err = file.Sync(); err != nil {
		return err
	}
	return ctx.Err()
}
func SyncDirectory(path string) error {
	directory, err := os.Open(path)
	if err != nil {
		return err
	}
	defer func() { _ = directory.Close() }()
	return directory.Sync()
}

// ReadBounded rejects oversized/corrupt input without loading it unboundedly.
func ReadBounded(ctx context.Context, path string, limit int64) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if limit <= 0 || limit == int64(^uint64(0)>>1) {
		return nil, ragy.ErrInvalidArgument
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()
	data, err := io.ReadAll(io.LimitReader(file, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, ragy.ErrProtocol
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	return data, nil
}
