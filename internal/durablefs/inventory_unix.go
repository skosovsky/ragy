//go:build darwin || linux

package durablefs

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"

	ragy "github.com/skosovsky/ragy"
)

const inventoryReadBatch = 64

// InventoryKeys enumerates bounded immediate keys without reading unknown payloads.
// Callers hold their target fence. The supplied reserved key is never inventory data.
func InventoryKeys(ctx context.Context, root, reserved string, limit int) (map[string]struct{}, error) {
	if limit <= 0 {
		return nil, ragy.ErrInvalidArgument
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	directory, err := os.Open(root)
	if err != nil {
		return nil, err
	}
	defer func() { _ = directory.Close() }()
	keys := make(map[string]struct{})
	for {
		entries, readErr := directory.ReadDir(inventoryReadBatch)
		for _, entry := range entries {
			if err = ctx.Err(); err != nil {
				return nil, err
			}
			if entry.Name() == reserved {
				continue
			}
			if len(keys) == limit {
				return nil, ragy.ErrUnavailable
			}
			keys[entry.Name()] = struct{}{}
		}
		if errors.Is(readErr, io.EOF) {
			return keys, ctx.Err()
		}
		if readErr != nil {
			return nil, readErr
		}
	}
}

// ValidInventoryKey is one opaque immediate entry, never a path to other storage.
func ValidInventoryKey(key string) bool {
	return key != "" && key != "." && key != ".." && filepath.Base(key) == key && key != "target.lock"
}
