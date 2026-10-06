//go:build darwin || linux

package durablefs_test

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/internal/durablefs"
)

func TestInventoryKeysBoundsOpaqueEntriesAndExcludesReservedLock(t *testing.T) {
	// Arrange: no unknown payload is interpreted; both file and directory are keys.
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "target.lock"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "opaque"), []byte("not a catalog"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(root, ".retired-opaque"), 0o700); err != nil {
		t.Fatal(err)
	}
	// Act: the lock does not consume an inventory slot.
	keys, err := durablefs.InventoryKeys(t.Context(), root, "target.lock", 2)
	// Assert.
	if err != nil || len(keys) != 2 {
		t.Fatal("opaque enumeration", keys, err)
	}
	if _, exists := keys["target.lock"]; exists {
		t.Fatal("lock treated as data")
	}
	if _, err = durablefs.InventoryKeys(t.Context(), root, "target.lock", 1); !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatal("unbounded enumeration", err)
	}
}

func TestOpaqueInventoryKeyCannotReferenceOtherStorage(t *testing.T) {
	for _, key := range []string{"", ".", "..", "target.lock", "../other", "/other", "nested/other"} {
		// Arrange/Act/Assert: only a nonreserved immediate basename can claim an entry.
		if durablefs.ValidInventoryKey(key) {
			t.Fatal("path accepted as inventory key", key)
		}
	}
	for _, key := range []string{"opaque", ".retired-opaque", "unknown-source"} {
		if !durablefs.ValidInventoryKey(key) {
			t.Fatal("opaque basename rejected", key)
		}
	}
}
