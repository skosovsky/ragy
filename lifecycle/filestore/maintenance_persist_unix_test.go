//go:build darwin || linux

package filestore

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// persist calls Err immediately before rename and after directory fsync. This
// context deterministically places cancellation on either side of the commit.
type commitCancellation struct {
	context.Context

	calls, cancelAt int
}

func (c *commitCancellation) Err() error {
	c.calls++
	if c.calls >= c.cancelAt {
		return context.Canceled
	}
	return nil
}

func TestPersistCancellationBeforeAndAfterRename(t *testing.T) {
	for _, at := range []int{1, 2} {
		t.Run(map[int]string{1: "before", 2: "after"}[at], func(t *testing.T) {
			// Arrange.
			root := t.TempDir()
			store, err := New(root, 4096)
			if err != nil {
				t.Fatal(err)
			}
			path := store.path("n") + ".json"
			before := []byte("known committed state")
			next := []byte("next committed state")
			if err = os.WriteFile(path, before, 0600); err != nil {
				t.Fatal(err)
			}
			ctx := &commitCancellation{Context: context.Background(), cancelAt: at}
			// Act.
			err = store.persist(ctx, "n", next)
			after, readErr := os.ReadFile(path)
			temporary, globErr := filepath.Glob(filepath.Join(root, ".lifecycle-*"))
			// Assert: canceled post-rename outcome must be reconciled, not presumed rolled back.
			expected := before
			if at == 2 {
				expected = next
			}
			if !errors.Is(err, context.Canceled) || readErr != nil || globErr != nil ||
				string(after) != string(expected) ||
				len(temporary) != 0 {
				t.Fatal("incorrect commit boundary", err, readErr, globErr, string(after), temporary)
			}
		})
	}
}

func TestPersistRenameFaultRemovesTemporaryAndPreservesCommittedState(t *testing.T) {
	// Arrange: an actual destination directory makes rename fail deterministically.
	root := t.TempDir()
	store, err := New(root, 4096)
	if err != nil {
		t.Fatal(err)
	}
	knownPath := store.path("known") + ".json"
	known := []byte("known committed state")
	if err = os.WriteFile(knownPath, known, 0600); err != nil {
		t.Fatal(err)
	}
	blocked := store.path("blocked") + ".json"
	if err = os.Mkdir(blocked, 0700); err != nil {
		t.Fatal(err)
	}
	// Act.
	err = store.persist(t.Context(), "blocked", []byte("replacement"))
	after, readErr := os.ReadFile(knownPath)
	temporary, globErr := filepath.Glob(filepath.Join(root, ".lifecycle-*"))
	info, statErr := os.Stat(blocked)
	// Assert.
	if err == nil || readErr != nil || globErr != nil || statErr != nil || !info.IsDir() ||
		string(after) != string(known) ||
		len(temporary) != 0 {
		t.Fatal("rename fault leaked temporary or modified state", err, readErr, globErr, statErr)
	}
}
