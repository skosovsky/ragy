package managed

import (
	"sync"
	"testing"
	"time"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/lifecycle"
)

func TestSnapshotCacheBoundsEvictsAndRejectsStaleBuilds(t *testing.T) {
	// Arrange.
	adapter := &Adapter[struct{}]{
		config: Config[struct{}]{MaxCachedSnapshots: 2},
		cache:  make(map[snapshotKey]cachedSnapshot[struct{}]),
	}
	index := &lexical.BM25Snapshot[struct{}]{}
	first := snapshotKey{binding: "first"}
	second := snapshotKey{binding: "second"}
	third := snapshotKey{binding: "third"}
	// Act: refresh first, then install third, evicting the least recently used.
	adapter.cacheSnapshot(first, index)
	adapter.cacheSnapshot(second, index)
	adapter.cached(first)
	adapter.cacheSnapshot(third, index)
	// Assert.
	if len(adapter.cache) != 2 || adapter.cached(second) != nil || adapter.cached(first) != index ||
		adapter.cached(third) != index {
		t.Fatal("cache bound/LRU failed")
	}
	// Act: mutation invalidates every resident and rejects an in-flight old build.
	adapter.mu.Lock()
	adapter.invalidateCacheLocked()
	adapter.mu.Unlock()
	adapter.cacheSnapshot(first, index)
	// Assert.
	if len(adapter.cache) != 0 || adapter.cached(first) != nil {
		t.Fatal("stale build resurrected cache")
	}
}

func TestSnapshotCacheConcurrentAdmissionRemainsBounded(t *testing.T) {
	// Arrange.
	adapter := &Adapter[struct{}]{
		config: Config[struct{}]{MaxCachedSnapshots: 2},
		cache:  make(map[snapshotKey]cachedSnapshot[struct{}]),
	}
	var workers sync.WaitGroup
	// Act.
	for i := range 16 {
		workers.Go(func() {
			key := snapshotKey{generation: 0, binding: string(rune('a' + i))}
			adapter.cacheSnapshot(key, &lexical.BM25Snapshot[struct{}]{})
			adapter.cached(key)
		})
	}
	workers.Wait()
	// Assert.
	if len(adapter.cache) != 2 {
		t.Fatalf("unbounded concurrent cache: %d", len(adapter.cache))
	}
}

func TestRetiredEmptyInventoryCannotConfirmCachedVersion(t *testing.T) {
	// Arrange: an empty target still cannot read an explicitly retired skeleton.
	identity := lifecycle.Identity{Namespace: "n", Source: "s", Revision: "r", Transformation: "t", Access: "a"}
	manifest := lifecycle.Manifest{
		ID:          "old",
		Identity:    identity,
		PublishedAt: time.Unix(1, 0),
		Retired:     true,
		Targets:     []lifecycle.Target{{Name: "lexical", State: lifecycle.TargetReady, Revision: "r"}},
	}
	version := staged[struct{}]{manifest: manifest}
	target := access.TargetRevision{
		Target:            "lexical",
		Namespace:         "n",
		Source:            "s",
		Revision:          "r",
		Transformation:    "t",
		AccessFingerprint: "a",
	}
	// Act.
	confirmed := confirmedVersion(lifecycle.Snapshot{Manifests: []lifecycle.Manifest{manifest}}, target, version)
	// Assert.
	if confirmed {
		t.Fatal("retired empty inventory admitted")
	}
}
