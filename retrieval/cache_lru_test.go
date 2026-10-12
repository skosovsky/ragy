package retrieval

import (
	"context"
	"errors"
	"maps"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"
)

func assertCacheList(t *testing.T, cache *MemoryCache[struct{}]) {
	t.Helper()
	cache.mu.Lock()
	defer cache.mu.Unlock()
	seen := map[*memoryCacheItem[struct{}]]bool{}
	var prev *memoryCacheItem[struct{}]
	for item := cache.oldest; item != nil; item = item.next {
		if seen[item] || item.prev != prev || cache.entries[item.key] != item {
			t.Fatal("invalid LRU linkage or retained detached node")
		}
		seen[item] = true
		prev = item
	}
	if prev != cache.newest || len(seen) != len(cache.entries) || len(seen) > cache.maxEntries {
		t.Fatal("LRU retained state exceeds live entries/capacity")
	}
}

func TestMemoryCacheBoundedRetentionAndOverwriteLRU(t *testing.T) {
	// Arrange.
	cache, entry, keys := cachePerformanceFixture(t, 64)
	ctx := context.Background()
	// Act: repeatedly evict while reusing a fixed pool of live nodes.
	nodes := map[*memoryCacheItem[struct{}]]bool{}
	for _, item := range cache.entries {
		nodes[item] = true
	}
	for i := range 20000 {
		if err := cache.Store(ctx, keys[i%len(keys)], entry); err != nil {
			t.Fatal(err)
		}
	}
	// Assert: no accumulating tombstones or detached list nodes, and bounded node identities.
	assertCacheList(t, cache)
	for _, item := range cache.entries {
		if !nodes[item] {
			t.Fatal("eviction allocated an additional retained node")
		}
	}
	var bookkeeping memoryCacheItem[struct{}]
	t.Logf(
		"64-bit LRU bookkeeping: %d bytes per node excluding CacheEntry; map values now pointers",
		unsafe.Sizeof(bookkeeping)-unsafe.Sizeof(CacheEntry[struct{}]{}),
	)
	// Arrange/Act: overwrite the oldest live entry, then force eviction.
	oldest, next := cache.oldest.key, cache.oldest.next.key
	if err := cache.Store(ctx, oldest, entry); err != nil {
		t.Fatal(err)
	}
	if err := cache.Store(ctx, "new", entry); err != nil {
		t.Fatal(err)
	}
	// Assert: overwrite promotes its key just like a hit.
	if _, ok := cache.entries[oldest]; !ok {
		t.Fatal("overwritten entry evicted")
	}
	if _, ok := cache.entries[next]; ok {
		t.Fatal("oldest entry survived eviction")
	}
	assertCacheList(t, cache)
	// Act: expire every live entry at the exact TTL boundary.
	cache.now = func() time.Time { return entry.ExpiresAt }
	for key := range cache.entries {
		if _, hit, err := cache.Load(ctx, key); err != nil || hit {
			t.Fatalf("expired hit=%v: %v", hit, err)
		}
	}
	// Assert: the list no longer roots keys/documents after expiration.
	assertCacheList(t, cache)
	if len(cache.entries) != 0 || cache.oldest != nil || cache.newest != nil {
		t.Fatal("expired entries retained")
	}
}

func TestMemoryCacheCallbacksOutsideLockAndCancellation(t *testing.T) {
	// Arrange: host callbacks inspect the mutex; they must never run while it is held.
	now := time.Unix(1, 0)
	var cache *MemoryCache[struct{}]
	unlocked := func() {
		if !cache.mu.TryLock() {
			t.Fatal("host callback invoked under cache mutex")
		}
		cache.mu.Unlock()
	}
	var initErr error
	cache, initErr = NewMemoryCache[struct{}](
		1,
		func() time.Time { unlocked(); return now },
		func(v struct{}) (struct{}, error) { unlocked(); return v, nil },
	)
	if initErr != nil {
		t.Fatal(initErr)
	}
	entry := CacheEntry[struct{}]{Documents: []Document[struct{}]{{ID: "doc"}}, ExpiresAt: now.Add(time.Hour)}
	// Act.
	if err := cache.Store(context.Background(), "a", entry); err != nil {
		t.Fatal(err)
	}
	_, loadedHit, loadErr := cache.Load(context.Background(), "a")
	// Assert.
	if loadErr != nil || !loadedHit {
		t.Fatalf("hit=%v: %v", loadedHit, loadErr)
	}
	// Arrange: cancel during the final host callback, including empty-document paths.
	ctx, cancel := context.WithCancel(context.Background())
	cache.now = func() time.Time { cancel(); return now }
	// Act/Assert: no new entry admitted and cancellation is returned.
	if err := cache.Store(
		ctx,
		"b",
		CacheEntry[struct{}]{ExpiresAt: entry.ExpiresAt},
	); !errors.Is(
		err,
		context.Canceled,
	) {
		t.Fatalf("store cancellation: %v", err)
	}
	if _, ok := cache.entries["a"]; !ok {
		t.Fatal("canceled store evicted existing entry")
	}
	ctx, cancel = context.WithCancel(context.Background())
	if _, hit, err := cache.Load(ctx, "a"); !errors.Is(err, context.Canceled) || hit {
		t.Fatalf("load cancellation hit=%v: %v", hit, err)
	}
	// Arrange: cancellation inside the last metadata clone also must stop delivery.
	cache.now = func() time.Time { return now }
	ctx, cancel = context.WithCancel(context.Background())
	cache.clone = func(v struct{}) (struct{}, error) { cancel(); return v, nil }
	// Act/Assert.
	if _, hit, err := cache.Load(ctx, "a"); !errors.Is(err, context.Canceled) || hit {
		t.Fatalf("clone cancellation hit=%v: %v", hit, err)
	}
}

func TestMemoryCacheConcurrentEvictionKeepsLoadSnapshot(t *testing.T) {
	// Arrange: pause a hit in its metadata snapshotter while another request reuses
	// its LRU node for a new key with a different expiry and metadata.
	now := time.Unix(1, 0)
	started, resume := make(chan struct{}), make(chan struct{})
	var block atomic.Bool
	cache, initErr := NewMemoryCache[map[string]string](
		1,
		func() time.Time { return now },
		func(meta map[string]string) (map[string]string, error) {
			if block.CompareAndSwap(true, false) {
				close(started)
				<-resume
			}
			return maps.Clone(meta), nil
		},
	)
	if initErr != nil {
		t.Fatal(initErr)
	}
	original := CacheEntry[map[string]string]{
		Documents: []Document[map[string]string]{{ID: "original", Meta: map[string]string{"value": "original"}}},
		ExpiresAt: now.Add(time.Hour),
	}
	if err := cache.Store(context.Background(), "a", original); err != nil {
		t.Fatal(err)
	}
	original.Documents[0].Meta["value"] = "caller mutation"
	block.Store(true)
	type result struct {
		entry CacheEntry[map[string]string]
		hit   bool
		err   error
	}
	done := make(chan result, 1)
	// Act.
	go func() { entry, hit, err := cache.Load(context.Background(), "a"); done <- result{entry, hit, err} }()
	<-started
	replacement := CacheEntry[map[string]string]{
		Documents: []Document[map[string]string]{{ID: "replacement", Meta: map[string]string{"value": "replacement"}}},
		ExpiresAt: now.Add(2 * time.Hour),
	}
	storeErr := cache.Store(context.Background(), "b", replacement)
	close(resume)
	loaded := <-done
	// Assert: node reuse cannot change the already captured document/TTL snapshot.
	if storeErr != nil {
		t.Fatal(storeErr)
	}
	if loaded.err != nil || !loaded.hit || loaded.entry.ExpiresAt != original.ExpiresAt ||
		loaded.entry.Documents[0].Meta["value"] != "original" {
		t.Fatalf("snapshot changed during eviction: %+v", loaded)
	}
	loaded.entry.Documents[0].Meta["value"] = "loaded mutation"
	next, replacementHit, replacementErr := cache.Load(context.Background(), "b")
	if replacementErr != nil || !replacementHit || next.Documents[0].Meta["value"] != "replacement" {
		t.Fatalf("replacement alias: %v, %v", replacementHit, replacementErr)
	}
}
