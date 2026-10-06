package managed

import "github.com/skosovsky/ragy/lexical"

// Configuration and cache live in one Adapter instance. A generation identifies
// its exact retained payload inventory, including changes under the same tuple.
type snapshotKey struct {
	binding    string
	predicate  string
	inventory  string
	generation uint64
}
type cachedSnapshot[TMeta any] struct {
	index *lexical.BM25Snapshot[TMeta]
	used  uint64
}

func (a *Adapter[TMeta]) invalidateCacheLocked() {
	a.generation++
	clear(a.cache)
}
func (a *Adapter[TMeta]) cached(key snapshotKey) *lexical.BM25Snapshot[TMeta] {
	a.mu.Lock()
	defer a.mu.Unlock()
	entry, exists := a.cache[key]
	if !exists || key.generation != a.generation {
		return nil
	}
	a.cacheClock++
	entry.used = a.cacheClock
	a.cache[key] = entry
	return entry.index
}

func (a *Adapter[TMeta]) cacheSnapshot(
	key snapshotKey,
	index *lexical.BM25Snapshot[TMeta],
) *lexical.BM25Snapshot[TMeta] {
	a.mu.Lock()
	defer a.mu.Unlock()
	if key.generation != a.generation {
		return index
	}
	a.cacheClock++
	if existing, exists := a.cache[key]; exists {
		existing.used = a.cacheClock
		a.cache[key] = existing
		return existing.index
	}
	if len(a.cache) >= a.config.MaxCachedSnapshots {
		var oldest snapshotKey
		var stamp uint64
		for candidate, entry := range a.cache {
			if stamp == 0 || entry.used < stamp {
				oldest, stamp = candidate, entry.used
			}
		}
		delete(a.cache, oldest)
	}
	a.cache[key] = cachedSnapshot[TMeta]{index: index, used: a.cacheClock}
	return index
}
