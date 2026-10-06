package retrieval

import (
	"context"
	"fmt"
	"sync"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
)

// MetadataCloner snapshots host-owned BYOT metadata. It must return an independent
// value and be safe for concurrent requests. No reflection-based cloning is assumed.
type MetadataCloner[TMeta any] func(TMeta) (TMeta, error)

// CacheEntry is a successful complete target result with an explicit expiry.
// Partial/failed calls are never admitted as successful cached results.
type CacheEntry[TMeta any] struct {
	Documents []Document[TMeta]
	ExpiresAt time.Time
}

// ResultCache stores owned snapshots and returns independent snapshots. External
// implementations must apply the same metadata ownership contract as MemoryCache.
type ResultCache[TMeta any] interface {
	Load(context.Context, string) (CacheEntry[TMeta], bool, error)
	Store(context.Context, string, CacheEntry[TMeta]) error
}

type memoryCacheItem[TMeta any] struct {
	entry CacheEntry[TMeta]
	used  uint64
}

// MemoryCache is a bounded in-process cache with host clock and metadata snapshots.
// It has no workers, retries or hidden refreshes; expiry is enforced on each load.
type MemoryCache[TMeta any] struct {
	mu         sync.Mutex
	entries    map[string]memoryCacheItem[TMeta]
	maxEntries int
	sequence   uint64
	now        func() time.Time
	clone      MetadataCloner[TMeta]
}

// NewMemoryCache requires explicit capacity, clock and BYOT snapshot policy.
func NewMemoryCache[TMeta any](
	maxEntries int,
	now func() time.Time,
	clone MetadataCloner[TMeta],
) (*MemoryCache[TMeta], error) {
	if maxEntries <= 0 || now == nil || clone == nil {
		return nil, fmt.Errorf("%w: cache capacity/clock/snapshotter", ragy.ErrInvalidArgument)
	}
	return &MemoryCache[TMeta]{
		mu:         sync.Mutex{},
		entries:    map[string]memoryCacheItem[TMeta]{},
		maxEntries: maxEntries,
		sequence:   0,
		now:        now,
		clone:      clone,
	}, nil
}

// Load rejects expired entries and copies their owned metadata before exposure.
func (m *MemoryCache[TMeta]) Load(ctx context.Context, key string) (CacheEntry[TMeta], bool, error) {
	if err := ctx.Err(); err != nil {
		return CacheEntry[TMeta]{}, false, err
	}
	m.mu.Lock()
	item, exists := m.entries[key]
	if exists && !m.now().Before(item.entry.ExpiresAt) {
		delete(m.entries, key)
		exists = false
	}
	if exists {
		m.sequence++
		item.used = m.sequence
		m.entries[key] = item
	}
	m.mu.Unlock()
	if !exists {
		return CacheEntry[TMeta]{}, false, nil
	}
	docs, err := cloneCacheDocuments(ctx, item.entry.Documents, m.clone)
	if err != nil {
		return CacheEntry[TMeta]{}, false, err
	}
	return CacheEntry[TMeta]{Documents: docs, ExpiresAt: item.entry.ExpiresAt}, true, nil
}

// Store snapshots outside the cache lock and evicts the least recently used entry.
func (m *MemoryCache[TMeta]) Store(ctx context.Context, key string, entry CacheEntry[TMeta]) error {
	if key == "" || entry.ExpiresAt.IsZero() {
		return fmt.Errorf("%w: cache key/expiry", ragy.ErrInvalidArgument)
	}
	docs, err := cloneCacheDocuments(ctx, entry.Documents, m.clone)
	if err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.now().Before(entry.ExpiresAt) {
		return nil
	}
	if _, exists := m.entries[key]; !exists && len(m.entries) >= m.maxEntries {
		oldestKey := ""
		oldest := ^uint64(0)
		for existing, item := range m.entries {
			if item.used < oldest {
				oldest = item.used
				oldestKey = existing
			}
		}
		delete(m.entries, oldestKey)
	}
	m.sequence++
	m.entries[key] = memoryCacheItem[TMeta]{
		entry: CacheEntry[TMeta]{Documents: docs, ExpiresAt: entry.ExpiresAt},
		used:  m.sequence,
	}
	return nil
}

func cloneCacheDocuments[TMeta any](
	ctx context.Context,
	docs []Document[TMeta],
	clone MetadataCloner[TMeta],
) ([]Document[TMeta], error) {
	out := copyDocuments(docs)
	for i := range out {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := ValidateDocument(out[i]); err != nil {
			return nil, ragy.WrapProjectionError(err, "cache snapshot validate")
		}
		meta, err := clone(out[i].Meta)
		if err != nil {
			return nil, ragy.WrapProjectionError(err, "cache metadata snapshot")
		}
		out[i].Meta = meta
	}
	return out, nil
}

// CacheConfig declares the complete host cache identity and metadata ownership.
// Identity and HostIdentity are evaluated for every request, before any cache hit.
type CacheConfig[TIntent, TRequestMeta, TMeta any] struct {
	SnapshotRequest func(Request[TIntent, TRequestMeta]) (Request[TIntent, TRequestMeta], error)
	Next            RequestBackend[TIntent, TRequestMeta, TMeta]
	Store           ResultCache[TMeta]
	TTL             time.Duration
	Now             func() time.Time
	Identity        func(context.Context, Request[TIntent, TRequestMeta]) (CacheIdentity, error)
	HostIdentity    func(Request[TIntent, TRequestMeta]) ([]byte, error)
	CloneMeta       MetadataCloner[TMeta]
	Resolver        IdentityResolver[TMeta]
}

// CachedBackend wraps a target, preserving its admission guarantees. It caches only
// successful complete results and always checks host freshness before delivery.
type CachedBackend[TIntent, TRequestMeta, TMeta any] struct {
	config CacheConfig[TIntent, TRequestMeta, TMeta]
}

// NewCachedBackend rejects an implicit identity, unbounded TTL or missing snapshots.
func NewCachedBackend[TIntent, TRequestMeta, TMeta any](
	config CacheConfig[TIntent, TRequestMeta, TMeta],
) (*CachedBackend[TIntent, TRequestMeta, TMeta], error) {
	if config.SnapshotRequest == nil || isNilReadTarget(config.Next) || isNilReadTarget(config.Store) ||
		config.TTL <= 0 ||
		config.Now == nil ||
		config.Identity == nil ||
		config.HostIdentity == nil ||
		config.CloneMeta == nil {
		return nil, fmt.Errorf("%w: cache configuration", ragy.ErrInvalidArgument)
	}
	config.Resolver = DefaultResolver(config.Resolver)
	return &CachedBackend[TIntent, TRequestMeta, TMeta]{config: config}, nil
}

// Schema forwards the actual target schema; an undeclared target cannot gain scope
// support by obtaining a previously cached result.
func (b *CachedBackend[TIntent, TRequestMeta, TMeta]) Schema() filter.Schema {
	if provider, ok := b.config.Next.(ReadCapabilityProvider); ok {
		return provider.Schema()
	}
	return filter.Schema{}
}

// ReadCapabilities forwards only guarantees declared by the underlying target.
func (b *CachedBackend[TIntent, TRequestMeta, TMeta]) ReadCapabilities() access.Capabilities {
	if provider, ok := b.config.Next.(ReadCapabilityProvider); ok {
		return provider.ReadCapabilities()
	}
	return access.Capabilities{RequirePinnedPublication: false, ScopeProfile: false, PinnedPublication: false}
}

// Retrieve gates both hits and misses and never reuses data across binding identities.
func (b *CachedBackend[TIntent, TRequestMeta, TMeta]) Retrieve(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (ResultSet[TMeta], error) {
	if err := req.Read.Check(ctx); err != nil {
		return NewResultSet[TMeta](nil, b.config.Resolver), err
	}
	captured, snapshotErr := b.config.SnapshotRequest(req)
	if snapshotErr != nil {
		return NewResultSet[TMeta](nil, b.config.Resolver), snapshotErr
	}
	captured.Read = req.Read
	captured = CopyRequestOptions(captured)
	rs, err := b.retrieve(ctx, captured)
	return DeliverRead(ctx, req.Read, rs, err, b.config.Resolver)
}

func (b *CachedBackend[TIntent, TRequestMeta, TMeta]) retrieve(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (ResultSet[TMeta], error) {
	empty := NewResultSet[TMeta](nil, b.config.Resolver)
	if err := admitBackendRead(ctx, req, b.config.Next); err != nil {
		return empty, err
	}
	identity, err := b.config.Identity(ctx, req)
	if err != nil {
		return empty, err
	}
	host, err := b.config.HostIdentity(req)
	if err != nil {
		return empty, err
	}
	key, err := RequestCacheKey(req, identity, host)
	if err != nil {
		return empty, err
	}
	if gateErr := req.Read.Check(ctx); gateErr != nil {
		return empty, gateErr
	}
	entry, hit, err := b.config.Store.Load(ctx, key)
	if err != nil {
		return empty, err
	}
	if hit {
		cached, usable, hitErr := b.checkedHit(ctx, req, key, host, entry)
		if hitErr != nil || usable {
			return cached, hitErr
		}
	}
	if gateErr := req.Read.Check(ctx); gateErr != nil {
		return empty, gateErr
	}
	rs, err := b.config.Next.Retrieve(ctx, req)
	rs, err = DeliverRead(ctx, req.Read, rs, err, b.config.Resolver)
	if err != nil {
		return rs, err
	}
	rs = ensureResultSet(rs, b.config.Resolver)
	docs, err := cloneCacheDocuments(ctx, rs.Documents(), b.config.CloneMeta)
	if err != nil {
		return empty, err
	}
	currentIdentity, identityErr := b.config.Identity(ctx, req)
	if identityErr != nil {
		return NewResultSet(docs, b.config.Resolver), identityErr
	}
	currentKey, keyErr := RequestCacheKey(req, currentIdentity, host)
	if keyErr != nil {
		return NewResultSet(docs, b.config.Resolver), keyErr
	}
	if currentKey != key {
		return NewResultSet(docs, b.config.Resolver), nil
	}
	entry = CacheEntry[TMeta]{Documents: docs, ExpiresAt: b.config.Now().Add(b.config.TTL)}
	if gateErr := req.Read.Check(ctx); gateErr != nil {
		return empty, gateErr
	}
	if storeErr := b.config.Store.Store(ctx, key, entry); storeErr != nil {
		return NewResultSet(docs, b.config.Resolver), storeErr
	}
	out, err := cloneCacheDocuments(ctx, docs, b.config.CloneMeta)
	if err != nil {
		return empty, err
	}
	return NewResultSet(out, b.config.Resolver), nil
}

func (b *CachedBackend[TIntent, TRequestMeta, TMeta]) checkedHit(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
	key string,
	host []byte,
	entry CacheEntry[TMeta],
) (ResultSet[TMeta], bool, error) {
	empty := NewResultSet[TMeta](nil, b.config.Resolver)
	if !b.config.Now().Before(entry.ExpiresAt) {
		return empty, false, nil
	}
	if gateErr := req.Read.Check(ctx); gateErr != nil {
		return empty, false, gateErr
	}
	identity, err := b.config.Identity(ctx, req)
	if err != nil {
		return empty, false, err
	}
	currentKey, err := RequestCacheKey(req, identity, host)
	if err != nil {
		return empty, false, err
	}
	if currentKey != key {
		return empty, false, nil
	}
	docs, err := cloneCacheDocuments(ctx, entry.Documents, b.config.CloneMeta)
	if err != nil {
		return empty, false, err
	}
	return NewResultSet(docs, b.config.Resolver), true, nil
}
