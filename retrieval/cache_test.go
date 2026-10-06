package retrieval_test

import (
	"context"
	"errors"
	"maps"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

type cacheClock struct {
	mu      sync.Mutex
	instant time.Time
}

func (c *cacheClock) now() time.Time { c.mu.Lock(); defer c.mu.Unlock(); return c.instant }
func (c *cacheClock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.instant = c.instant.Add(d)
}

type cacheMeta map[string]string

func cloneCacheMeta(meta cacheMeta) (cacheMeta, error) { return maps.Clone(meta), nil }

type cacheBackend struct {
	schema    filter.Schema
	documents []retrieval.Document[cacheMeta]
	calls     atomic.Int64
	after     func()
	err       error
}

func (b *cacheBackend) Schema() filter.Schema { return b.schema }
func (*cacheBackend) ReadCapabilities() access.Capabilities {
	return access.Capabilities{RequirePinnedPublication: false, ScopeProfile: true}
}

func (b *cacheBackend) Retrieve(
	ctx context.Context,
	req retrieval.Query[struct{}],
) (retrieval.ResultSet[cacheMeta], error) {
	prepared, err := retrieval.PrepareRead(ctx, req, b)
	if err != nil {
		return retrieval.NewResultSet[cacheMeta](nil, nil), err
	}
	b.calls.Add(1)
	var docs []retrieval.Document[cacheMeta]
	for _, doc := range b.documents {
		match, matchErr := filter.MatchCondition(
			prepared.Options.Filters,
			func(field string) (any, bool) { value, exists := doc.Meta[field]; return value, exists },
		)
		if matchErr != nil {
			return retrieval.NewResultSet[cacheMeta](nil, nil), matchErr
		}
		if match {
			docs = append(docs, doc)
		}
	}
	if b.after != nil {
		b.after()
	}
	return retrieval.NewResultSet(docs, nil), b.err
}

type cacheStoreSpy struct {
	entry  retrieval.CacheEntry[cacheMeta]
	hit    bool
	loads  int
	stores int
	after  func()
}

func (s *cacheStoreSpy) Load(_ context.Context, _ string) (retrieval.CacheEntry[cacheMeta], bool, error) {
	s.loads++
	if s.after != nil {
		s.after()
	}
	return s.entry, s.hit, nil
}

func (s *cacheStoreSpy) Store(_ context.Context, _ string, _ retrieval.CacheEntry[cacheMeta]) error {
	s.stores++
	return nil
}

func cachedWithSpy(
	t *testing.T,
	backend retrieval.Backend[struct{}, cacheMeta],
	store *cacheStoreSpy,
	clock *cacheClock,
	identity func(context.Context, retrieval.Query[struct{}]) (retrieval.CacheIdentity, error),
) *retrieval.CachedBackend[struct{}, retrieval.NoRequestMeta, cacheMeta] {
	t.Helper()
	if identity == nil {
		identity = func(context.Context, retrieval.Query[struct{}]) (retrieval.CacheIdentity, error) {
			return cacheIdentity(), nil
		}
	}
	cached, err := retrieval.NewCachedBackend(retrieval.CacheConfig[struct{}, retrieval.NoRequestMeta, cacheMeta]{
		SnapshotRequest: func(req retrieval.Query[struct{}]) (retrieval.Query[struct{}], error) { return req, nil },
		Next:            backend, Store: store, TTL: 30 * time.Second, Now: clock.now,
		Identity:     identity,
		HostIdentity: func(retrieval.Query[struct{}]) ([]byte, error) { return []byte("host"), nil },
		CloneMeta:    cloneCacheMeta,
	})
	if err != nil {
		t.Fatal(err)
	}
	return cached
}

type cacheOpaqueBackend struct{ next *cacheBackend }

func (b cacheOpaqueBackend) Retrieve(
	ctx context.Context,
	req retrieval.Query[struct{}],
) (retrieval.ResultSet[cacheMeta], error) {
	return b.next.Retrieve(ctx, req)
}

func TestCacheCannotBypassUnsupportedTargetAdmission(t *testing.T) {
	// Arrange: cached data is available but the target declares no scope capability.
	clock := &cacheClock{instant: time.Unix(100, 0)}
	var epoch atomic.Int64
	epoch.Store(7)
	schema := cacheSchema(t)
	backend := &cacheBackend{schema: schema}
	store := &cacheStoreSpy{
		entry: retrieval.CacheEntry[cacheMeta]{
			Documents: []retrieval.Document[cacheMeta]{{ID: "cached-secret"}},
			ExpiresAt: clock.now().Add(30 * time.Second),
		},
		hit: true,
	}
	cached := cachedWithSpy(t, cacheOpaqueBackend{next: backend}, store, clock, nil)
	// Act.
	rs, err := cached.Retrieve(context.Background(), retrieval.Query[struct{}]{
		Read: cacheRead(t, schema, "a", 7, clock, &epoch), Options: retrieval.RetrieveOptions{TopK: 1},
	})
	// Assert: reject before any cache or target I/O, despite the populated cache.
	if !errors.Is(err, ragy.ErrUnsupported) || !access.IsProtectionFailure(err) || !rs.IsEmpty() || store.loads != 0 ||
		backend.calls.Load() != 0 {
		t.Fatalf("cache bypassed target admission: %v, loads=%d, calls=%d", err, store.loads, backend.calls.Load())
	}
}

func TestCacheRevocationDuringLoadPreventsHitAndRefresh(t *testing.T) {
	// Arrange: a cache load revokes an otherwise valid binding before returning payload.
	clock := &cacheClock{instant: time.Unix(100, 0)}
	var epoch atomic.Int64
	epoch.Store(7)
	schema := cacheSchema(t)
	backend := &cacheBackend{schema: schema}
	store := &cacheStoreSpy{
		entry: retrieval.CacheEntry[cacheMeta]{
			Documents: []retrieval.Document[cacheMeta]{{ID: "cached-secret", Meta: cacheMeta{"tenant": "a"}}},
			ExpiresAt: clock.now().Add(30 * time.Second),
		},
		hit: true, after: func() { epoch.Store(8) },
	}
	cached := cachedWithSpy(t, backend, store, clock, nil)
	req := retrieval.Query[struct{}]{
		Read:    cacheRead(t, schema, "a", 7, clock, &epoch),
		Options: retrieval.RetrieveOptions{TopK: 1},
	}
	// Act.
	rs, err := cached.Retrieve(context.Background(), req)
	// Assert.
	if !access.IsProtectionFailure(err) || !rs.IsEmpty() || store.loads != 1 || store.stores != 0 ||
		backend.calls.Load() != 0 {
		t.Fatalf(
			"revocation during cache I/O escaped: %v, loads=%d, stores=%d, calls=%d",
			err,
			store.loads,
			store.stores,
			backend.calls.Load(),
		)
	}
}

func TestCacheRevisionChangeDuringLoadRejectsOldHit(t *testing.T) {
	// Arrange: the old cache entry is loaded across a live index identity change.
	clock := &cacheClock{instant: time.Unix(100, 0)}
	backend := &cacheBackend{schema: cacheSchema(t), documents: []retrieval.Document[cacheMeta]{{ID: "new"}}}
	revision := "index1"
	store := &cacheStoreSpy{
		entry: retrieval.CacheEntry[cacheMeta]{
			Documents: []retrieval.Document[cacheMeta]{{ID: "old"}},
			ExpiresAt: clock.now().Add(30 * time.Second),
		},
		hit: true, after: func() { revision = "index2" },
	}
	cached := cachedWithSpy(
		t,
		backend,
		store,
		clock,
		func(context.Context, retrieval.Query[struct{}]) (retrieval.CacheIdentity, error) {
			identity := cacheIdentity()
			identity.IndexRevision = revision
			return identity, nil
		},
	)
	// Act.
	rs, err := cached.Retrieve(
		context.Background(),
		retrieval.Query[struct{}]{Read: retrieval.UnrestrictedRead(), Options: retrieval.RetrieveOptions{TopK: 1}},
	)
	// Assert: fetch fresh data once, never populate the old revision key.
	if err != nil || rs.Len() != 1 || rs.Documents()[0].ID != "new" || backend.calls.Load() != 1 || store.stores != 0 {
		t.Fatalf(
			"old revision hit escaped: %v, %v, calls=%d, stores=%d",
			rs.Documents(),
			err,
			backend.calls.Load(),
			store.stores,
		)
	}
}

func TestCacheNeverStoresPartialFailure(t *testing.T) {
	// Arrange: the target returns both a useful partial result and an ordinary error.
	clock := &cacheClock{instant: time.Unix(100, 0)}
	backend := &cacheBackend{
		schema: cacheSchema(t), documents: []retrieval.Document[cacheMeta]{{ID: "partial"}}, err: ragy.ErrUnavailable,
	}
	store := &cacheStoreSpy{}
	cached := cachedWithSpy(t, backend, store, clock, nil)
	// Act.
	for range 2 {
		rs, err := cached.Retrieve(
			context.Background(),
			retrieval.Query[struct{}]{Read: retrieval.UnrestrictedRead(), Options: retrieval.RetrieveOptions{TopK: 1}},
		)
		// Assert: retain authorized partial output, but every new call must reach the target.
		if !errors.Is(err, ragy.ErrUnavailable) || rs.Len() != 1 {
			t.Fatalf("partial failure changed: %v, %v", rs.Documents(), err)
		}
	}
	if backend.calls.Load() != 2 || store.stores != 0 {
		t.Fatalf("partial cached or retried: calls=%d, stores=%d", backend.calls.Load(), store.stores)
	}
}

func cacheSchema(t *testing.T) filter.Schema {
	t.Helper()
	builder := filter.NewSchema()
	if _, err := builder.String("tenant"); err != nil {
		t.Fatal(err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatal(err)
	}
	return schema
}

func cacheRead(
	t *testing.T,
	schema filter.Schema,
	tenant string,
	epoch int64,
	clock *cacheClock,
	current *atomic.Int64,
) access.Binding {
	t.Helper()
	field, err := schema.StringField("tenant")
	if err != nil {
		t.Fatal(err)
	}
	builder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	mandatory, err := filter.Eq(builder, field, tenant).Build()
	if err != nil {
		t.Fatal(err)
	}
	now := clock.now()
	binding, err := access.Scoped(
		access.ScopedConfig{
			Snapshot: access.Snapshot{
				Identity:    "same-host-snapshot-id",
				PolicyEpoch: epoch,
				IssuedAt:    now,
				ExpiresAt:   now.Add(30 * time.Second),
			},
			Mandatory:   mandatory,
			Schema:      schema,
			Publication: access.CurrentPublication(),
			Now:         clock.now,
			Authority: access.AuthorityFunc(func(_ context.Context, snapshot access.Snapshot) error {
				if snapshot.PolicyEpoch != current.Load() {
					return ragy.ErrUnavailable
				}
				return nil
			}),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return binding
}
func cacheIdentity() retrieval.CacheIdentity {
	return retrieval.CacheIdentity{
		Index:         "fixture-index",
		IndexRevision: "index1",
		Recipe:        "baseline",
		Configuration: "config1",
		Capabilities:  []string{"scope", "lexical"},
	}
}

func newCachedFixture(
	t *testing.T,
	backend *cacheBackend,
	clock *cacheClock,
	identity func(context.Context, retrieval.Query[struct{}]) (retrieval.CacheIdentity, error),
) *retrieval.CachedBackend[struct{}, retrieval.NoRequestMeta, cacheMeta] {
	t.Helper()
	store, err := retrieval.NewMemoryCache[cacheMeta](8, clock.now, cloneCacheMeta)
	if err != nil {
		t.Fatal(err)
	}
	if identity == nil {
		identity = func(context.Context, retrieval.Query[struct{}]) (retrieval.CacheIdentity, error) {
			return cacheIdentity(), nil
		}
	}
	cached, err := retrieval.NewCachedBackend(
		retrieval.CacheConfig[struct{}, retrieval.NoRequestMeta, cacheMeta]{
			Next:            backend,
			Store:           store,
			TTL:             30 * time.Second,
			Now:             clock.now,
			Identity:        identity,
			HostIdentity:    func(retrieval.Query[struct{}]) ([]byte, error) { return []byte("no-host-metadata"), nil },
			SnapshotRequest: func(req retrieval.Query[struct{}]) (retrieval.Query[struct{}], error) { return req, nil },
			CloneMeta:       cloneCacheMeta,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return cached
}

func TestCacheIsolatesTenantsAndPolicyEpochBeforeTTL(t *testing.T) {
	// Arrange: identical query and host snapshot ID, distinct mandatory tenants.
	clock := &cacheClock{instant: time.Unix(100, 0)}
	var epoch atomic.Int64
	epoch.Store(7)
	schema := cacheSchema(t)
	backend := &cacheBackend{
		schema: schema,
		documents: []retrieval.Document[cacheMeta]{
			{ID: "a-public", Content: "policy", Meta: cacheMeta{"tenant": "a"}},
			{ID: "b-public", Content: "policy", Meta: cacheMeta{"tenant": "b"}},
		},
	}
	cached := newCachedFixture(t, backend, clock, nil)
	readA := cacheRead(t, schema, "a", 7, clock, &epoch)
	readB := cacheRead(t, schema, "b", 7, clock, &epoch)
	for _, tc := range []struct {
		read access.Binding
		want string
	}{{readA, "a-public"}, {readB, "b-public"}, {readA, "a-public"}} {
		// Act.
		rs, err := cached.Retrieve(
			context.Background(),
			retrieval.Query[struct{}]{Read: tc.read, Text: "policy", Options: retrieval.RetrieveOptions{TopK: 10}},
		)
		// Assert.
		if err != nil || rs.Len() != 1 || rs.Documents()[0].ID != tc.want {
			t.Fatalf("cross-tenant cache reuse: %v, %v", rs.Documents(), err)
		}
	}
	if backend.calls.Load() != 2 {
		t.Fatalf("cache did not reuse authorized tenant result: calls=%d", backend.calls.Load())
	}
	// Act: revoke epoch 7 at t10, before the 30-second cache TTL.
	clock.advance(10 * time.Second)
	epoch.Store(8)
	rs, err := cached.Retrieve(
		context.Background(),
		retrieval.Query[struct{}]{Read: readA, Text: "policy", Options: retrieval.RetrieveOptions{TopK: 10}},
	)
	// Assert: no hit delivery and no backend refresh using revoked authorization.
	if !access.IsProtectionFailure(err) || !rs.IsEmpty() || backend.calls.Load() != 2 {
		t.Fatalf("revoked cached result delivered/refreshed: %v, calls=%d", err, backend.calls.Load())
	}
	read8 := cacheRead(t, schema, "a", 8, clock, &epoch)
	rs, err = cached.Retrieve(
		context.Background(),
		retrieval.Query[struct{}]{Read: read8, Text: "policy", Options: retrieval.RetrieveOptions{TopK: 10}},
	)
	if err != nil || rs.Len() != 1 || backend.calls.Load() != 3 {
		t.Fatalf("new epoch reused old cache: %v, calls=%d", err, backend.calls.Load())
	}
}

func TestCacheExpiryRequiresNewHostBinding(t *testing.T) {
	// Arrange.
	clock := &cacheClock{instant: time.Unix(100, 0)}
	var epoch atomic.Int64
	epoch.Store(7)
	schema := cacheSchema(t)
	backend := &cacheBackend{
		schema:    schema,
		documents: []retrieval.Document[cacheMeta]{{ID: "a-public", Meta: cacheMeta{"tenant": "a"}}},
	}
	cached := newCachedFixture(t, backend, clock, nil)
	oldRead := cacheRead(t, schema, "a", 7, clock, &epoch)
	req := retrieval.Query[struct{}]{Read: oldRead, Text: "policy", Options: retrieval.RetrieveOptions{TopK: 1}}
	if _, err := cached.Retrieve(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	// Act.
	clock.advance(31 * time.Second)
	rs, err := cached.Retrieve(context.Background(), req)
	// Assert: neither cached data nor a fresh backend call is authorized by expired scope.
	if !access.IsProtectionFailure(err) || !rs.IsEmpty() || backend.calls.Load() != 1 {
		t.Fatalf("expired authorization reused: %v", err)
	}
	// Act: the host explicitly validates and creates a new binding for a new attempt.
	req.Read = cacheRead(t, schema, "a", 7, clock, &epoch)
	rs, err = cached.Retrieve(context.Background(), req)
	// Assert.
	if err != nil || rs.Len() != 1 || backend.calls.Load() != 2 {
		t.Fatalf("new host decision failed to refresh expired cache: %v", err)
	}
}

func TestCacheSnapshotsMutableBYOTMetadata(t *testing.T) {
	// Arrange.
	clock := &cacheClock{instant: time.Unix(100, 0)}
	schema := cacheSchema(t)
	source := cacheMeta{"tenant": "a", "value": "original"}
	backend := &cacheBackend{schema: schema, documents: []retrieval.Document[cacheMeta]{{ID: "a", Meta: source}}}
	cached := newCachedFixture(t, backend, clock, nil)
	req := retrieval.Query[struct{}]{
		Read:    retrieval.UnrestrictedRead(),
		Text:    "policy",
		Options: retrieval.RetrieveOptions{TopK: 1},
	}
	// Act.
	first, err := cached.Retrieve(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	first.Documents()[0].Meta["value"] = "caller-mutated"
	source["value"] = "backend-mutated"
	second, err := cached.Retrieve(context.Background(), req)
	// Assert.
	if err != nil || second.Documents()[0].Meta["value"] != "original" || backend.calls.Load() != 1 {
		t.Fatalf("cache aliases mutable metadata: %v, %v", second.Documents(), err)
	}
}

func TestIndexChangeDuringMissDoesNotPopulateOldRevisionKey(t *testing.T) {
	// Arrange.
	clock := &cacheClock{instant: time.Unix(100, 0)}
	schema := cacheSchema(t)
	revision := "index1"
	backend := &cacheBackend{
		schema:    schema,
		documents: []retrieval.Document[cacheMeta]{{ID: "a", Meta: cacheMeta{"tenant": "a"}}},
		after:     func() { revision = "index2" },
	}
	cached := newCachedFixture(
		t,
		backend,
		clock,
		func(context.Context, retrieval.Query[struct{}]) (retrieval.CacheIdentity, error) {
			identity := cacheIdentity()
			identity.IndexRevision = revision
			return identity, nil
		},
	)
	req := retrieval.Query[struct{}]{
		Read:    retrieval.UnrestrictedRead(),
		Text:    "policy",
		Options: retrieval.RetrieveOptions{TopK: 1},
	}
	// Act.
	if _, err := cached.Retrieve(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	revision = "index1"
	backend.after = nil
	if _, err := cached.Retrieve(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	// Assert.
	if backend.calls.Load() != 2 {
		t.Fatal("a changing live revision was cached under its old identity")
	}
}

func TestRequestCacheKeyPartitionsCoreAndHostProfiles(t *testing.T) {
	// Arrange.
	req := retrieval.Query[struct{}]{
		Read:    retrieval.UnrestrictedRead(),
		Text:    "raw policy",
		Plan:    &retrieval.PlannedQuery[struct{}]{Text: "policy"},
		Options: retrieval.RetrieveOptions{TopK: 10, FetchLimit: 100, Vector: []float32{1, 0}},
	}
	identity := cacheIdentity()
	base, err := retrieval.RequestCacheKey(req, identity, []byte("host1"))
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name   string
		change func(*retrieval.Query[struct{}], *retrieval.CacheIdentity)
		host   string
	}{
		{
			name:   "topk",
			change: func(r *retrieval.Query[struct{}], _ *retrieval.CacheIdentity) { r.Options.TopK = 9 },
			host:   "host1",
		},
		{
			name:   "fetch",
			change: func(r *retrieval.Query[struct{}], _ *retrieval.CacheIdentity) { r.Options.FetchLimit = 101 },
			host:   "host1",
		},
		{
			name:   "vector",
			change: func(r *retrieval.Query[struct{}], _ *retrieval.CacheIdentity) { r.Options.Vector[0] = 0 },
			host:   "host1",
		},
		{name: "threshold", change: func(r *retrieval.Query[struct{}], _ *retrieval.CacheIdentity) {
			r.Options.Threshold = &retrieval.ScoreThreshold{
				Value:     -1,
				State:     retrieval.ScorePresent,
				Semantics: "maxsim",
			}
		}, host: "host1"},
		{
			name:   "expanded",
			change: func(r *retrieval.Query[struct{}], _ *retrieval.CacheIdentity) { r.Plan.ExpandedText = "different" },
			host:   "host1",
		},
		{
			name:   "index",
			change: func(_ *retrieval.Query[struct{}], i *retrieval.CacheIdentity) { i.IndexRevision = "index2" },
			host:   "host1",
		},
		{
			name:   "recipe",
			change: func(_ *retrieval.Query[struct{}], i *retrieval.CacheIdentity) { i.Recipe = "multiquery" },
			host:   "host1",
		},
		{
			name:   "config",
			change: func(_ *retrieval.Query[struct{}], i *retrieval.CacheIdentity) { i.Configuration = "config2" },
			host:   "host1",
		},
		{
			name:   "caps",
			change: func(_ *retrieval.Query[struct{}], i *retrieval.CacheIdentity) { i.Capabilities = []string{"tensor"} },
			host:   "host1",
		},
		{name: "host-intent", change: func(*retrieval.Query[struct{}], *retrieval.CacheIdentity) {}, host: "host2"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			copyReq := retrieval.CopyRequestOptions(req)
			copyIdentity := identity
			tc.change(&copyReq, &copyIdentity)
			// Act.
			key, keyErr := retrieval.RequestCacheKey(copyReq, copyIdentity, []byte(tc.host))
			// Assert.
			if keyErr != nil || key == base {
				t.Fatalf("profile collision: %s, %v", tc.name, keyErr)
			}
		})
	}
	if strings.Contains(base, req.Text) || len(base) != 64 {
		t.Fatal("cache key exposes query or is not a digest")
	}
}

func TestCacheConstructorRejectsMissingOwnership(t *testing.T) {
	// Arrange/Act.
	_, err := retrieval.NewCachedBackend(retrieval.CacheConfig[struct{}, retrieval.NoRequestMeta, cacheMeta]{})
	// Assert.
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("implicit unsafe cache configuration accepted: %v", err)
	}
}

func TestCacheKeyIncludesPinnedPublicationInventory(t *testing.T) {
	// Arrange.
	target := access.TargetRevision{
		Target:            "dense",
		Namespace:         "n",
		Source:            "source",
		Revision:          "r1",
		Transformation:    "transform1",
		AccessFingerprint: "access1",
	}
	publication, err := access.PinPublication("pub1", []access.TargetRevision{target})
	if err != nil {
		t.Fatal(err)
	}
	binding, err := access.UnrestrictedAt(publication)
	if err != nil {
		t.Fatal(err)
	}
	req := retrieval.Query[struct{}]{Read: binding, Text: "policy", Options: retrieval.RetrieveOptions{TopK: 1}}
	base, err := retrieval.RequestCacheKey(req, cacheIdentity(), nil)
	if err != nil {
		t.Fatal(err)
	}
	// Act: even if a host incorrectly reuses the publication label, differing inventory cannot collide.
	target.Revision = "r2"
	changed, err := access.PinPublication("pub1", []access.TargetRevision{target})
	if err != nil {
		t.Fatal(err)
	}
	req.Read, err = access.UnrestrictedAt(changed)
	if err != nil {
		t.Fatal(err)
	}
	key, err := retrieval.RequestCacheKey(req, cacheIdentity(), nil)
	// Assert.
	if err != nil || key == base {
		t.Fatalf("publication inventory omitted from key: %v", err)
	}
}

func TestCopyRequestOptionsOwnsNestedCorePointers(t *testing.T) {
	// Arrange.
	number := 1.0
	instant := time.Unix(100, 0)
	req := retrieval.Query[struct{}]{
		Read: retrieval.UnrestrictedRead(),
		Options: retrieval.RetrieveOptions{
			TopK:      1,
			Vector:    []float32{1},
			Threshold: &retrieval.ScoreThreshold{Value: 1},
			Graph:     &retrieval.GraphOptions{Seeds: []string{"a"}, Page: &ragy.Page{Limit: 1}},
		},
		Plan: &retrieval.PlannedQuery[struct{}]{
			Ranges: []retrieval.RangeConstraint{
				{Field: "range", Start: &retrieval.RangeBound{Number: &number, Time: &instant}},
			},
		},
	}
	// Act.
	snapshot := retrieval.CopyRequestOptions(req)
	req.Options.Vector[0] = 2
	req.Options.Threshold.Value = 2
	req.Options.Graph.Seeds[0] = "b"
	req.Options.Graph.Page.Limit = 2
	number = 2
	instant = instant.Add(time.Hour)
	// Assert.
	if snapshot.Options.Vector[0] != 1 || snapshot.Options.Threshold.Value != 1 ||
		snapshot.Options.Graph.Seeds[0] != "a" ||
		snapshot.Options.Graph.Page.Limit != 1 ||
		*snapshot.Plan.Ranges[0].Start.Number != 1 ||
		!snapshot.Plan.Ranges[0].Start.Time.Equal(time.Unix(100, 0)) {
		t.Fatal("query snapshot aliases original core options")
	}
}

func TestMemoryCacheCapacityExpiryAndConcurrentOwnership(t *testing.T) {
	// Arrange.
	clock := &cacheClock{instant: time.Unix(100, 0)}
	store, err := retrieval.NewMemoryCache[cacheMeta](2, clock.now, cloneCacheMeta)
	if err != nil {
		t.Fatal(err)
	}
	entry := retrieval.CacheEntry[cacheMeta]{
		Documents: []retrieval.Document[cacheMeta]{{ID: "a", Meta: cacheMeta{"value": "original"}}},
		ExpiresAt: clock.now().Add(30 * time.Second),
	}
	if err = store.Store(context.Background(), "a", entry); err != nil {
		t.Fatal(err)
	}
	if err = store.Store(context.Background(), "b", entry); err != nil {
		t.Fatal(err)
	}
	if _, _, err = store.Load(context.Background(), "a"); err != nil {
		t.Fatal(err)
	}
	if err = store.Store(context.Background(), "c", entry); err != nil {
		t.Fatal(err)
	}
	// Act/Assert: only the least recently used entry is evicted.
	if _, hit, loadErr := store.Load(context.Background(), "b"); loadErr != nil || hit {
		t.Fatalf("bounded cache failed eviction: %v", loadErr)
	}
	var workers sync.WaitGroup
	for range 20 {
		workers.Go(func() {
			loaded, hit, loadErr := store.Load(context.Background(), "a")
			if loadErr != nil || !hit {
				t.Errorf("concurrent load: %v, %v", hit, loadErr)
				return
			}
			loaded.Documents[0].Meta["value"] = "caller-mutated"
		})
	}
	workers.Wait()
	owned, hit, err := store.Load(context.Background(), "a")
	if err != nil || !hit || owned.Documents[0].Meta["value"] != "original" {
		t.Fatalf("cache snapshots are not independent: %v", err)
	}
	// Act/Assert: TTL is applied at the exact boundary.
	clock.advance(30 * time.Second)
	if _, hit, loadErr := store.Load(context.Background(), "a"); loadErr != nil || hit {
		t.Fatalf("expired cache entry reused: %v", loadErr)
	}
}
