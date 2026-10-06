package final_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/retrieval"
)

type cacheMeta struct {
	Tenant string `json:"tenant"`
}
type cacheLoadBarrier struct {
	retrieval.ResultCache[cacheMeta]

	armed   atomic.Bool
	entered chan struct{}
	release chan struct{}
}

func (s *cacheLoadBarrier) Load(ctx context.Context, key string) (retrieval.CacheEntry[cacheMeta], bool, error) {
	entry, hit, err := s.ResultCache.Load(ctx, key)
	if s.armed.Load() && hit {
		close(s.entered)
		select {
		case <-s.release:
		case <-ctx.Done():
			return retrieval.CacheEntry[cacheMeta]{}, false, ctx.Err()
		}
	}
	return entry, hit, err
}

func TestActualBM25CacheRevocationBetweenLoadAndDeliveryClone(t *testing.T) {
	// Arrange: actual scoped BM25 plus a real memory cache; the wrapper only exposes a deterministic I/O boundary.
	fields := filter.NewSchema()
	tenant, err := fields.String("tenant")
	if err != nil {
		t.Fatal(err)
	}
	schema, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	builder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	mandatory, err := filter.Eq(builder, tenant, "a").Build()
	if err != nil {
		t.Fatal(err)
	}
	var revoked atomic.Bool
	now := time.Unix(100, 0)
	read, err := access.Scoped(access.ScopedConfig{
		Schema:      schema,
		Mandatory:   mandatory,
		Publication: access.CurrentPublication(),
		Snapshot: access.Snapshot{
			Identity:    "consumer-a",
			PolicyEpoch: 1,
			IssuedAt:    now,
			ExpiresAt:   now.Add(time.Minute),
		},
		Now: func() time.Time { return now },
		Authority: access.AuthorityFunc(func(context.Context, access.Snapshot) error {
			if revoked.Load() {
				return ragy.ErrUnavailable
			}
			return nil
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	index, err := lexical.NewBM25Index[cacheMeta](
		schema,
		lexical.Config[cacheMeta]{SearchFields: []string{"content"}},
		lexical.DefaultTokenizer{},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, doc := range []retrieval.Document[cacheMeta]{
		{ID: "allowed", Content: "needle", Meta: cacheMeta{Tenant: "a"}},
		{ID: "forbidden", Content: "needle", Meta: cacheMeta{Tenant: "b"}},
	} {
		if err = index.Upsert(doc); err != nil {
			t.Fatal(err)
		}
	}
	memory, err := retrieval.NewMemoryCache(2, time.Now, func(m cacheMeta) (cacheMeta, error) { return m, nil })
	if err != nil {
		t.Fatal(err)
	}
	barrier := &cacheLoadBarrier{ResultCache: memory, entered: make(chan struct{}), release: make(chan struct{})}
	var clones atomic.Int64
	cached, err := retrieval.NewCachedBackend(retrieval.CacheConfig[struct{}, retrieval.NoRequestMeta, cacheMeta]{
		Next: index, Store: barrier, TTL: time.Minute, Now: time.Now,
		CloneMeta: func(m cacheMeta) (cacheMeta, error) { clones.Add(1); return m, nil },
		SnapshotRequest: func(q retrieval.Query[struct{}]) (retrieval.Query[struct{}], error) {
			return retrieval.CopyRequestOptions(q), nil
		},
		Identity: func(context.Context, retrieval.Query[struct{}]) (retrieval.CacheIdentity, error) {
			return retrieval.CacheIdentity{
				Index:         "bm25",
				IndexRevision: "r1",
				Recipe:        "direct",
				Configuration: "fixture",
				Capabilities:  []string{"scope"},
			}, nil
		},
		HostIdentity: func(retrieval.Query[struct{}]) ([]byte, error) { return []byte("host"), nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	query := retrieval.Query[struct{}]{Read: read, Text: "needle", Options: retrieval.RetrieveOptions{TopK: 5}}
	warmed, err := cached.Retrieve(t.Context(), query)
	if err != nil || warmed.Len() != 1 || warmed.Documents()[0].ID != "allowed" {
		t.Fatal("warm scoped cache", err)
	}
	before := clones.Load()
	barrier.armed.Store(true)
	type outcome struct {
		result retrieval.ResultSet[cacheMeta]
		err    error
	}
	done := make(chan outcome, 1)
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	// Act: revoke after an actual cache hit has been loaded, before checkedHit can clone metadata.
	go func() { result, callErr := cached.Retrieve(ctx, query); done <- outcome{result, callErr} }()
	select {
	case <-barrier.entered:
	case <-ctx.Done():
		t.Fatal("cache barrier never reached", ctx.Err())
	}
	revoked.Store(true)
	close(barrier.release)
	result := <-done
	// Assert: revocation remains fatal, no cached payload is delivered or cloned downstream.
	if !errors.Is(result.err, ragy.ErrUnavailable) || !access.IsProtectionFailure(result.err) ||
		!result.result.IsEmpty() {
		t.Fatal("revoked cache hit delivered", result.err)
	}
	if clones.Load() != before {
		t.Fatal("delivery clone ran after revocation")
	}
}
