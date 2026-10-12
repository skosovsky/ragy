package retrieval

import (
	"context"
	"fmt"
	"strconv"
	"sync/atomic"
	"testing"
	"time"
)

func cachePerformanceFixture(tb testing.TB, capacity int) (*MemoryCache[struct{}], CacheEntry[struct{}], []string) {
	tb.Helper()
	now := time.Unix(1, 0)
	cache, initErr := NewMemoryCache[struct{}](
		capacity,
		func() time.Time { return now },
		func(v struct{}) (struct{}, error) { return v, nil },
	)
	if initErr != nil {
		tb.Fatal(initErr)
	}
	entry := CacheEntry[struct{}]{ExpiresAt: now.Add(time.Hour)}
	keys := make([]string, capacity+1)
	for i := range keys {
		keys[i] = strconv.Itoa(capacity + i)
	}
	for i := range capacity {
		if err := cache.Store(context.Background(), strconv.Itoa(i), entry); err != nil {
			tb.Fatal(err)
		}
	}
	return cache, entry, keys
}

// BenchmarkAuditCacheEvict preserves the original empty-entry audit fixture.
func BenchmarkAuditCacheEvict(b *testing.B) {
	for _, n := range []int{64, 512, 4096, 16384} {
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			cache, entry, keys := cachePerformanceFixture(b, n)
			ctx := context.Background()
			b.ReportAllocs()
			b.ResetTimer()
			for i := range b.N {
				if err := cache.Store(ctx, keys[i%len(keys)], entry); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

//nolint:gocognit,nestif // Keep the timed operations explicit so benchmark workload and overhead remain comparable.
func BenchmarkMemoryCacheOperations(b *testing.B) {
	for _, n := range []int{64, 512, 4096, 16384} {
		for _, op := range []string{"overwrite", "hit", "miss", "parallel-mixed"} {
			b.Run(fmt.Sprintf("%s/%d", op, n), func(b *testing.B) {
				cache, entry, keys := cachePerformanceFixture(b, n)
				ctx := context.Background()
				var sequence atomic.Uint64
				b.ReportAllocs()
				b.ResetTimer()
				if op == "parallel-mixed" {
					b.RunParallel(func(pb *testing.PB) {
						for pb.Next() {
							i := sequence.Add(1)
							key := keys[i%uint64(len(keys))]
							if i%3 == 0 {
								if err := cache.Store(ctx, key, entry); err != nil {
									b.Error(err)
								}
							} else {
								if _, _, err := cache.Load(ctx, key); err != nil {
									b.Error(err)
								}
							}
						}
					})
					return
				}
				for range b.N {
					switch op {
					case "overwrite":
						if err := cache.Store(ctx, "0", entry); err != nil {
							b.Fatal(err)
						}
					case "hit":
						if _, hit, err := cache.Load(ctx, "0"); err != nil || !hit {
							b.Fatalf("hit=%v: %v", hit, err)
						}
					case "miss":
						if _, hit, err := cache.Load(ctx, "absent"); err != nil || hit {
							b.Fatalf("hit=%v: %v", hit, err)
						}
					}
				}
			})
		}
	}
}
