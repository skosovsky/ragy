package retrieval_test

import (
	"context"
	"testing"
	"time"

	"github.com/skosovsky/ragy/observation"
	"github.com/skosovsky/ragy/retrieval"
)

func TestObservedCacheHitAndMissAreActual(t *testing.T) {
	for _, hit := range []bool{false, true} {
		t.Run(map[bool]string{false: "miss", true: "hit"}[hit], func(t *testing.T) {
			// Arrange.
			clock := &cacheClock{instant: time.Unix(100, 0)}
			backend := &cacheBackend{
				schema:    cacheSchema(t),
				documents: []retrieval.Document[cacheMeta]{{ID: "private-document", Content: "private-content"}},
			}
			store := &cacheStoreSpy{
				hit: hit,
				entry: retrieval.CacheEntry[cacheMeta]{
					Documents: backend.documents,
					ExpiresAt: clock.now().Add(time.Minute),
				},
			}
			cached := cachedWithSpy(t, backend, store, clock, nil)
			var events []observation.Event
			session, err := observation.New(
				observation.Config{
					MaxEvents: 16,
					Observer: observation.ObserverFunc(
						func(_ context.Context, event observation.Event) error { events = append(events, event); return nil },
					),
				},
			)
			if err != nil {
				t.Fatal(err)
			}
			// Act.
			result, err := cached.Retrieve(
				observation.WithSession(context.Background(), session),
				retrieval.Query[struct{}]{
					Read:    retrieval.UnrestrictedRead(),
					Text:    "private query",
					Options: retrieval.RetrieveOptions{TopK: 1},
				},
			)
			// Assert.
			wantStage := observation.StageCacheMiss
			wantCalls := int64(1)
			if hit {
				wantStage = observation.StageCacheHit
				wantCalls = 0
			}
			if err != nil || result.Len() != 1 || backend.calls.Load() != wantCalls {
				t.Fatalf("result %v error %v calls %v", result, err, backend.calls.Load())
			}
			if len(events) != 4 || events[0].Stage != observation.StageCache || events[1].Stage != wantStage ||
				events[1].Parent != events[0].Operation ||
				events[2].Completion.Outcome != observation.OutcomeSuccess ||
				events[3].Completion.Count.Value != 1 {
				t.Fatalf("events %+v", events)
			}
		})
	}
}
