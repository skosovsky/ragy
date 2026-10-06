package retrieval_test

import (
	"context"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/retrieval"
)

func TestCacheSeparatesFilterPlanAndGraphProfiles(t *testing.T) {
	schema := cacheSchema(t)
	field, err := schema.StringField("tenant")
	if err != nil {
		t.Fatal(err)
	}
	builder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	eq, err := filter.Eq(builder, field, "a").Build()
	if err != nil {
		t.Fatal(err)
	}
	in, err := filter.In(builder, field, "a", "b").Build()
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name   string
		change func(*retrieval.Query[struct{}])
	}{
		{"query-eq", func(q *retrieval.Query[struct{}]) { q.Options.Filters = eq }},
		{"query-in", func(q *retrieval.Query[struct{}]) { q.Options.Filters = in }},
		{"plan-filter", func(q *retrieval.Query[struct{}]) { q.Plan.Filters = eq }},
		{"plan-range", func(q *retrieval.Query[struct{}]) {
			q.Plan.Ranges = []retrieval.RangeConstraint{
				{Field: "tenant", Start: &retrieval.RangeBound{Text: "a", Inclusive: true}},
			}
		}},
		{"plan-cache-key", func(q *retrieval.Query[struct{}]) { q.Plan.CacheKey = "other-plan" }},
		{"plan-text", func(q *retrieval.Query[struct{}]) { q.Plan.Text = "other policy" }},
		{"graph-seeds", func(q *retrieval.Query[struct{}]) { q.Options.Graph.Seeds = []string{"other-seed"} }},
		{"graph-direction", func(q *retrieval.Query[struct{}]) { q.Options.Graph.Direction = graph.DirectionInbound }},
		{"graph-depth", func(q *retrieval.Query[struct{}]) { q.Options.Graph.Depth = 2 }},
		{"graph-node-filter", func(q *retrieval.Query[struct{}]) { q.Options.Graph.NodeFilter = eq }},
		{"graph-edge-filter", func(q *retrieval.Query[struct{}]) { q.Options.Graph.EdgeFilter = in }},
		{"graph-page", func(q *retrieval.Query[struct{}]) { q.Options.Graph.Page = &ragy.Page{Limit: 2} }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Arrange: a real owned memory cache over a conforming counted backend.
			clock := &cacheClock{instant: time.Unix(100, 0)}
			backend := &cacheBackend{
				schema:    schema,
				documents: []retrieval.Document[cacheMeta]{{ID: "a", Meta: cacheMeta{"tenant": "a"}}},
			}
			cached := newCachedFixture(t, backend, clock, nil)
			base := retrieval.Query[struct{}]{
				Read: retrieval.UnrestrictedRead(),
				Text: "policy",
				Plan: &retrieval.PlannedQuery[struct{}]{Text: "policy"},
				Options: retrieval.RetrieveOptions{
					TopK: 10,
					Graph: &retrieval.GraphOptions{
						Seeds:     []string{"seed"},
						Direction: graph.DirectionOutbound,
						Depth:     1,
					},
				},
			}
			changed := retrieval.CopyRequestOptions(base)
			tc.change(&changed)
			// Act: identical profiles hit; the differing option must dispatch once separately.
			for _, q := range []retrieval.Query[struct{}]{base, base, changed, changed, base} {
				if _, err := cached.Retrieve(context.Background(), q); err != nil {
					t.Fatal(err)
				}
			}
			// Assert: hashing alone is insufficient; observe actual cache reuse/isolation.
			if backend.calls.Load() != 2 {
				t.Fatalf("cache profile collision or failed reuse: calls=%d", backend.calls.Load())
			}
		})
	}
}
