package observation_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/observation"
	"github.com/skosovsky/ragy/retrieval"
)

const eventCapacity = 256

type metadata struct {
	Tenant string `json:"tenant"`
	Secret string `json:"secret"`
}
type backend struct {
	index   *lexical.BM25Index[metadata]
	calls   atomic.Int64
	failure error
	empty   bool
	partial bool
}

func (b *backend) Schema() filter.Schema                 { return b.index.Schema() }
func (b *backend) ReadCapabilities() access.Capabilities { return b.index.ReadCapabilities() }
func (b *backend) Retrieve(ctx context.Context, q retrieval.Query[struct{}]) (retrieval.ResultSet[metadata], error) {
	b.calls.Add(1)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if b.partial {
		rs, err := b.index.Retrieve(ctx, q)
		if err != nil {
			return rs, err
		}
		return rs, b.failure
	}
	if b.failure != nil || b.empty {
		return retrieval.NewResultSet[metadata](nil, nil), b.failure
	}
	return b.index.Retrieve(ctx, q)
}
func fixture(t *testing.T) *backend {
	t.Helper()
	schema, err := filter.NewSchema().Build()
	if err != nil {
		t.Fatal(err)
	}
	index, err := lexical.NewBM25Index[metadata](
		schema,
		lexical.Config[metadata]{SearchFields: []string{"content"}},
		lexical.DefaultTokenizer{},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err = index.Upsert(
		retrieval.Document[metadata]{
			ID:      "private-id",
			Content: "needle private-content",
			Meta:    metadata{Tenant: "tenant-secret", Secret: "credential-secret"},
		},
	); err != nil {
		t.Fatal(err)
	}
	return &backend{index: index}
}
func cached(t *testing.T, next *backend) *retrieval.CachedBackend[struct{}, retrieval.NoRequestMeta, metadata] {
	t.Helper()
	clone := func(m metadata) (metadata, error) { return m, nil }
	store, err := retrieval.NewMemoryCache(2, time.Now, clone)
	if err != nil {
		t.Fatal(err)
	}
	result, err := retrieval.NewCachedBackend(retrieval.CacheConfig[struct{}, retrieval.NoRequestMeta, metadata]{
		Next: next, Store: store, TTL: time.Minute, Now: time.Now, CloneMeta: clone,
		SnapshotRequest: func(q retrieval.Query[struct{}]) (retrieval.Query[struct{}], error) { return q, nil },
		Identity: func(context.Context, retrieval.Query[struct{}]) (retrieval.CacheIdentity, error) {
			return retrieval.CacheIdentity{
				Index:         "index-secret",
				IndexRevision: "r1",
				Recipe:        "recipe-secret",
				Configuration: "config-secret",
				Capabilities:  []string{"lexical"},
			}, nil
		},
		HostIdentity: func(retrieval.Query[struct{}]) ([]byte, error) { return []byte("host-secret"), nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	return result
}

type collector struct {
	mu     sync.Mutex
	events []observation.Event
	fail   bool
}

func (c *collector) Observe(_ context.Context, e observation.Event) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.events = append(c.events, e)
	if c.fail {
		return errors.New("raw-provider-credential-secret")
	}
	return nil
}
func (c *collector) snapshot() []observation.Event {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]observation.Event(nil), c.events...)
}
func session(t *testing.T, c *collector) *observation.Session {
	t.Helper()
	s, err := observation.New(observation.Config{MaxEvents: eventCapacity, Observer: c})
	if err != nil {
		t.Fatal(err)
	}
	return s
}
func query() retrieval.Query[struct{}] {
	return retrieval.Query[struct{}]{
		Read:    retrieval.UnrestrictedRead(),
		Text:    "needle",
		Options: retrieval.RetrieveOptions{TopK: 1},
	}
}

type node = retrieval.ExecutionNode[struct{}, metadata, retrieval.NoExecutionMeta]

func pipeline(t *testing.T, root node) *retrieval.ExecutionPipeline[struct{}, metadata, retrieval.NoExecutionMeta] {
	t.Helper()
	p, err := retrieval.NewExecutionPipelineBuilder[struct{}, metadata, retrieval.NoExecutionMeta]().WithRoot(root).
		Build()
	if err != nil {
		t.Fatal(err)
	}
	return p
}
func terminal(events []observation.Event, stage observation.Stage, outcome observation.Outcome) bool {
	for _, e := range events {
		if e.Kind == observation.KindEnd && e.Stage == stage && e.Completion.Outcome == outcome {
			return true
		}
	}
	return false
}
func assertPrivate(t *testing.T, events []observation.Event) {
	t.Helper()
	raw, err := json.Marshal(events)
	if err != nil {
		t.Fatal(err)
	}
	for _, secret := range []string{"needle", "private-id", "private-content", "tenant-secret", "credential-secret", "index-secret", "host-secret", "raw-provider"} {
		if strings.Contains(string(raw), secret) {
			t.Fatalf("diagnostic leaked %q", secret)
		}
	}
}

func TestActualBM25CacheDisabledAndExporterFailure(t *testing.T) {
	// Arrange: actual BM25 behind the public cache and typed pipeline; exporter always fails.
	b := fixture(t)
	c := &collector{fail: true}
	s := session(t, c)
	p := pipeline(t, retrieval.BackendNode[struct{}, metadata, retrieval.NoExecutionMeta]{Backend: cached(t, b)})
	// Act: disabled first request populates cache; enabled request hits it.
	first, err := p.Execute(t.Context(), query())
	if err != nil {
		t.Fatal(err)
	}
	if len(c.snapshot()) != 0 {
		t.Fatal("disabled telemetry invoked exporter")
	}
	second, err := p.Execute(observation.WithSession(t.Context(), s), query())
	// Assert: exporter failure has no effect on actual content or dispatch count.
	if err != nil || first.Len() != 1 || second.Len() != 1 || b.calls.Load() != 1 {
		t.Fatal("diagnostics changed retrieval", err, b.calls.Load())
	}
	before, after := first.Documents()[0], second.Documents()[0]
	if before.ID != after.ID || before.Content != after.Content || before.Meta != after.Meta {
		t.Fatal("exporter changed actual retrieved document")
	}
	events := c.snapshot()
	if !terminal(events, observation.StageCacheHit, observation.OutcomeSuccess) {
		t.Fatal("cache hit not observed", events)
	}
	if s.Stats().Failures == 0 {
		t.Fatal("failed exporter not accounted")
	}
	assertPrivate(t, events)
}

func TestActualFallbackRescueBranches(t *testing.T) {
	for _, path := range []string{"fallback", "rescue"} {
		t.Run(path, func(t *testing.T) {
			// Arrange: first branch empty or failing; second branch dispatches actual BM25.
			first, second := fixture(t), fixture(t)
			c := &collector{}
			s := session(t, c)
			a := retrieval.BackendNode[struct{}, metadata, retrieval.NoExecutionMeta]{Backend: first}
			b := retrieval.BackendNode[struct{}, metadata, retrieval.NoExecutionMeta]{Backend: second}
			var root node
			stage := observation.StageFallback
			if path == "fallback" {
				first.empty = true
				root = retrieval.FallbackNode[struct{}, metadata, retrieval.NoExecutionMeta]{Primary: a, Secondary: b}
			} else {
				first.failure = ragy.ErrUnavailable
				stage = observation.StageRescue
				root = retrieval.RescueNode[struct{}, metadata, retrieval.NoExecutionMeta]{Primary: a, Secondary: b}
			}
			// Act.
			result, err := pipeline(t, root).Execute(observation.WithSession(t.Context(), s), query())
			// Assert: exactly the required two branches execute; all diagnostics remain payload-free.
			if err != nil || result.Len() != 1 || first.calls.Load() != 1 || second.calls.Load() != 1 {
				t.Fatal("composition dispatch mismatch", err)
			}
			events := c.snapshot()
			if !terminal(events, stage, observation.OutcomeSuccess) {
				t.Fatal("branch not observed", events)
			}
			assertPrivate(t, events)
		})
	}
}

func TestConcurrentExternalObserverBound(t *testing.T) {
	// Arrange: immutable BYOT metadata, shared finite session and actual BM25 pipeline.
	b := fixture(t)
	c := &collector{}
	s := session(t, c)
	p := pipeline(t, retrieval.BackendNode[struct{}, metadata, retrieval.NoExecutionMeta]{Backend: b})
	const workers = 8
	var wg sync.WaitGroup
	// Act.
	for range workers {
		wg.Go(func() {
			result, err := p.Execute(observation.WithSession(t.Context(), s), query())
			if err != nil || result.Len() != 1 {
				t.Error("concurrent retrieval failed", err)
			}
		})
	}
	wg.Wait()
	// Assert: no retries, accepted starts pair with exactly one completion, unknown usage stays unknown.
	if b.calls.Load() != workers {
		t.Fatal("unexpected dispatches")
	}
	events := c.snapshot()
	starts := map[uint64]bool{}
	ends := map[uint64]bool{}
	for _, e := range events {
		if e.Kind == observation.KindStart {
			starts[e.Operation] = true
		} else {
			if ends[e.Operation] {
				t.Fatal("duplicate completion")
			}
			ends[e.Operation] = true
			if e.Completion.Usage.InputTokens.Known {
				t.Fatal("BM25 invented model usage")
			}
		}
	}
	if len(starts) == 0 || len(starts) != len(ends) || s.Stats().Events > eventCapacity {
		t.Fatal("unpaired or unbounded diagnostics")
	}
	for op := range starts {
		if !ends[op] {
			t.Fatal("missing completion")
		}
	}
	assertPrivate(t, events)
}

func TestActualTerminalClassifications(t *testing.T) {
	for _, name := range []string{"partial", "unsupported", "canceled"} {
		t.Run(name, func(t *testing.T) {
			// Arrange: actual typed boundary with either retained BM25 evidence or rejected dispatch.
			b := fixture(t)
			c := &collector{}
			s := session(t, c)
			ctx := observation.WithSession(t.Context(), s)
			expected := observation.OutcomePartial
			switch name {
			case "partial":
				b.partial = true
				b.failure = errors.New("private-provider-body")
			case "unsupported":
				b.failure = ragy.ErrUnsupported
				expected = observation.OutcomeUnsupported
			case "canceled":
				canceled, cancel := context.WithCancel(ctx)
				cancel()
				ctx = canceled
				expected = observation.OutcomeCanceled
			}
			// Act.
			result, err := pipeline(
				t,
				retrieval.BackendNode[struct{}, metadata, retrieval.NoExecutionMeta]{Backend: b},
			).Execute(ctx, query())
			// Assert: partial retains actual evidence; canceled admission dispatches nothing, no fabricated success.
			if err == nil {
				t.Fatal("expected terminal error")
			}
			if name == "partial" && result.Len() != 1 {
				t.Fatal("partial evidence suppressed")
			}
			if name == "canceled" && b.calls.Load() != 0 {
				t.Fatal("canceled admission dispatched")
			}
			events := c.snapshot()
			if !terminal(events, observation.StagePipeline, expected) {
				t.Fatal("wrong terminal outcome", events)
			}
			assertPrivate(t, events)
		})
	}
}

func TestExhaustedTelemetryCapacityDoesNotExhaustRetrieval(t *testing.T) {
	// Arrange: only one start/end pair fits; child diagnostics must be dropped.
	b := fixture(t)
	c := &collector{}
	s, err := observation.New(observation.Config{MaxEvents: 2, Observer: c})
	if err != nil {
		t.Fatal(err)
	}
	p := pipeline(t, retrieval.BackendNode[struct{}, metadata, retrieval.NoExecutionMeta]{Backend: b})
	// Act.
	result, err := p.Execute(observation.WithSession(t.Context(), s), query())
	// Assert: bounded diagnostic drops never change actual retrieval or invent child observations.
	if err != nil || result.Len() != 1 || b.calls.Load() != 1 {
		t.Fatal("telemetry capacity changed retrieval", err)
	}
	stats := s.Stats()
	if stats.Events != 2 || stats.Dropped == 0 || len(c.snapshot()) != 2 {
		t.Fatal("telemetry capacity not enforced", stats)
	}
	if !terminal(c.snapshot(), observation.StagePipeline, observation.OutcomeSuccess) {
		t.Fatal("reserved completion missing")
	}
}
