//go:build darwin || linux

package joint_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"go.opentelemetry.io/otel/trace/noop"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	ragyotel "github.com/skosovsky/ragy/adapters/observability/otel"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/retrieval"
)

func (b *mixedBackend) AdmitPublication(p access.Publication) error { return p.AdmitTarget(b.target) }

func decorateJoint(
	t *testing.T,
	next retrieval.Backend[struct{}, meta],
	order string,
) retrieval.Backend[struct{}, meta] {
	t.Helper()
	if order == "plain" {
		return next
	}
	for _, kind := range order {
		switch kind {
		case 'c':
			cache, err := retrieval.NewMemoryCache(16, time.Now, cloneMeta)
			if err != nil {
				t.Fatal(err)
			}
			wrapped, err := retrieval.NewCachedBackend(retrieval.CacheConfig[struct{}, retrieval.NoRequestMeta, meta]{
				Next:      next,
				Store:     cache,
				TTL:       time.Minute,
				Now:       time.Now,
				CloneMeta: cloneMeta,
				SnapshotRequest: func(r retrieval.Query[struct{}]) (retrieval.Query[struct{}], error) {
					return retrieval.CopyRequestOptions(r), nil
				},
				Identity: func(context.Context, retrieval.Query[struct{}]) (retrieval.CacheIdentity, error) {
					return retrieval.CacheIdentity{
						Index:         "actual-dense",
						IndexRevision: "joint",
						Recipe:        "direct",
						Configuration: "qa",
						Capabilities:  []string{"scope", "pin"},
					}, nil
				},
				HostIdentity: func(retrieval.Query[struct{}]) ([]byte, error) { return []byte("normalized-dense-space"), nil },
			})
			if err != nil {
				t.Fatal(err)
			}
			next = wrapped
		case 't':
			wrapped, err := ragyotel.WrapBackend(next, noop.NewTracerProvider().Tracer("qa"))
			if err != nil {
				t.Fatal(err)
			}
			next = wrapped
		case 'p':
			next = retrieval.ProjectedBackend[struct{}, retrieval.NoRequestMeta, struct{}, retrieval.NoRequestMeta, meta]{
				Next:    next,
				Project: retrieval.CopyRequestOptions[struct{}, retrieval.NoRequestMeta],
			}
		}
	}
	return next
}

func TestActualPinnedDecoratorPermutations(t *testing.T) {
	for _, partial := range []bool{false, true} {
		for _, order := range []string{"plain", "ctp", "cpt", "tcp", "tpc", "pct", "ptc"} {
			t.Run(order+map[bool]string{false: "/complete", true: "/partial"}[partial], func(t *testing.T) {
				checkPinnedPermutation(t, order, partial)
			})
		}
	}
}

func TestExcludedDecoratedTargetNeverDispatches(t *testing.T) {
	for _, order := range []string{"plain", "ctp", "cpt", "tcp", "tpc", "pct", "ptc"} {
		t.Run(order, func(t *testing.T) {
			// Arrange: complete target exclusion is explicit in a partial binding.
			f := newFixture(t, "lexical")
			input := jointCorpus()
			f.ingest(t, plan("joint", "", "lexical", input), input)
			read := f.pin(t)
			var targets []access.TargetRevision
			for _, target := range read.Publication().Targets() {
				if target.Target != "dense" {
					targets = append(targets, target)
				}
			}
			p, err := access.PinPartialPublication("excluded-dense", targets, []string{"dense"})
			if err != nil {
				t.Fatal(err)
			}
			raw := &mixedBackend{f: f, input: input, target: "dense"}
			backend := decorateJoint(t, raw, order)
			node := retrieval.BackendNode[struct{}, meta, retrieval.NoExecutionMeta]{Backend: backend}
			// Act.
			_, err = node.Execute(
				t.Context(),
				retrieval.Query[struct{}]{Read: f.bind(t, p), Options: retrieval.RetrieveOptions{TopK: 1}},
				retrieval.NoExecutionMeta{},
			)
			// Assert.
			if !access.IsUnsupportedCapability(err) || raw.calls.Load() != 0 {
				t.Fatalf("excluded target dispatched: %v %d", err, raw.calls.Load())
			}
			if errors.Is(err, context.Canceled) {
				t.Fatal("wrong error classification")
			}
		})
	}
}

func TestRetiredPinNeverUsesCachedPayload(t *testing.T) {
	for _, order := range []string{"plain", "ctp", "cpt", "tcp", "tpc", "pct", "ptc"} {
		for _, warmed := range []bool{false, true} {
			t.Run(order+map[bool]string{false: "/cold", true: "/read-before-cleanup"}[warmed], func(t *testing.T) {
				checkRetiredPin(t, order, warmed)
			})
		}
	}
}

func checkPinnedPermutation(t *testing.T, order string, partial bool) {
	t.Helper()
	// Arrange: actual persistent dense + managed BM25 published under one inventory.
	f := newFixture(t, "lexical")
	input := jointCorpus()
	f.ingest(t, plan("joint", "", "lexical", input), input)
	read := f.pin(t)
	if partial {
		p, err := access.PinPartialPublication(
			"partial",
			read.Publication().Targets(),
			[]string{"excluded"},
		)
		if err != nil {
			t.Fatal(err)
		}
		read = f.bind(t, p)
	}
	backend := decorateJoint(t, &mixedBackend{f: f, input: input, target: "dense"}, order)
	pipeline, err := retrieval.NewExecutionPipelineBuilder[struct{}, meta, retrieval.NoExecutionMeta]().WithRoot(retrieval.BackendNode[struct{}, meta, retrieval.NoExecutionMeta]{Backend: backend}).
		Build()
	if err != nil {
		t.Fatal(err)
	}
	query := retrieval.Query[struct{}]{
		Read:    read,
		Text:    "policy",
		Options: retrieval.RetrieveOptions{TopK: 3},
	}
	// Act: repeated pinned reads revalidate the retained leaf.
	for range 2 {
		result, readErr := pipeline.Execute(t.Context(), query)
		// Assert.
		if readErr != nil || result.Len() != 1 || result.Documents()[0].Meta.Artifact != "allowed" ||
			result.Coverage.IsPartial() != partial {
			t.Fatalf("decorated pin: %v %#v %v", readErr, result.Documents(), result.Coverage.State())
		}
	}
	f.probe.mu.Lock()
	f.probe.revoked = true
	before := len(f.probe.ids)
	f.probe.mu.Unlock()
	result, err := pipeline.Execute(t.Context(), query)
	f.probe.mu.Lock()
	after := len(f.probe.ids)
	f.probe.mu.Unlock()
	if err == nil || !access.IsProtectionFailure(err) || !result.IsEmpty() || after != before {
		t.Fatalf("revoked cache read: %v io=%d/%d", err, before, after)
	}
}

func checkRetiredPin(t *testing.T, order string, warmed bool) {
	t.Helper()
	// Arrange: actual published persistent revision, optionally read before retirement.
	f := newFixture(t, "lexical")
	input := jointCorpus()
	f.ingest(t, plan("joint", "", "lexical", input), input)
	query := retrieval.Query[struct{}]{
		Read:    f.pin(t),
		Text:    "policy",
		Options: retrieval.RetrieveOptions{TopK: 3},
	}
	backend := decorateJoint(t, &mixedBackend{f: f, input: input, target: "dense"}, order)
	if warmed {
		rs, err := backend.Retrieve(t.Context(), query)
		if err != nil || rs.Len() != 1 {
			t.Fatalf("initial pin: %v", err)
		}
	}
	cleanupDense(t, f, input)
	f.probe.mu.Lock()
	before := len(f.probe.ids)
	f.probe.mu.Unlock()
	// Act: retain the old pin and the exact same composed backend/cache.
	result, err := backend.Retrieve(t.Context(), query)
	// Assert: no latest substitution, stale payload or payload I/O.
	f.probe.mu.Lock()
	after := len(f.probe.ids)
	f.probe.mu.Unlock()
	if !errors.Is(err, ragy.ErrUnavailable) || result.Len() != 0 || after != before {
		t.Fatalf("retired read: %v io=%d/%d", err, before, after)
	}
}

func cleanupDense(t *testing.T, f *fixture, input batch) {
	t.Helper()
	tombstone := plan("deleted", "joint", "lexical", input)
	tombstone.Identity.Revision = "r2"
	tombstone.Tombstone = true
	tombstone.Targets = nil
	if _, err := f.executor.Prepare(t.Context(), tombstone); err != nil {
		t.Fatal(err)
	}
	if _, err := f.executor.Publish(t.Context(), "fixture-a", "deleted"); err != nil {
		t.Fatal(err)
	}
	cleaner, err := lifecycle.NewCleaner(
		lifecycle.CleanerConfig{
			Store: f.store,
			Now:   func() time.Time { return f.now },
			Targets: []lifecycle.CleanupRegistration{
				{Name: "dense", Port: f.dense},
				{Name: "lexical", Port: f.secondary.lexical},
			},
			Policy: lifecycle.CleanupPolicy{Deadline: time.Minute, Backoff: []time.Duration{time.Second}},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = cleaner.Begin(t.Context(), "fixture-a", "deleted"); err != nil {
		t.Fatal(err)
	}
	job, err := cleaner.Attempt(t.Context(), "fixture-a", "deleted", "joint", "dense", false)
	if err != nil || len(job.Items) == 0 || job.Items[0].State != lifecycle.CleanupDone {
		t.Fatalf("cleanup: %v %#v", err, job)
	}
}
