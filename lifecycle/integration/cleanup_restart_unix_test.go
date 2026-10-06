//go:build darwin || linux

package integration_test

import (
	"context"
	"errors"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/dense"
	densefs "github.com/skosovsky/ragy/dense/persistent"
	"github.com/skosovsky/ragy/graph"
	graphmanaged "github.com/skosovsky/ragy/graph/managed"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
	"github.com/skosovsky/ragy/tensor"
	tensorquery "github.com/skosovsky/ragy/tensor/query"
)

type lostCleanupResponse struct {
	base        lifecycle.CleanupPort
	calls       int
	inspections int
}

func (p *lostCleanupResponse) Cleanup(
	ctx context.Context,
	request lifecycle.CleanupRequest,
) (lifecycle.CleanupState, error) {
	p.calls++
	state, err := p.base.Cleanup(ctx, request)
	if err != nil {
		return state, err
	}
	return lifecycle.CleanupUnknown, context.DeadlineExceeded
}

func (p *lostCleanupResponse) InspectCleanup(
	ctx context.Context,
	request lifecycle.CleanupRequest,
) (lifecycle.CleanupState, error) {
	p.inspections++
	return p.base.InspectCleanup(ctx, request)
}

func TestActualCleanupLostResponseReconcilesAfterCleanerRestart(t *testing.T) {
	for _, target := range []string{"lexical", "tensor", "graph"} {
		for _, lost := range []string{"dense", target} {
			t.Run("dense+"+target+"/lost-"+lost, func(t *testing.T) { actualCleanupRestart(t, target, lost) })
		}
	}
}
func actualCleanupRestart(t *testing.T, target, lost string) {
	t.Helper()
	// Arrange: tombstone is a read barrier before either actual target is removed.
	f := newFixture(t, target)
	old := sourceBatch("policy", "r1", []string{"p1", "p2"})
	faq := sourceBatch("faq", "r1", []string{"f1"})
	f.ingest(t, plan("policy-r1", "", target, old), old)
	f.ingest(t, plan("faq-r1", "", target, faq), faq)
	captured := f.pin(t)
	deleted := plan("deleted", "policy-r1", target, old)
	deleted.Identity.Revision = "r2"
	deleted.Tombstone, deleted.Targets = true, nil
	if _, err := f.executor.Prepare(t.Context(), deleted); err != nil {
		t.Fatal(err)
	}
	if _, err := f.executor.Publish(t.Context(), "fixture-a", deleted.ID); err != nil {
		t.Fatal(err)
	}
	var base lifecycle.CleanupPort = f.dense
	if lost != "dense" {
		base = actualSecondaryCleanup(f)
	}
	fault := &lostCleanupResponse{base: base}
	cleaner := cleanupWithFault(t, f, lost, fault)
	if _, err := cleaner.Begin(t.Context(), "fixture-a", deleted.ID); err != nil {
		t.Fatal(err)
	}
	// Act: the real target removes data but its response is lost.
	_, err := cleaner.Attempt(t.Context(), "fixture-a", deleted.ID, "policy-r1", lost, false)
	if !errors.Is(err, lifecycle.ErrOutcomeUnknown) || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal("lost response inferred success", err)
	}
	restarted := cleanupWithFault(t, f, lost, fault)
	_, err = restarted.Attempt(t.Context(), "fixture-a", deleted.ID, "policy-r1", lost, false)
	if !errors.Is(err, lifecycle.ErrOutcomeUnknown) || fault.calls != 1 {
		t.Fatal("uncertain deletion repeated", fault.calls, err)
	}
	job, err := restarted.Reconcile(t.Context(), "fixture-a", deleted.ID, "policy-r1", lost)
	if err != nil || fault.calls != 1 || fault.inspections != 1 {
		t.Fatal("actual inspection recovery", err)
	}
	assertCleanupItem(t, job, lost, lifecycle.CleanupDone, 1)
	assertRetiredTargetUnavailable(t, f, captured, old, lost)
	// Finish the other target and replay completion without deletion or inspection.
	other := target
	if lost == target {
		other = "dense"
	}
	job, err = restarted.Attempt(t.Context(), "fixture-a", deleted.ID, "policy-r1", other, false)
	if err != nil || !job.Complete {
		t.Fatal("joint cleanup incomplete", job, err)
	}
	if _, err = restarted.Reconcile(
		t.Context(),
		"fixture-a",
		deleted.ID,
		"policy-r1",
		lost,
	); err != nil || fault.calls != 1 ||
		fault.inspections != 1 {
		t.Fatal("completed reconciliation repeated I/O", err)
	}
	denseDocs, secondaryDocs := f.read(t, f.pin(t), old, faq)
	assertRevision(t, denseDocs, "r1", 0)
	assertRevision(t, secondaryDocs, "r1", 0)
}
func actualSecondaryCleanup(f *fixture) lifecycle.CleanupPort {
	switch f.target {
	case "tensor":
		return f.secondary.tensor
	case "graph":
		return f.secondary.graph
	default:
		return f.secondary.lexical
	}
}
func cleanupWithFault(t *testing.T, f *fixture, lost string, fault lifecycle.CleanupPort) *lifecycle.Cleaner {
	t.Helper()
	return checkpointCleaner(t, f, f.store, lost, fault)
}

func checkpointCleaner(
	t *testing.T,
	f *fixture,
	store lifecycle.Store,
	lost string,
	fault lifecycle.CleanupPort,
) *lifecycle.Cleaner {
	t.Helper()
	var densePort lifecycle.CleanupPort = f.dense
	secondary := actualSecondaryCleanup(f)
	if lost == "dense" {
		densePort = fault
	} else {
		secondary = fault
	}
	cleaner, err := lifecycle.NewCleaner(
		lifecycle.CleanerConfig{
			Store: store,
			Now:   func() time.Time { return f.now },
			Targets: []lifecycle.CleanupRegistration{
				{Name: "dense", Port: densePort},
				{Name: f.target, Port: secondary},
			},
			Policy: lifecycle.CleanupPolicy{
				Deadline: time.Minute,
				Backoff:  []time.Duration{time.Second, 2 * time.Second, 4 * time.Second},
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return cleaner
}

func assertCleanupItem(
	t *testing.T,
	job lifecycle.CleanupJob,
	target string,
	state lifecycle.CleanupState,
	attempts uint64,
) {
	t.Helper()
	for _, item := range job.Items {
		if item.Target == target && item.Manifest == "policy-r1" {
			if item.State != state || item.Attempts != attempts {
				t.Fatal("cleanup checkpoint", item)
			}
			return
		}
	}
	t.Fatal("retired target checkpoint missing")
}
func assertRetiredTargetUnavailable(t *testing.T, f *fixture, read access.Binding, old batch, target string) {
	t.Helper()
	var err error
	switch target {
	case "dense":
		_, err = f.dense.Retrieve(
			t.Context(),
			retrieval.Query[densefs.Intent]{
				Read:    read,
				Intent:  densefs.Intent{Embedding: dense.Embedding{Space: denseSpace(), Vector: []float32{1, 0}}},
				Options: retrieval.RetrieveOptions{TopK: 10},
			},
		)
	case "lexical":
		_, err = f.secondary.lexical.Retrieve(
			t.Context(),
			retrieval.Query[struct{}]{Read: read, Text: "needle", Options: retrieval.RetrieveOptions{TopK: 10}},
		)
	case "tensor":
		_, err = f.secondary.tensor.Query(
			t.Context(),
			retrieval.Query[tensorquery.Intent]{
				Read: read,
				Intent: tensorquery.Intent{
					Embedding:       tensor.Embedding{Space: tensorSpace(), Tokens: tensor.Tensor{{1, 0}}},
					Candidates:      []source.Reference{old.Tensor[0].Reference},
					CandidateBudget: 100,
				},
				Options: retrieval.RetrieveOptions{TopK: 10},
			},
		)
	case "graph":
		_, err = f.secondary.graph.FindByIDs(
			t.Context(),
			graphmanaged.Request{
				Read: read,
				Traversal: graph.TraversalRequest{
					Seeds:     []string{old.Graph.Nodes[0].Value.ID},
					Direction: graph.DirectionOutbound,
					Depth:     1,
				},
				MaxNodes: 50,
				MaxEdges: 100,
			},
		)
	}
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatal("retired target served old or substituted snapshot", target, err)
	}
}
