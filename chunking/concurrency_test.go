package chunking_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/skosovsky/ragy/chunking"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
	"github.com/skosovsky/ragy/testutil"
)

type cooperativeGenerator struct {
	secondStarted chan struct{}
	secondDone    chan struct{}
	calls         atomic.Int64
	active        atomic.Int64
	failure       error
}

func (g *cooperativeGenerator) Context(
	ctx context.Context,
	_ retrieval.Document[struct{}],
	c chunking.Chunk[struct{}],
) (string, error) {
	g.calls.Add(1)
	g.active.Add(1)
	defer g.active.Add(-1)
	if c.Index == 0 {
		<-g.secondStarted
		return "", g.failure
	}
	if c.Index == 1 {
		close(g.secondStarted)
		<-ctx.Done()
		close(g.secondDone)
		return "", ctx.Err()
	}
	return "unexpected extra callback", nil
}
func TestContextualFailureCancelsAndJoinsCallbacks(t *testing.T) {
	// Arrange: one callback fails only after its cooperative sibling has started.
	base, err := chunking.NewRecursive[struct{}](2, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	failure := errors.New("enrichment failed")
	generator := &cooperativeGenerator{
		secondStarted: make(chan struct{}),
		secondDone:    make(chan struct{}),
		failure:       failure,
	}
	splitter, err := chunking.NewContextual[struct{}](base, generator, 2)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	chunks, err := splitter.Split(t.Context(), retrieval.Document[struct{}]{ID: "concurrent", Content: "abcdefghi"})
	// Assert: returned failure owns no partial output; callbacks have stopped.
	if !errors.Is(err, failure) || chunks != nil || generator.active.Load() != 0 || generator.calls.Load() != 2 {
		t.Fatalf("joined callbacks: %v active=%d calls=%d", err, generator.active.Load(), generator.calls.Load())
	}
	select {
	case <-generator.secondDone:
	default:
		t.Fatal("callback outlived Split")
	}
}
func TestCanceledSplitNeverDispatchesGenerator(t *testing.T) {
	// Arrange.
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	base, err := chunking.NewRecursive[struct{}](2, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	generator := &cooperativeGenerator{secondStarted: make(chan struct{}), secondDone: make(chan struct{})}
	splitter, err := chunking.NewContextual[struct{}](base, generator, 2)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	_, err = splitter.Split(ctx, retrieval.Document[struct{}]{ID: "cancelled", Content: "abcd"})
	// Assert.
	if !errors.Is(err, context.Canceled) || generator.calls.Load() != 0 {
		t.Fatal("cancelled callback dispatched", err)
	}
}

type cancelingSegmenter struct{ cancel context.CancelFunc }

func (s cancelingSegmenter) Split(context.Context, string) ([]source.ByteSpan, error) {
	s.cancel()
	return nil, nil
}
func TestSemanticChecksCancellationAfterSegmenter(t *testing.T) {
	// Arrange: canceled callback output must not cause another model call.
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	embedder := &testutil.DenseEmbedder{}
	splitter, err := chunking.NewSemantic[struct{}](embedder, cancelingSegmenter{cancel: cancel}, 0.5, 1)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	_, err = splitter.Split(ctx, retrieval.Document[struct{}]{ID: "semantic", Content: "One. Two."})
	// Assert.
	if !errors.Is(err, context.Canceled) || len(embedder.Requests) != 0 {
		t.Fatal("cancelled segmenter dispatched model", err)
	}
}
