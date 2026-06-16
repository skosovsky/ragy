package chunking_test

import (
	"context"
	"errors"
	"slices"
	"sync"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/chunking"
	"github.com/skosovsky/ragy/internal/contracttest"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/testutil"
)

type fixedSegmenter struct {
	Parts    []string
	Requests []string
}

func (s *fixedSegmenter) Split(text string) []string {
	s.Requests = append(s.Requests, text)
	return append([]string(nil), s.Parts...)
}

func TestRecursiveRequiresSourceID(t *testing.T) {
	splitter, err := chunking.NewRecursive[contracttest.StructMeta](16, 0, nil)
	if err != nil {
		t.Fatalf("NewRecursive(): %v", err)
	}

	if _, err := splitter.Split(
		context.Background(),
		retrieval.Document[contracttest.StructMeta]{Content: "hello"},
	); !errors.Is(err, ragy.ErrMissingSourceID) {
		t.Fatalf("Split() error = %v", err)
	}
}

func TestSemanticFailsOnEmbeddingCardinalityMismatch(t *testing.T) {
	embedder := &testutil.DenseEmbedder{Vectors: [][]float32{{1, 0}}}
	splitter, err := chunking.NewSemantic[contracttest.StructMeta](
		embedder,
		chunking.DefaultSentenceSegmenter{},
		0.5,
		1,
	)
	if err != nil {
		t.Fatalf("NewSemantic(): %v", err)
	}

	_, err = splitter.Split(context.Background(), retrieval.Document[contracttest.StructMeta]{
		ID:      "doc-1",
		Content: "First sentence. Second sentence.",
	})
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Split() error = %v", err)
	}
}

func TestSemanticFailsOnEmptyEmbeddingVector(t *testing.T) {
	embedder := &testutil.DenseEmbedder{Vectors: [][]float32{{}}}
	splitter, err := chunking.NewSemantic[contracttest.StructMeta](
		embedder,
		chunking.DefaultSentenceSegmenter{},
		0.5,
		1,
	)
	if err != nil {
		t.Fatalf("NewSemantic(): %v", err)
	}

	_, err = splitter.Split(context.Background(), retrieval.Document[contracttest.StructMeta]{
		ID:      "doc-1",
		Content: "First sentence.",
	})
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Split() error = %v", err)
	}
}

func TestSemanticFailsOnEmbeddingDimensionMismatch(t *testing.T) {
	embedder := &testutil.DenseEmbedder{Vectors: [][]float32{{1, 0}, {1, 0, 0}}}
	splitter, err := chunking.NewSemantic[contracttest.StructMeta](
		embedder,
		chunking.DefaultSentenceSegmenter{},
		0.5,
		1,
	)
	if err != nil {
		t.Fatalf("NewSemantic(): %v", err)
	}

	_, err = splitter.Split(context.Background(), retrieval.Document[contracttest.StructMeta]{
		ID:      "doc-1",
		Content: "First sentence. Second sentence.",
	})
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Split() error = %v", err)
	}
}

func TestSemanticFailsOnZeroNormEmbeddingVector(t *testing.T) {
	embedder := &testutil.DenseEmbedder{Vectors: [][]float32{{0, 0}}}
	splitter, err := chunking.NewSemantic[contracttest.StructMeta](
		embedder,
		chunking.DefaultSentenceSegmenter{},
		0.5,
		1,
	)
	if err != nil {
		t.Fatalf("NewSemantic(): %v", err)
	}

	_, err = splitter.Split(context.Background(), retrieval.Document[contracttest.StructMeta]{
		ID:      "doc-1",
		Content: "First sentence.",
	})
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Split() error = %v", err)
	}
}

func TestNewSemanticRejectsNilEmbedder(t *testing.T) {
	_, err := chunking.NewSemantic[contracttest.StructMeta](nil, chunking.DefaultSentenceSegmenter{}, 0.5, 1)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("NewSemantic(nil embedder) error = %v", err)
	}
}

func TestNewSemanticRejectsNilSegmenter(t *testing.T) {
	embedder := &testutil.DenseEmbedder{}

	if _, err := chunking.NewSemantic[contracttest.StructMeta](
		embedder,
		nil,
		0.5,
		1,
	); !errors.Is(
		err,
		ragy.ErrInvalidArgument,
	) {
		t.Fatalf("NewSemantic(nil segmenter) error = %v", err)
	}
}

func TestSemanticUsesInjectedSentenceSegmenter(t *testing.T) {
	segmenter := &fixedSegmenter{Parts: []string{"alpha beta", "gamma delta"}}
	embedder := &testutil.DenseEmbedder{Vectors: [][]float32{{1, 0}, {0, 1}}}
	splitter, err := chunking.NewSemantic[contracttest.StructMeta](embedder, segmenter, 0.5, 1)
	if err != nil {
		t.Fatalf("NewSemantic(): %v", err)
	}

	chunks, err := splitter.Split(context.Background(), retrieval.Document[contracttest.StructMeta]{
		ID:      "doc-1",
		Content: "ignored by fixed segmenter",
	})
	if err != nil {
		t.Fatalf("Split(): %v", err)
	}

	if !slices.Equal(segmenter.Requests, []string{"ignored by fixed segmenter"}) {
		t.Fatalf("segmenter requests = %#v, want source content", segmenter.Requests)
	}
	if len(embedder.Requests) != 1 || !slices.Equal(embedder.Requests[0], segmenter.Parts) {
		t.Fatalf("embedder requests = %#v, want %#v", embedder.Requests, segmenter.Parts)
	}
	if len(chunks) != 2 {
		t.Fatalf("len(chunks) = %d, want 2", len(chunks))
	}
	if chunks[0].Content != "alpha beta" || chunks[1].Content != "gamma delta" {
		t.Fatalf("chunks = %#v, want injected segmentation output", chunks)
	}
}

func TestDefaultSentenceSegmenterSplitsWithoutWhitespaceAfterBoundary(t *testing.T) {
	segmenter := chunking.DefaultSentenceSegmenter{}

	out := segmenter.Split("你好。世界")
	if len(out) != 2 {
		t.Fatalf("len(out) = %d, want 2 (%#v)", len(out), out)
	}
	if out[0] != "你好。" || out[1] != "世界" {
		t.Fatalf("out = %#v, want [\"你好。\", \"世界\"]", out)
	}
}

func TestContextualKeepsContentAndSetsContext(t *testing.T) {
	base, err := chunking.NewRecursive[contracttest.StructMeta](32, 0, []string{" "})
	if err != nil {
		t.Fatalf("NewRecursive(): %v", err)
	}

	contextual, err := chunking.NewContextual(base, &testutil.ContextGenerator{Value: "derived"}, 2)
	if err != nil {
		t.Fatalf("NewContextual(): %v", err)
	}

	chunks, err := contextual.Split(context.Background(), retrieval.Document[contracttest.StructMeta]{
		ID:      "doc-1",
		Content: "one two three four",
	})
	if err != nil {
		t.Fatalf("Split(): %v", err)
	}

	if len(chunks) == 0 {
		t.Fatal("Split() returned no chunks")
	}

	if chunks[0].Context != "derived" {
		t.Fatalf("chunks[0].Context = %q, want derived", chunks[0].Context)
	}

	if chunks[0].Content == "" {
		t.Fatal("chunks[0].Content = empty")
	}
}

type blockingContextGenerator struct {
	blockIndex int
	mu         sync.Mutex
	calls      int
	started    chan struct{}
	once       sync.Once
}

func (g *blockingContextGenerator) Context(
	ctx context.Context,
	_ retrieval.Document[contracttest.StructMeta],
	chunk chunking.Chunk[contracttest.StructMeta],
) (string, error) {
	g.once.Do(func() {
		if g.started != nil {
			close(g.started)
		}
	})
	g.mu.Lock()
	g.calls++
	g.mu.Unlock()
	if chunk.Index == g.blockIndex {
		<-ctx.Done()
		return "", ctx.Err()
	}
	return "derived", nil
}

func TestContextualSplitReturnsErrorOnContextCancel(t *testing.T) {
	t.Parallel()

	base, err := chunking.NewRecursive[contracttest.StructMeta](8, 0, []string{" "})
	if err != nil {
		t.Fatalf("NewRecursive(): %v", err)
	}

	gen := &blockingContextGenerator{blockIndex: 1, started: make(chan struct{})}
	contextual, err := chunking.NewContextual(base, gen, 2)
	if err != nil {
		t.Fatalf("NewContextual(): %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	doc := retrieval.Document[contracttest.StructMeta]{
		ID:      "doc-1",
		Content: "one two three four five six seven eight nine",
	}

	done := make(chan error, 1)
	go func() {
		_, splitErr := contextual.Split(ctx, doc)
		done <- splitErr
	}()

	select {
	case <-gen.started:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for enrichment to start")
	}
	cancel()

	splitErr := <-done
	if splitErr == nil {
		t.Fatal("Split() error = nil, want context cancel")
	}
	if !errors.Is(splitErr, context.Canceled) {
		t.Fatalf("Split() error = %v, want context canceled", splitErr)
	}
}

func TestRecursiveCopiesMetaFromSource(t *testing.T) {
	splitter, err := chunking.NewRecursive[contracttest.StructMeta](32, 0, nil)
	if err != nil {
		t.Fatalf("NewRecursive(): %v", err)
	}

	meta := contracttest.StructMeta{
		Tenant: "acme",
		Age:    7,
	}
	chunks, err := splitter.Split(context.Background(), retrieval.Document[contracttest.StructMeta]{
		ID:      "doc-1",
		Content: "hello world",
		Meta:    meta,
	})
	if err != nil {
		t.Fatalf("Split(): %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("Split() returned no chunks")
	}

	if chunks[0].Meta.Tenant != "acme" {
		t.Fatalf("chunks[0].Meta[tenant] = %#v, want acme", chunks[0].Meta.Tenant)
	}
	if chunks[0].Meta.Age != 7 {
		t.Fatalf("chunks[0].Meta[age] = %#v, want int64(7)", chunks[0].Meta.Age)
	}
}

func TestRecursiveReturnsZeroMetaWhenSourceMetaEmpty(t *testing.T) {
	splitter, err := chunking.NewRecursive[contracttest.StructMeta](32, 0, nil)
	if err != nil {
		t.Fatalf("NewRecursive(): %v", err)
	}

	chunks, err := splitter.Split(context.Background(), retrieval.Document[contracttest.StructMeta]{
		ID:      "doc-1",
		Content: "hello world",
	})
	if err != nil {
		t.Fatalf("Split(): %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("Split() returned no chunks")
	}
	if chunks[0].Meta != (contracttest.StructMeta{}) {
		t.Fatalf("chunks[0].Meta = %#v, want empty meta", chunks[0].Meta)
	}
}
