package obs

import (
	"context"
	"testing"

	"go.opentelemetry.io/otel/trace/noop"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/retrievers"
	"github.com/skosovsky/ragy/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTracedRetriever_Retrieve(t *testing.T) {
	ctx := context.Background()
	tp := noop.NewTracerProvider()
	tr := tp.Tracer("test")
	emb := testutil.NewMockDenseEmbedder(4)
	store := testutil.NewInMemoryVectorStore()
	base := retrievers.NewBaseVectorRetriever(emb, store)
	wrapped := TracedRetriever(base, tr)
	res, err := wrapped.Retrieve(ctx, ragy.SearchRequest{Query: "hello", Limit: 5})
	require.NoError(t, err)
	require.NotNil(t, res)
}

func TestTracedVectorStore_Search(t *testing.T) {
	ctx := context.Background()
	tr := noop.NewTracerProvider().Tracer("test")
	store := testutil.NewInMemoryVectorStore()
	wrapped := TracedVectorStore(store, tr)
	docs, err := wrapped.Search(ctx, ragy.SearchRequest{Limit: 5})
	require.NoError(t, err)
	assert.Empty(t, docs)
}

type noopReranker struct{}

func (noopReranker) Rerank(_ context.Context, _ string, docs []ragy.Document, _ int) ([]ragy.Document, error) {
	return docs, nil
}

func TestTracedReranker_Rerank(t *testing.T) {
	ctx := context.Background()
	tr := noop.NewTracerProvider().Tracer("test")
	wrap := TracedReranker(noopReranker{}, tr)
	out, err := wrap.Rerank(ctx, "q", []ragy.Document{{ID: "1", Content: "x"}}, 10)
	require.NoError(t, err)
	require.Len(t, out, 1)
}

func TestTracedDenseEmbedder_Embed(t *testing.T) {
	ctx := context.Background()
	tr := noop.NewTracerProvider().Tracer("test")
	emb := testutil.NewMockDenseEmbedder(4)
	wrapped := TracedDenseEmbedder(emb, tr)
	vecs, err := wrapped.Embed(ctx, []string{"hello", "world"})
	require.NoError(t, err)
	assert.Len(t, vecs, 2)
	assert.Len(t, vecs[0], 4)
}

func TestTracedTensorEmbedder_EmbedTensors(t *testing.T) {
	ctx := context.Background()
	tr := noop.NewTracerProvider().Tracer("test")
	emb := testutil.NewMockTensorEmbedder(4, 8)
	wrapped := TracedTensorEmbedder(emb, tr)
	tensors, err := wrapped.EmbedTensors(ctx, []string{"hello"})
	require.NoError(t, err)
	assert.Len(t, tensors, 1)
	assert.Len(t, tensors[0], 4)
}

type noopTransformer struct{}

func (noopTransformer) Transform(_ context.Context, query string) ([]string, error) {
	return []string{query, query + " expanded"}, nil
}

func TestTracedQueryTransformer_Transform(t *testing.T) {
	ctx := context.Background()
	tr := noop.NewTracerProvider().Tracer("test")
	wrapped := TracedQueryTransformer(noopTransformer{}, tr)
	queries, err := wrapped.Transform(ctx, "hello")
	require.NoError(t, err)
	assert.Len(t, queries, 2)
}

type noopMultimodalEmbedder struct{}

func (noopMultimodalEmbedder) EmbedMultimodal(_ context.Context, texts []string, _ [][]ragy.Media) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = []float32{0.1, 0.2}
	}
	return out, nil
}

func TestTracedMultimodalEmbedder_EmbedMultimodal(t *testing.T) {
	ctx := context.Background()
	tr := noop.NewTracerProvider().Tracer("test")
	wrapped := TracedMultimodalEmbedder(noopMultimodalEmbedder{}, tr)
	vecs, err := wrapped.EmbedMultimodal(ctx, []string{"text"}, [][]ragy.Media{nil})
	require.NoError(t, err)
	assert.Len(t, vecs, 1)
}
