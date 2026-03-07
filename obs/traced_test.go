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
