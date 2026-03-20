package otel

import (
	"context"
	"iter"
	"testing"

	"github.com/skosovsky/ragy"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/trace/noop"
)

type stubRetriever struct{}

func (stubRetriever) Retrieve(_ context.Context, _ ragy.SearchRequest) ([]ragy.Document, error) {
	return []ragy.Document{{ID: "1"}}, nil
}

func (stubRetriever) Stream(ctx context.Context, _ ragy.SearchRequest) iter.Seq2[ragy.Document, error] {
	return ragy.YieldDocuments(ctx, []ragy.Document{{ID: "s"}}, nil)
}

func TestWrapRetriever_Retrieve(t *testing.T) {
	tr := noop.NewTracerProvider().Tracer("test")
	wrapped := WrapRetriever(stubRetriever{}, tr, "r")
	out, err := wrapped.Retrieve(context.Background(), ragy.SearchRequest{})
	require.NoError(t, err)
	require.Len(t, out, 1)
	assert.Equal(t, "1", out[0].ID)
}

func TestWrapRetriever_Stream(t *testing.T) {
	tr := noop.NewTracerProvider().Tracer("test")
	wrapped := WrapRetriever(stubRetriever{}, tr, "r")
	var ids []string
	for d, err := range wrapped.Stream(context.Background(), ragy.SearchRequest{}) {
		require.NoError(t, err)
		ids = append(ids, d.ID)
	}
	assert.Equal(t, []string{"s"}, ids)
}
