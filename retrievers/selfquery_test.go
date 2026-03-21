package retrievers

import (
	"context"
	"iter"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/testutil"
)

func TestSelfQueryRetriever_Retrieve(t *testing.T) {
	ctx := context.Background()
	emb := testutil.NewMockDenseEmbedder(4)
	store := testutil.NewInMemoryVectorStore()
	vec, _ := emb.Embed(ctx, []string{"pain"})
	_ = store.Upsert(ctx, []ragy.Document{{
		ID:       "1",
		Content:  "pain management",
		Metadata: map[string]any{testutil.EmbeddingKey: vec[0], "status": "active"},
	}})
	inner := NewBaseVectorRetriever(emb, store)
	sq := NewSelfQueryRetriever(inner)
	pq := &ragy.ParsedQuery{
		SemanticQuery: "pain",
		Filter:        filter.Equal("status", "active"),
		Limit:         0,
	}
	res, err := sq.Retrieve(ctx, ragy.SearchRequest{Query: "user query about pain", Limit: 5, ParsedQuery: pq})
	require.NoError(t, err)
	require.GreaterOrEqual(t, len(res), 1)
}

func TestSelfQueryRetriever_MergeFilters(t *testing.T) {
	ctx := context.Background()
	var lastReq ragy.SearchRequest
	inner := &mockRetriever{
		retrieve: func(_ context.Context, req ragy.SearchRequest) ([]ragy.Document, error) {
			lastReq = req
			return []ragy.Document{}, nil
		},
	}
	sq := NewSelfQueryRetriever(inner)
	pq := &ragy.ParsedQuery{
		SemanticQuery: "x",
		Filter:        filter.Equal("status", "active"),
		Limit:         0,
	}
	req := ragy.SearchRequest{
		Query:       "natural language query",
		Limit:       10,
		Filter:      filter.Equal("tenant_id", "t1"),
		ParsedQuery: pq,
	}
	_, err := sq.Retrieve(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, lastReq.Filter)
	andExpr, ok := lastReq.Filter.(filter.And)
	require.True(t, ok, "expected filter to be And for merged RBAC + parsed")
	require.Len(t, andExpr.Exprs, 2)
	eq0, ok := andExpr.Exprs[0].(filter.Eq)
	require.True(t, ok)
	assert.Equal(t, "tenant_id", eq0.Field)
	assert.Equal(t, "t1", eq0.Value)
	eq1, ok := andExpr.Exprs[1].(filter.Eq)
	require.True(t, ok)
	assert.Equal(t, "status", eq1.Field)
	assert.Equal(t, "active", eq1.Value)
}

func TestSelfQueryRetriever_MissingParsedQuery(t *testing.T) {
	ctx := context.Background()
	inner := NewBaseVectorRetriever(testutil.NewMockDenseEmbedder(4), testutil.NewInMemoryVectorStore())
	sq := NewSelfQueryRetriever(inner)
	_, err := sq.Retrieve(ctx, ragy.SearchRequest{Query: "anything", Limit: 5})
	require.Error(t, err)
	assert.ErrorIs(t, err, ragy.ErrMissingParsedQuery)
}

type mockRetriever struct {
	retrieve func(ctx context.Context, req ragy.SearchRequest) ([]ragy.Document, error)
}

func (m *mockRetriever) Retrieve(ctx context.Context, req ragy.SearchRequest) ([]ragy.Document, error) {
	if m.retrieve != nil {
		return m.retrieve(ctx, req)
	}
	return nil, nil
}

func (m *mockRetriever) Stream(ctx context.Context, req ragy.SearchRequest) iter.Seq2[ragy.Document, error] {
	docs, err := m.Retrieve(ctx, req)
	return ragy.YieldDocuments(ctx, docs, err)
}
