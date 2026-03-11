package retrievers

import (
	"context"
	"errors"
	"testing"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSelfQueryRetriever_Retrieve(t *testing.T) {
	ctx := context.Background()
	emb := testutil.NewMockDenseEmbedder(4)
	store := testutil.NewInMemoryVectorStore()
	vec, _ := emb.Embed(ctx, []string{"pain"})
	_ = store.Upsert(ctx, []ragy.Document{{
		ID:       "1",
		Content:  "pain management",
		Metadata: map[string]any{testutil.EmbeddingKey: vec[0]},
	}})
	inner := NewBaseVectorRetriever(emb, store)
	parser := &mockQueryParser{
		parse: func(_ context.Context, _ string) (ragy.ParsedQuery, error) {
			return ragy.ParsedQuery{
				SemanticQuery: "pain",
				Filter:        filter.Equal("status", "active"),
				Limit:         0,
			}, nil
		},
	}
	sq := NewSelfQueryRetriever(inner, parser)
	res, err := sq.Retrieve(ctx, ragy.SearchRequest{Query: "user query about pain", Limit: 5})
	require.NoError(t, err)
	require.GreaterOrEqual(t, len(res.Documents), 1)
	assert.Equal(t, "pain", res.EvalData["parsed_semantic_query"])
	assert.Equal(t, true, res.EvalData["parsed_has_filter"])
}

func TestSelfQueryRetriever_MergeFilters(t *testing.T) {
	ctx := context.Background()
	var lastReq ragy.SearchRequest
	inner := &mockRetriever{
		retrieve: func(_ context.Context, req ragy.SearchRequest) (ragy.RetrievalResult, error) {
			lastReq = req
			return ragy.RetrievalResult{Documents: []ragy.Document{}}, nil
		},
	}
	parser := &mockQueryParser{
		parse: func(_ context.Context, _ string) (ragy.ParsedQuery, error) {
			return ragy.ParsedQuery{
				SemanticQuery: "x",
				Filter:        filter.Equal("status", "active"),
				Limit:         0,
			}, nil
		},
	}
	sq := NewSelfQueryRetriever(inner, parser)
	req := ragy.SearchRequest{
		Query:  "natural language query",
		Limit:  10,
		Filter: filter.Equal("tenant_id", "t1"),
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

func TestSelfQueryRetriever_ParserError(t *testing.T) {
	ctx := context.Background()
	errMock := errors.New("parser failed")
	inner := &mockRetriever{
		retrieve: func(_ context.Context, _ ragy.SearchRequest) (ragy.RetrievalResult, error) {
			return ragy.RetrievalResult{}, nil
		},
	}
	parser := &mockQueryParser{
		parse: func(_ context.Context, _ string) (ragy.ParsedQuery, error) {
			return ragy.ParsedQuery{}, errMock
		},
	}
	sq := NewSelfQueryRetriever(inner, parser)
	_, err := sq.Retrieve(ctx, ragy.SearchRequest{Query: "anything", Limit: 5})
	require.Error(t, err)
	assert.Equal(t, errMock, err)
}

type mockRetriever struct {
	retrieve func(ctx context.Context, req ragy.SearchRequest) (ragy.RetrievalResult, error)
}

func (m *mockRetriever) Retrieve(ctx context.Context, req ragy.SearchRequest) (ragy.RetrievalResult, error) {
	if m.retrieve != nil {
		return m.retrieve(ctx, req)
	}
	return ragy.RetrievalResult{}, nil
}

type mockQueryParser struct {
	parse func(ctx context.Context, naturalQuery string) (ragy.ParsedQuery, error)
}

func (m *mockQueryParser) Parse(ctx context.Context, naturalQuery string) (ragy.ParsedQuery, error) {
	if m.parse != nil {
		return m.parse(ctx, naturalQuery)
	}
	return ragy.ParsedQuery{}, nil
}
