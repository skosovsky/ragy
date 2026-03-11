package cache

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/goleak"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/testutil"
)

func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m)
}

func TestVectorCache_Hit(t *testing.T) {
	store := testutil.NewInMemoryVectorStore()
	embedder := testutil.NewMockDenseEmbedder(4)
	c := NewVectorCache(store, embedder)
	ctx := context.Background()

	err := c.Set(ctx, "Hello", "response")
	require.NoError(t, err)

	resp, hit, err := c.Get(ctx, "Hello", 0.8)
	require.NoError(t, err)
	assert.True(t, hit)
	assert.Equal(t, "response", resp)
}

func TestVectorCache_Miss_BelowThreshold(t *testing.T) {
	store := testutil.NewInMemoryVectorStore()
	embedder := testutil.NewMockDenseEmbedder(4)
	c := NewVectorCache(store, embedder)
	ctx := context.Background()

	err := c.Set(ctx, "Hello", "resp")
	require.NoError(t, err)

	// Different query yields different embedding; similarity with "Hello" will be < 0.99
	resp, hit, err := c.Get(ctx, "something else", 0.99)
	require.NoError(t, err)
	assert.False(t, hit)
	assert.Empty(t, resp)
}

func TestVectorCache_Miss_EmptyStore(t *testing.T) {
	store := testutil.NewInMemoryVectorStore()
	embedder := testutil.NewMockDenseEmbedder(4)
	c := NewVectorCache(store, embedder)
	ctx := context.Background()

	resp, hit, err := c.Get(ctx, "any query", 0.5)
	require.NoError(t, err)
	assert.False(t, hit)
	assert.Empty(t, resp)
}

// TestVectorCache_GetOnlyCacheEntriesInMixedStore ensures Get only returns cache entries
// when the same store holds both cache docs (_cache_type=semantic) and normal KB documents.
func TestVectorCache_GetOnlyCacheEntriesInMixedStore(t *testing.T) {
	store := testutil.NewInMemoryVectorStore()
	embedder := testutil.NewMockDenseEmbedder(4)
	ctx := context.Background()

	// Insert a normal KB document (no _cache_type) with same embedding dimension
	vecs, err := embedder.Embed(ctx, []string{"Hello"})
	require.NoError(t, err)
	require.Len(t, vecs, 1)
	_ = store.Upsert(ctx, []ragy.Document{
		{
			ID:       "kb-doc",
			Content:  "wrong answer from knowledge base",
			Metadata: map[string]any{testutil.EmbeddingKey: vecs[0]},
		},
	})

	// Add cache entry via VectorCache (has _cache_type: semantic)
	c := NewVectorCache(store, embedder)
	err = c.Set(ctx, "Hello", "cached_response")
	require.NoError(t, err)

	resp, hit, err := c.Get(ctx, "Hello", 0.8)
	require.NoError(t, err)
	assert.True(t, hit)
	assert.Equal(t, "cached_response", resp, "Get must return cache entry, not KB doc")
}

func TestTracedSemanticCache(t *testing.T) {
	store := testutil.NewInMemoryVectorStore()
	embedder := testutil.NewMockDenseEmbedder(4)
	inner := NewVectorCache(store, embedder)
	tracer := trace.NewNoopTracerProvider().Tracer("test")
	wrapped := NewTracedSemanticCache(inner, tracer)
	ctx := context.Background()

	err := wrapped.Set(ctx, "q", "a")
	require.NoError(t, err)

	resp, hit, err := wrapped.Get(ctx, "q", 0.8)
	require.NoError(t, err)
	assert.True(t, hit)
	assert.Equal(t, "a", resp)
}

// emptyVecEmbedder returns a single empty vector for tests (defensive behavior).
type emptyVecEmbedder struct{}

func (emptyVecEmbedder) Embed(_ context.Context, _ []string) ([][]float32, error) {
	return [][]float32{{}}, nil
}

func TestVectorCache_Get_EmptyVectorReturnsError(t *testing.T) {
	store := testutil.NewInMemoryVectorStore()
	c := NewVectorCache(store, &emptyVecEmbedder{})
	ctx := context.Background()
	_, _, err := c.Get(ctx, "q", 0.5)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "empty embedding")
}

func TestVectorCache_Set_EmptyVectorReturnsError(t *testing.T) {
	store := testutil.NewInMemoryVectorStore()
	c := NewVectorCache(store, &emptyVecEmbedder{})
	ctx := context.Background()
	err := c.Set(ctx, "q", "resp")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "empty embedding")
}

// semanticStubEmbedder returns controlled vectors: same vector for queries that map to the same key (simulates semantic similarity).
type semanticStubEmbedder struct {
	vecByKey map[string][]float32
	dim      int
}

func newSemanticStubEmbedder(dim int, sameVecQueries ...[]string) *semanticStubEmbedder {
	m := make(map[string][]float32)
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = 0.1 * float32(i+1)
	}
	for _, group := range sameVecQueries {
		for _, q := range group {
			vecCopy := make([]float32, len(vec))
			copy(vecCopy, vec)
			m[q] = vecCopy
		}
		// next group gets a different vector
		for i := range vec {
			vec[i] += 0.5
		}
	}
	return &semanticStubEmbedder{vecByKey: m, dim: dim}
}

func (s *semanticStubEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		if v, ok := s.vecByKey[t]; ok {
			cp := make([]float32, len(v))
			copy(cp, v)
			out[i] = cp
		} else {
			// unknown query: return distinct vector so similarity is low
			vec := make([]float32, s.dim)
			for j := range vec {
				vec[j] = float32(len(t)+j) * 0.01
			}
			out[i] = vec
		}
	}
	return out, nil
}

// TestVectorCache_SemanticHit_ControlledStub: semantically similar queries (same vector from stub) yield cache hit.
func TestVectorCache_SemanticHit_ControlledStub(t *testing.T) {
	stub := newSemanticStubEmbedder(4, []string{"what is the weather", "how is the weather today"})
	store := testutil.NewInMemoryVectorStore()
	c := NewVectorCache(store, stub)
	ctx := context.Background()

	err := c.Set(ctx, "what is the weather", "It's sunny.")
	require.NoError(t, err)

	resp, hit, err := c.Get(ctx, "how is the weather today", 0.99)
	require.NoError(t, err)
	assert.True(t, hit)
	assert.Equal(t, "It's sunny.", resp)
}

// TestVectorCache_SemanticMiss_ControlledStub: different queries (different vectors from stub) yield miss.
func TestVectorCache_SemanticMiss_ControlledStub(t *testing.T) {
	stub := newSemanticStubEmbedder(4, []string{"only this"})
	store := testutil.NewInMemoryVectorStore()
	c := NewVectorCache(store, stub)
	ctx := context.Background()

	err := c.Set(ctx, "only this", "resp")
	require.NoError(t, err)

	resp, hit, err := c.Get(ctx, "other query", 0.5)
	require.NoError(t, err)
	assert.False(t, hit)
	assert.Empty(t, resp)
}

var errEmbedderFailure = errors.New("embedder failed")

type errEmbedder struct{}

func (errEmbedder) Embed(_ context.Context, _ []string) ([][]float32, error) {
	return nil, errEmbedderFailure
}

func TestVectorCache_Get_EmbedderError(t *testing.T) {
	store := testutil.NewInMemoryVectorStore()
	c := NewVectorCache(store, &errEmbedder{})
	ctx := context.Background()
	_, _, err := c.Get(ctx, "q", 0.5)
	require.Error(t, err)
	assert.ErrorIs(t, err, errEmbedderFailure)
}

func TestVectorCache_Set_EmbedderError(t *testing.T) {
	store := testutil.NewInMemoryVectorStore()
	c := NewVectorCache(store, &errEmbedder{})
	ctx := context.Background()
	err := c.Set(ctx, "q", "r")
	require.Error(t, err)
	assert.ErrorIs(t, err, errEmbedderFailure)
}

// noVecEmbedder returns no vectors (empty slice).
type noVecEmbedder struct{}

func (noVecEmbedder) Embed(_ context.Context, _ []string) ([][]float32, error) {
	return nil, nil
}

func TestVectorCache_Get_NoEmbeddingReturned(t *testing.T) {
	store := testutil.NewInMemoryVectorStore()
	c := NewVectorCache(store, &noVecEmbedder{})
	ctx := context.Background()
	_, _, err := c.Get(ctx, "q", 0.5)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no embedding")
}

func TestVectorCache_Set_NoEmbeddingReturned(t *testing.T) {
	store := testutil.NewInMemoryVectorStore()
	c := NewVectorCache(store, &noVecEmbedder{})
	ctx := context.Background()
	err := c.Set(ctx, "q", "r")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no embedding")
}

var errSearch = errors.New("search failed")

type failingSearchStore struct {
	inner ragy.VectorStore
}

func (f *failingSearchStore) Search(_ context.Context, _ ragy.SearchRequest) ([]ragy.Document, error) {
	return nil, errSearch
}

func (f *failingSearchStore) Upsert(ctx context.Context, docs []ragy.Document) error {
	return f.inner.Upsert(ctx, docs)
}

func (f *failingSearchStore) DeleteByFilter(ctx context.Context, expr filter.Expr) error {
	return f.inner.DeleteByFilter(ctx, expr)
}

func TestVectorCache_Get_SearchError(t *testing.T) {
	embedder := testutil.NewMockDenseEmbedder(4)
	store := &failingSearchStore{inner: testutil.NewInMemoryVectorStore()}
	c := NewVectorCache(store, embedder)
	ctx := context.Background()
	_, _, err := c.Get(ctx, "q", 0.5)
	require.Error(t, err)
	assert.ErrorIs(t, err, errSearch)
}

var errUpsert = errors.New("upsert failed")

type failingUpsertStore struct {
	inner ragy.VectorStore
}

func (f *failingUpsertStore) Search(ctx context.Context, req ragy.SearchRequest) ([]ragy.Document, error) {
	return f.inner.Search(ctx, req)
}

func (f *failingUpsertStore) Upsert(_ context.Context, _ []ragy.Document) error {
	return errUpsert
}

func (f *failingUpsertStore) DeleteByFilter(ctx context.Context, expr filter.Expr) error {
	return f.inner.DeleteByFilter(ctx, expr)
}

func TestVectorCache_Set_UpsertError(t *testing.T) {
	embedder := testutil.NewMockDenseEmbedder(4)
	store := &failingUpsertStore{inner: testutil.NewInMemoryVectorStore()}
	c := NewVectorCache(store, embedder)
	ctx := context.Background()
	err := c.Set(ctx, "q", "r")
	require.Error(t, err)
	assert.ErrorIs(t, err, errUpsert)
}
