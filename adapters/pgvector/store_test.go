package pgvector

import (
	"context"
	"testing"
	"time"

	"github.com/docker/go-connections/nat"
	"github.com/jackc/pgx/v5/pgxpool"
	_ "github.com/jackc/pgx/v5/stdlib"
	pgxvec "github.com/pgvector/pgvector-go/pgx"
	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/testcontainers/testcontainers-go"
	tcpostgres "github.com/testcontainers/testcontainers-go/modules/postgres"
	"github.com/testcontainers/testcontainers-go/wait"
)

const (
	createTable = `
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS knowledge_base (
  id TEXT PRIMARY KEY,
  content TEXT NOT NULL,
  embedding vector(3),
  metadata JSONB
);
`
)

// TestStore_Integration requires Docker; it starts a pgvector container and runs Search/Upsert/DeleteByFilter.
func TestStore_Integration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}
	ctx := context.Background()
	ctr, err := tcpostgres.Run(ctx, "pgvector/pgvector:pg16",
		tcpostgres.WithDatabase("test"),
		tcpostgres.WithUsername("postgres"),
		tcpostgres.WithPassword("postgres"),
		testcontainers.WithWaitStrategy(
			wait.ForSQL(nat.Port("5432/tcp"), "pgx", func(host string, port nat.Port) string {
				return "postgres://postgres:postgres@" + host + ":" + port.Port() + "/test?sslmode=disable"
			}).
				WithStartupTimeout(60*time.Second).
				WithPollInterval(200*time.Millisecond),
		),
	)
	require.NoError(t, err)
	defer func() { _ = ctr.Terminate(ctx) }()
	connStr, err := ctr.ConnectionString(ctx, "sslmode=disable")
	require.NoError(t, err)
	// Create extension and table before registering pgx vector types (RegisterTypes requires vector to exist).
	configBootstrap, err := pgxpool.ParseConfig(connStr)
	require.NoError(t, err)
	poolBootstrap, err := pgxpool.NewWithConfig(ctx, configBootstrap)
	require.NoError(t, err)
	_, err = poolBootstrap.Exec(ctx, createTable)
	require.NoError(t, err)
	poolBootstrap.Close()
	// Now create pool with vector type registration for store operations.
	config, err := pgxpool.ParseConfig(connStr)
	require.NoError(t, err)
	config.AfterConnect = pgxvec.RegisterTypes
	pool, err := pgxpool.NewWithConfig(ctx, config)
	require.NoError(t, err)
	defer pool.Close()

	store, err := New(pool, WithTable("knowledge_base"), WithUpsertBatchSize(10))
	require.NoError(t, err)

	// Upsert
	docs := []ragy.Document{
		{ID: "1", Content: "one", Metadata: map[string]any{ragy.EmbeddingMetadataKey: []float32{1, 0, 0}, "tenant_id": "t1"}},
		{ID: "2", Content: "two", Metadata: map[string]any{ragy.EmbeddingMetadataKey: []float32{0, 1, 0}, "tenant_id": "t1"}},
		{ID: "3", Content: "three", Metadata: map[string]any{ragy.EmbeddingMetadataKey: []float32{0, 0, 1}, "tenant_id": "t2"}},
	}
	err = store.Upsert(ctx, docs)
	require.NoError(t, err)

	// Search without filter
	req := ragy.SearchRequest{
		DenseVector: []float32{1, 0, 0},
		Limit:       5,
	}
	out, err := store.Search(ctx, req)
	require.NoError(t, err)
	require.GreaterOrEqual(t, len(out), 1)
	assert.Equal(t, "1", out[0].ID)

	// Search with filter (Eq)
	req.Filter = filter.Equal("tenant_id", "t1")
	out, err = store.Search(ctx, req)
	require.NoError(t, err)
	assert.Len(t, out, 2)

	// DeleteByFilter: remove tenant t2 (doc "3"); vector search returns nearest neighbors, so we get remaining docs 1 and 2.
	err = store.DeleteByFilter(ctx, filter.Equal("tenant_id", "t2"))
	require.NoError(t, err)
	out, err = store.Search(ctx, ragy.SearchRequest{DenseVector: []float32{0, 0, 1}, Limit: 5})
	require.NoError(t, err)
	assert.Len(t, out, 2)
	ids := map[string]bool{out[0].ID: true, out[1].ID: true}
	assert.True(t, ids["1"] && ids["2"], "results must be docs 1 and 2; deleted doc 3 must not appear")
}

func TestHybridRRFConfidence(t *testing.T) {
	k := 60
	maxScore := 2.0 / float64(k+1)
	assert.InDelta(t, 1.0, hybridRRFConfidence(maxScore, k), 1e-9)
	assert.InDelta(t, 0.5, hybridRRFConfidence(maxScore/2, k), 1e-9)
	assert.Equal(t, 0.0, hybridRRFConfidence(-1, k))
	assert.Equal(t, 1.0, hybridRRFConfidence(maxScore*2, k))
	// Single-list contribution capped at 1/(k+1) → half of dual max → confidence 0.5
	oneTerm := 1.0 / float64(k+1)
	assert.InDelta(t, 0.5, hybridRRFConfidence(oneTerm, k), 1e-9)
}

func TestSQLFilterVisitor_Unit(t *testing.T) {
	meta, err := sanitizeIdent("metadata")
	require.NoError(t, err)
	v := NewSQLFilterVisitor(meta)
	t.Run("Eq", func(t *testing.T) {
		w, args, err := v.ToSQL(filter.Equal("x", "v"), 1)
		require.NoError(t, err)
		assert.Contains(t, w, "@>")
		assert.Len(t, args, 1)
	})
	t.Run("And", func(t *testing.T) {
		w, args, err := v.ToSQL(filter.All(
			filter.Equal("a", 1),
			filter.Equal("b", "x"),
		), 1)
		require.NoError(t, err)
		assert.Contains(t, w, "AND")
		assert.Len(t, args, 2)
	})
}
