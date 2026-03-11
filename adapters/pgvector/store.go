// Package pgvector provides a ragy.VectorStore implementation using PostgreSQL with the pgvector extension.
//
// Documents store the embedding in Metadata[ragy.EmbeddingMetadataKey] ([]float32). Use Batch Upsert and filter.Expr translation for WHERE.
//
// Table and column names (WithTable, WithMetadataColumn, etc.) must be set by the application and must not
// be taken from unvalidated user input, as they are used in raw SQL.
//
// Example schema with HNSW index (caller must create; Store does not create tables or indexes):
//
//	CREATE EXTENSION IF NOT EXISTS vector;
//	CREATE TABLE knowledge_base (
//	  id TEXT PRIMARY KEY,
//	  content TEXT NOT NULL,
//	  embedding vector(1536),
//	  metadata JSONB
//	);
//	CREATE INDEX ON knowledge_base USING hnsw (embedding vector_cosine_ops);
//
// For hybrid search, add a GIN index on the text column used for FTS, e.g.:
//
//	CREATE INDEX ON knowledge_base USING GIN (to_tsvector('english', content));
//
// Hybrid search (vector + full-text via tsvector in one query) is supported when WithHybridSearch(true)
// and WithLanguage(regconfig) are set; use WithFTSColumn to specify the column for FTS (default "content").
// FTS uses websearch_to_tsquery($lang, $query). Example DDL for hybrid: add GIN index on to_tsvector
// or a generated tsvector column; e.g. CREATE INDEX ON knowledge_base USING GIN (to_tsvector('english', content));
package pgvector

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

// DefaultUpsertBatchSize is the default micro-batch size for Upsert.
const DefaultUpsertBatchSize = 500

// Store implements ragy.VectorStore using pgxpool and pgvector.
type Store struct {
	pool       *pgxpool.Pool
	table      string
	embedCol   string
	idCol      string
	contentCol string
	metaCol    string
	batchSize  int
	hybrid     bool   // enable hybrid (vector + FTS) when req.Query and DenseVector are set
	ftsCol     string // column for full-text search (default "content")
	lang       string // regconfig for websearch_to_tsquery (default "english")
}

// Option configures the Store.
type Option func(*Store)

// WithTable sets the table name (default "knowledge_base").
func WithTable(name string) Option {
	return func(s *Store) { s.table = name }
}

// WithEmbedColumn sets the embedding column name (default "embedding").
func WithEmbedColumn(name string) Option {
	return func(s *Store) { s.embedCol = name }
}

// WithIDColumn sets the id column name (default "id").
func WithIDColumn(name string) Option {
	return func(s *Store) { s.idCol = name }
}

// WithContentColumn sets the content column name (default "content").
func WithContentColumn(name string) Option {
	return func(s *Store) { s.contentCol = name }
}

// WithMetadataColumn sets the metadata JSONB column name (default "metadata").
func WithMetadataColumn(name string) Option {
	return func(s *Store) { s.metaCol = name }
}

// WithUpsertBatchSize sets the micro-batch size for Upsert (default 500).
func WithUpsertBatchSize(n int) Option {
	return func(s *Store) { s.batchSize = n }
}

// WithHybridSearch enables hybrid search (vector + FTS with RRF) when req.Query and req.DenseVector are both set.
func WithHybridSearch(enabled bool) Option {
	return func(s *Store) { s.hybrid = enabled }
}

// WithFTSColumn sets the column used for full-text search (default "content"). Must be a text column.
func WithFTSColumn(name string) Option {
	return func(s *Store) { s.ftsCol = name }
}

// WithLanguage sets the text search config (regconfig) for websearch_to_tsquery, e.g. "english", "russian", "simple" (default "english").
func WithLanguage(lang string) Option {
	return func(s *Store) { s.lang = lang }
}

// New returns a new pgvector Store. The pool must have pgvector types registered (e.g. pgxvec.RegisterTypes in AfterConnect).
func New(pool *pgxpool.Pool, opts ...Option) *Store {
	s := &Store{
		pool:       pool,
		table:      "knowledge_base",
		embedCol:   "embedding",
		idCol:      "id",
		contentCol: "content",
		metaCol:    "metadata",
		batchSize:  DefaultUpsertBatchSize,
		ftsCol:     "content",
		lang:       "english",
	}
	for _, o := range opts {
		o(s)
	}
	return s
}

// Search implements ragy.VectorStore.
// When hybrid search is enabled (WithHybridSearch) and req.Query and req.DenseVector are set,
// runs vector + FTS with RRF merge in one query.
func (s *Store) Search(ctx context.Context, req ragy.SearchRequest) ([]ragy.Document, error) {
	if len(req.DenseVector) == 0 {
		return []ragy.Document{}, nil
	}
	vec := pgvector.NewVector(req.DenseVector)
	limit := req.Limit
	if limit <= 0 {
		limit = 10
	}
	offset := req.Offset
	if offset < 0 {
		offset = 0
	}
	where, whereArgs, err := buildWhere(req.Filter, s.metaCol)
	if err != nil {
		return nil, err
	}
	w := len(whereArgs)

	if s.hybrid && req.Query != "" {
		// Hybrid: two CTEs + FULL OUTER JOIN + RRF score.
		whereClause := ""
		if where != "" {
			whereClause = " WHERE " + where
		}
		// CTE vector_search: rank by vector distance.
		// Placeholders: where $1..$w, vec $w+1. CTE limit 1000 to bound work.
		args := append([]any{}, whereArgs...)
		args = append(args, vec, s.lang, req.Query)
		var whereFts string
		var whereFtsArgs []any
		if req.Filter != nil {
			whereFts, whereFtsArgs, err = buildWhereRec(req.Filter, w+4, s.metaCol) // fts WHERE starts at $w+4 after vec, lang, query
			if err != nil {
				return nil, err
			}
		}
		ftsWhere := fmt.Sprintf(" to_tsvector($%d::regconfig, %s) @@ websearch_to_tsquery($%d::regconfig, $%d)", w+2, s.ftsCol, w+2, w+3)
		if whereFts != "" {
			ftsWhere += " AND (" + whereFts + ")"
		}
		args = append(args, whereFtsArgs...)
		args = append(args, limit, offset)
		n := len(args)

		q := fmt.Sprintf(
			`WITH vector_search AS (
  SELECT %s, %s, %s, row_number() OVER (ORDER BY %s <=> $%d) AS rank_v
  FROM %s%s
  LIMIT 1000
),
fts_search AS (
  SELECT %s, %s, %s, row_number() OVER (ORDER BY ts_rank_cd(to_tsvector($%d::regconfig, %s), websearch_to_tsquery($%d::regconfig, $%d)) DESC NULLS LAST) AS rank_fts
  FROM %s
  WHERE %s
  LIMIT 1000
)
SELECT COALESCE(v.%s, f.%s), COALESCE(v.%s, f.%s), COALESCE(v.%s, f.%s),
  (COALESCE(1.0/(60+v.rank_v),0) + COALESCE(1.0/(60+f.rank_fts),0)) AS score
FROM vector_search v
FULL OUTER JOIN fts_search f ON v.%s = f.%s
ORDER BY score DESC NULLS LAST
LIMIT $%d OFFSET $%d`,
			s.idCol, s.contentCol, s.metaCol, s.embedCol, w+1, s.table, whereClause,
			s.idCol, s.contentCol, s.metaCol, w+2, s.ftsCol, w+2, w+3, s.table, ftsWhere,
			s.idCol, s.idCol, s.contentCol, s.contentCol, s.metaCol, s.metaCol,
			s.idCol, s.idCol, n-1, n,
		)
		rows, err := s.pool.Query(ctx, q, args...)
		if err != nil {
			return nil, err
		}
		defer rows.Close()
		var out []ragy.Document
		for rows.Next() {
			var id, content string
			var metaJSON []byte
			var score float64
			if err := rows.Scan(&id, &content, &metaJSON, &score); err != nil {
				return nil, err
			}
			meta := make(map[string]any)
			if len(metaJSON) > 0 {
				if err := json.Unmarshal(metaJSON, &meta); err != nil {
					return nil, err
				}
			}
			doc := ragy.Document{ID: id, Content: content, Metadata: meta, Score: float32(score)}
			out = append(out, doc)
		}
		return out, rows.Err()
	}

	// Vector-only search. Include cosine distance in SELECT so we can set doc.Score (1 - distance).
	baseArgs := append([]any{}, whereArgs...)
	baseArgs = append(baseArgs, vec, limit, offset)
	n := len(baseArgs)
	vecArgNum := n - 2 // placeholder index for vec in ORDER BY
	q := fmt.Sprintf("SELECT %s, %s, %s, (%s <=> $%d) AS distance FROM %s",
		s.idCol, s.contentCol, s.metaCol, s.embedCol, vecArgNum, s.table)
	if where != "" {
		q += " WHERE " + where
	}
	q += fmt.Sprintf(" ORDER BY %s <=> $%d LIMIT $%d OFFSET $%d",
		s.embedCol, vecArgNum, n-1, n)
	rows, err := s.pool.Query(ctx, q, baseArgs...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []ragy.Document
	for rows.Next() {
		var id, content string
		var metaJSON []byte
		var distance float64
		if err := rows.Scan(&id, &content, &metaJSON, &distance); err != nil {
			return nil, err
		}
		meta := make(map[string]any)
		if len(metaJSON) > 0 {
			if err := json.Unmarshal(metaJSON, &meta); err != nil {
				return nil, err
			}
		}
		doc := ragy.Document{ID: id, Content: content, Metadata: meta, Score: float32(1.0 - distance)}
		out = append(out, doc)
	}
	return out, rows.Err()
}

// Upsert implements ragy.VectorStore. Uses pgx.Batch with INSERT ... ON CONFLICT DO UPDATE (no CopyFrom).
func (s *Store) Upsert(ctx context.Context, docs []ragy.Document) error {
	if len(docs) == 0 {
		return nil
	}
	for i := 0; i < len(docs); i += s.batchSize {
		end := i + s.batchSize
		if end > len(docs) {
			end = len(docs)
		}
		batch := docs[i:end]
		if err := s.upsertBatch(ctx, batch); err != nil {
			return err
		}
	}
	return nil
}

func (s *Store) upsertBatch(ctx context.Context, docs []ragy.Document) error {
	batch := &pgx.Batch{}
	q := fmt.Sprintf(
		"INSERT INTO %s (%s, %s, %s, %s) VALUES ($1, $2, $3, $4) ON CONFLICT (%s) DO UPDATE SET %s = EXCLUDED.%s, %s = EXCLUDED.%s, %s = EXCLUDED.%s",
		s.table, s.idCol, s.contentCol, s.embedCol, s.metaCol,
		s.idCol,
		s.contentCol, s.contentCol, s.embedCol, s.embedCol, s.metaCol, s.metaCol,
	)
	for _, d := range docs {
		emb, _ := d.Metadata[ragy.EmbeddingMetadataKey].([]float32)
		if len(emb) == 0 {
			return fmt.Errorf("pgvector: document %q missing embedding", d.ID)
		}
		vec := pgvector.NewVector(emb)
		// Exclude embedding from JSONB to avoid storing it twice (vector column + metadata).
		metaCopy := make(map[string]any, len(d.Metadata))
		for k, v := range d.Metadata {
			if k == ragy.EmbeddingMetadataKey {
				continue
			}
			metaCopy[k] = v
		}
		metaJSON, _ := json.Marshal(metaCopy)
		batch.Queue(q, d.ID, d.Content, vec, metaJSON)
	}
	br := s.pool.SendBatch(ctx, batch)
	defer func() { _ = br.Close() }()
	for i := 0; i < len(docs); i++ {
		_, err := br.Exec()
		if err != nil {
			return err
		}
	}
	return nil
}

// DeleteByFilter implements ragy.VectorStore.
func (s *Store) DeleteByFilter(ctx context.Context, f filter.Expr) error {
	if f == nil {
		return nil
	}
	where, args, err := buildWhere(f, s.metaCol)
	if err != nil {
		return err
	}
	q := fmt.Sprintf("DELETE FROM %s WHERE %s", s.table, where)
	_, err = s.pool.Exec(ctx, q, args...)
	return err
}

var fieldSanitize = regexp.MustCompile(`^[a-zA-Z0-9_]+$`)

func buildWhere(expr filter.Expr, metaCol string) (clause string, args []any, err error) {
	if expr == nil {
		return "", nil, nil
	}
	return buildWhereRec(expr, 1, metaCol)
}

func buildWhereRec(expr filter.Expr, startArg int, metaCol string) (string, []any, error) {
	switch e := expr.(type) {
	case filter.Eq:
		if !fieldSanitize.MatchString(e.Field) {
			return "", nil, fmt.Errorf("pgvector: invalid field name %q", e.Field)
		}
		jb, err := json.Marshal(map[string]any{e.Field: e.Value})
		if err != nil {
			return "", nil, err
		}
		return fmt.Sprintf("%s @> $%d", metaCol, startArg), []any{jb}, nil
	case filter.Neq, filter.Gt, filter.Gte, filter.Lt, filter.Lte:
		field, val, op := "", any(nil), ""
		switch x := expr.(type) {
		case filter.Neq:
			field, val, op = x.Field, x.Value, "!="
		case filter.Gt:
			field, val, op = x.Field, x.Value, ">"
		case filter.Gte:
			field, val, op = x.Field, x.Value, ">="
		case filter.Lt:
			field, val, op = x.Field, x.Value, "<"
		case filter.Lte:
			field, val, op = x.Field, x.Value, "<="
		}
		if !fieldSanitize.MatchString(field) {
			return "", nil, fmt.Errorf("pgvector: invalid field name %q", field)
		}
		cast := "::text"
		switch val.(type) {
		case int, int32, int64, float32, float64:
			cast = "::numeric"
		case time.Time:
			cast = "::timestamptz"
		}
		return fmt.Sprintf("(%s->>'%s')%s %s $%d", metaCol, field, cast, op, startArg), []any{val}, nil
	case filter.In:
		if !fieldSanitize.MatchString(e.Field) {
			return "", nil, fmt.Errorf("pgvector: invalid field name %q", e.Field)
		}
		if len(e.Values) == 0 {
			return "false", nil, nil
		}
		placeholders := make([]string, len(e.Values))
		for i := range e.Values {
			placeholders[i] = fmt.Sprintf("$%d", startArg+i)
		}
		return fmt.Sprintf("(%s->>'%s') IN (%s)", metaCol, e.Field, joinParts(placeholders, ",")), e.Values, nil
	case filter.And:
		var parts []string
		var allArgs []any
		argIdx := startArg
		for _, sub := range e.Exprs {
			part, a, err := buildWhereRec(sub, argIdx, metaCol)
			if err != nil {
				return "", nil, err
			}
			parts = append(parts, "("+part+")")
			allArgs = append(allArgs, a...)
			argIdx += len(a)
		}
		return joinParts(parts, " AND "), allArgs, nil
	case filter.Or:
		var parts []string
		var allArgs []any
		argIdx := startArg
		for _, sub := range e.Exprs {
			part, a, err := buildWhereRec(sub, argIdx, metaCol)
			if err != nil {
				return "", nil, err
			}
			parts = append(parts, "("+part+")")
			allArgs = append(allArgs, a...)
			argIdx += len(a)
		}
		return joinParts(parts, " OR "), allArgs, nil
	case filter.Not:
		part, a, err := buildWhereRec(e.Expr, startArg, metaCol)
		if err != nil {
			return "", nil, err
		}
		return "NOT (" + part + ")", a, nil
	default:
		return "", nil, fmt.Errorf("pgvector: unsupported filter type %T", expr)
	}
}

func joinParts(parts []string, sep string) string {
	if len(parts) == 0 {
		return ""
	}
	s := parts[0]
	for i := 1; i < len(parts); i++ {
		s += sep + parts[i]
	}
	return s
}

// Ensure Store implements ragy.VectorStore.
var _ ragy.VectorStore = (*Store)(nil)
