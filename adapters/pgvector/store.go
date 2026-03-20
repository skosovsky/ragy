// Package pgvector provides a ragy.VectorStore implementation using PostgreSQL with the pgvector extension.
//
// Documents store the embedding in Metadata[ragy.EmbeddingMetadataKey] ([]float32). Use Batch Upsert and filter.Expr translation for WHERE.
//
// Table and column names (WithTable, WithMetadataColumn, etc.) must be set by the application and must not
// be taken from unvalidated user input; they are validated and passed through pgx.Identifier.Sanitize().
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
	"iter"
	"regexp"
	"strconv"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"
	pgvec "github.com/pgvector/pgvector-go"
	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

// DefaultUpsertBatchSize is the default micro-batch size for Upsert.
const DefaultUpsertBatchSize = 500

// DefaultRRFConstant is the default RRF rank constant k in 1/(k+rank).
const DefaultRRFConstant = 60

// Store implements ragy.VectorStore using pgxpool and pgvector.
type Store struct {
	pool       *pgxpool.Pool
	table      sanitizedIdent
	embedCol   sanitizedIdent
	idCol      sanitizedIdent
	contentCol sanitizedIdent
	metaCol    sanitizedIdent
	batchSize  int
	hybrid     bool
	ftsCol     sanitizedIdent
	lang       string
	rrfK int

	// upsertUnnestSQL is a single INSERT ... SELECT FROM unnest(...) ON CONFLICT ... built once in New.
	upsertUnnestSQL string
}

// Option configures the Store.
type Option func(*Store) error

// WithTable sets the table name (default "knowledge_base").
func WithTable(name string) Option {
	return func(s *Store) error {
		id, err := sanitizeIdent(name)
		if err != nil {
			return err
		}
		s.table = id
		return nil
	}
}

// WithEmbedColumn sets the embedding column name (default "embedding").
func WithEmbedColumn(name string) Option {
	return func(s *Store) error {
		id, err := sanitizeIdent(name)
		if err != nil {
			return err
		}
		s.embedCol = id
		return nil
	}
}

// WithIDColumn sets the id column name (default "id").
func WithIDColumn(name string) Option {
	return func(s *Store) error {
		id, err := sanitizeIdent(name)
		if err != nil {
			return err
		}
		s.idCol = id
		return nil
	}
}

// WithContentColumn sets the content column name (default "content").
func WithContentColumn(name string) Option {
	return func(s *Store) error {
		id, err := sanitizeIdent(name)
		if err != nil {
			return err
		}
		s.contentCol = id
		return nil
	}
}

// WithMetadataColumn sets the metadata JSONB column name (default "metadata").
func WithMetadataColumn(name string) Option {
	return func(s *Store) error {
		id, err := sanitizeIdent(name)
		if err != nil {
			return err
		}
		s.metaCol = id
		return nil
	}
}

// WithUpsertBatchSize sets the micro-batch size for Upsert (default 500).
func WithUpsertBatchSize(n int) Option {
	return func(s *Store) error {
		s.batchSize = n
		return nil
	}
}

// WithHybridSearch enables hybrid search (vector + FTS with RRF) when req.Query and req.DenseVector are both set.
func WithHybridSearch(enabled bool) Option {
	return func(s *Store) error {
		s.hybrid = enabled
		return nil
	}
}

// WithFTSColumn sets the column used for full-text search (default "content"). Must be a text column.
func WithFTSColumn(name string) Option {
	return func(s *Store) error {
		id, err := sanitizeIdent(name)
		if err != nil {
			return err
		}
		s.ftsCol = id
		return nil
	}
}

// WithLanguage sets the text search config (regconfig) for websearch_to_tsquery, e.g. "english", "russian", "simple" (default "english").
func WithLanguage(lang string) Option {
	return func(s *Store) error {
		s.lang = lang
		return nil
	}
}

// WithRRFConstant sets the RRF constant k used in 1/(k+rank) (default 60).
func WithRRFConstant(k int) Option {
	return func(s *Store) error {
		if k <= 0 {
			return fmt.Errorf("pgvector: RRF constant must be positive")
		}
		s.rrfK = k
		return nil
	}
}

// New returns a new pgvector Store. The pool must have pgvector types registered (e.g. pgxvec.RegisterTypes in AfterConnect).
func New(pool *pgxpool.Pool, opts ...Option) (*Store, error) {
	s := &Store{
		pool:       pool,
		batchSize:  DefaultUpsertBatchSize,
		lang:       "english",
		rrfK:       DefaultRRFConstant,
	}
	var err error
	s.table, err = sanitizeIdent("knowledge_base")
	if err != nil {
		return nil, err
	}
	s.embedCol, err = sanitizeIdent("embedding")
	if err != nil {
		return nil, err
	}
	s.idCol, err = sanitizeIdent("id")
	if err != nil {
		return nil, err
	}
	s.contentCol, err = sanitizeIdent("content")
	if err != nil {
		return nil, err
	}
	s.metaCol, err = sanitizeIdent("metadata")
	if err != nil {
		return nil, err
	}
	s.ftsCol, err = sanitizeIdent("content")
	if err != nil {
		return nil, err
	}
	for _, o := range opts {
		if err := o(s); err != nil {
			return nil, err
		}
	}
	s.upsertUnnestSQL = s.buildUpsertUnnestSQL()
	return s, nil
}

// buildUpsertUnnestSQL returns one INSERT ... SELECT FROM unnest($1::text[], $2::text[], $3::vector[], $4::jsonb[]) ... ON CONFLICT DO UPDATE.
func (s *Store) buildUpsertUnnestSQL() string {
	var b strings.Builder
	b.WriteString("INSERT INTO ")
	b.WriteString(string(s.table))
	b.WriteString(" (")
	b.WriteString(string(s.idCol))
	b.WriteString(", ")
	b.WriteString(string(s.contentCol))
	b.WriteString(", ")
	b.WriteString(string(s.embedCol))
	b.WriteString(", ")
	b.WriteString(string(s.metaCol))
	b.WriteString(") SELECT u.id, u.content, u.embedding, u.metadata FROM unnest($1::text[], $2::text[], $3::vector[], $4::jsonb[]) AS u(id, content, embedding, metadata) ON CONFLICT (")
	b.WriteString(string(s.idCol))
	b.WriteString(") DO UPDATE SET ")
	b.WriteString(string(s.contentCol))
	b.WriteString(" = EXCLUDED.")
	b.WriteString(string(s.contentCol))
	b.WriteString(", ")
	b.WriteString(string(s.embedCol))
	b.WriteString(" = EXCLUDED.")
	b.WriteString(string(s.embedCol))
	b.WriteString(", ")
	b.WriteString(string(s.metaCol))
	b.WriteString(" = EXCLUDED.")
	b.WriteString(string(s.metaCol))
	return b.String()
}

var fieldSanitize = regexp.MustCompile(`^[a-zA-Z0-9_]+$`)

// Search implements ragy.VectorStore.
func (s *Store) Search(ctx context.Context, req ragy.SearchRequest) ([]ragy.Document, error) {
	if len(req.DenseVector) == 0 {
		if len(req.SparseVector) > 0 {
			return nil, fmt.Errorf("pgvector: %w", ragy.ErrSparseVectorNotSupported)
		}
		return []ragy.Document{}, nil
	}
	vec := pgvec.NewVector(req.DenseVector)
	limit := req.Limit
	if limit <= 0 {
		limit = 10
	}
	offset := req.Offset
	if offset < 0 {
		offset = 0
	}

	if s.hybrid && req.Query != "" {
		return s.searchHybrid(ctx, vec, req.Query, limit, offset, req.Filter)
	}
	return s.searchVectorOnly(ctx, vec, limit, offset, req.Filter)
}

func (s *Store) searchVectorOnly(ctx context.Context, vec pgvec.Vector, limit, offset int, f filter.Expr) ([]ragy.Document, error) {
	v := NewSQLFilterVisitor(s.metaCol)
	whereSQL, whereArgs, err := v.ToSQL(f, 2)
	if err != nil {
		return nil, err
	}
	limitPh := 2 + len(whereArgs)
	offsetPh := limitPh + 1
	var b strings.Builder
	b.WriteString("SELECT ")
	b.WriteString(string(s.idCol))
	b.WriteString(", ")
	b.WriteString(string(s.contentCol))
	b.WriteString(", ")
	b.WriteString(string(s.metaCol))
	b.WriteString(", (")
	b.WriteString(string(s.embedCol))
	b.WriteString(" <=> $1) AS distance FROM ")
	b.WriteString(string(s.table))
	if whereSQL != "" {
		b.WriteString(" WHERE ")
		b.WriteString(whereSQL)
	}
	b.WriteString(" ORDER BY ")
	b.WriteString(string(s.embedCol))
	b.WriteString(" <=> $1 LIMIT $")
	b.WriteString(strconv.Itoa(limitPh))
	b.WriteString(" OFFSET $")
	b.WriteString(strconv.Itoa(offsetPh))

	args := []any{vec}
	args = append(args, whereArgs...)
	args = append(args, limit, offset)

	rows, err := s.pool.Query(ctx, b.String(), args...)
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
		conf := 1.0 - distance/2.0
		if conf < 0 {
			conf = 0
		}
		if conf > 1 {
			conf = 1
		}
		doc := ragy.Document{ID: id, Content: content, Metadata: meta, Score: float32(1.0 - distance), Confidence: conf}
		out = append(out, doc)
	}
	return out, rows.Err()
}

// hybridRRFConfidence maps the hybrid RRF fusion score to [0, 1]. Terms are 1/(k+rank); the
// theoretical maximum when both vector and FTS ranks are 1 is 2/(k+1), which we use as the scale.
func hybridRRFConfidence(score float64, rrfK int) float64 {
	if rrfK <= 0 {
		rrfK = DefaultRRFConstant
	}
	maxScore := 2.0 / float64(rrfK+1)
	if maxScore <= 0 {
		return 0
	}
	c := score / maxScore
	if c < 0 {
		return 0
	}
	if c > 1 {
		return 1
	}
	return c
}

func (s *Store) searchHybrid(ctx context.Context, vec pgvec.Vector, ftsQuery string, limit, offset int, f filter.Expr) ([]ragy.Document, error) {
	v := NewSQLFilterVisitor(s.metaCol)
	whereSQL, whereArgs, err := v.ToSQL(f, 4)
	if err != nil {
		return nil, err
	}
	var b strings.Builder
	rrf := strconv.Itoa(s.rrfK)

	b.WriteString("WITH vector_search AS ( SELECT ")
	b.WriteString(string(s.idCol))
	b.WriteString(", ")
	b.WriteString(string(s.contentCol))
	b.WriteString(", ")
	b.WriteString(string(s.metaCol))
	b.WriteString(", row_number() OVER (ORDER BY ")
	b.WriteString(string(s.embedCol))
	b.WriteString(" <=> $1) AS rank_v FROM ")
	b.WriteString(string(s.table))
	if whereSQL != "" {
		b.WriteString(" WHERE ")
		b.WriteString(whereSQL)
	}
	b.WriteString(" LIMIT 1000 ), fts_search AS ( SELECT ")
	b.WriteString(string(s.idCol))
	b.WriteString(", ")
	b.WriteString(string(s.contentCol))
	b.WriteString(", ")
	b.WriteString(string(s.metaCol))
	b.WriteString(", row_number() OVER (ORDER BY ts_rank_cd(to_tsvector($2::regconfig, ")
	b.WriteString(string(s.ftsCol))
	b.WriteString("), websearch_to_tsquery($2::regconfig, $3)) DESC NULLS LAST) AS rank_fts FROM ")
	b.WriteString(string(s.table))
	b.WriteString(" WHERE to_tsvector($2::regconfig, ")
	b.WriteString(string(s.ftsCol))
	b.WriteString(") @@ websearch_to_tsquery($2::regconfig, $3)")
	if whereSQL != "" {
		b.WriteString(" AND (")
		b.WriteString(whereSQL)
		b.WriteString(")")
	}
	b.WriteString(" LIMIT 1000 ) SELECT COALESCE(v.")
	b.WriteString(string(s.idCol))
	b.WriteString(", f.")
	b.WriteString(string(s.idCol))
	b.WriteString("), COALESCE(v.")
	b.WriteString(string(s.contentCol))
	b.WriteString(", f.")
	b.WriteString(string(s.contentCol))
	b.WriteString("), COALESCE(v.")
	b.WriteString(string(s.metaCol))
	b.WriteString(", f.")
	b.WriteString(string(s.metaCol))
	b.WriteString("), (COALESCE(1.0/(")
	b.WriteString(rrf)
	b.WriteString("+v.rank_v::float),0) + COALESCE(1.0/(")
	b.WriteString(rrf)
	b.WriteString("+f.rank_fts::float),0)) AS score FROM vector_search v FULL OUTER JOIN fts_search f ON v.")
	b.WriteString(string(s.idCol))
	b.WriteString(" = f.")
	b.WriteString(string(s.idCol))
	b.WriteString(" ORDER BY score DESC NULLS LAST LIMIT $")
	// $1 vec $2 lang $3 fts — then filter uses $4+; after filter come limit/offset
	nAfter := 3 + len(whereArgs)
	b.WriteString(strconv.Itoa(nAfter + 1))
	b.WriteString(" OFFSET $")
	b.WriteString(strconv.Itoa(nAfter + 2))

	args := []any{vec, s.lang, ftsQuery}
	args = append(args, whereArgs...)
	args = append(args, limit, offset)

	rows, err := s.pool.Query(ctx, b.String(), args...)
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
		doc.Confidence = hybridRRFConfidence(score, s.rrfK)
		out = append(out, doc)
	}
	return out, rows.Err()
}

// Stream implements ragy.VectorStore.
func (s *Store) Stream(ctx context.Context, req ragy.SearchRequest) iter.Seq2[ragy.Document, error] {
	docs, err := s.Search(ctx, req)
	return ragy.YieldDocuments(ctx, docs, err)
}

// Upsert implements ragy.VectorStore. Uses one UNNEST-based upsert per micro-batch (SQL prepared in New).
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
	ids := make([]string, len(docs))
	contents := make([]string, len(docs))
	vecs := make([]pgvec.Vector, len(docs))
	metas := make([][]byte, len(docs))
	for i, d := range docs {
		emb, _ := d.Metadata[ragy.EmbeddingMetadataKey].([]float32)
		if len(emb) == 0 {
			return fmt.Errorf("pgvector: document %q missing embedding", d.ID)
		}
		metaCopy := make(map[string]any, len(d.Metadata))
		for k, v := range d.Metadata {
			if k == ragy.EmbeddingMetadataKey {
				continue
			}
			metaCopy[k] = v
		}
		metaJSON, err := json.Marshal(metaCopy)
		if err != nil {
			return err
		}
		ids[i] = d.ID
		contents[i] = d.Content
		vecs[i] = pgvec.NewVector(emb)
		metas[i] = metaJSON
	}
	_, err := s.pool.Exec(ctx, s.upsertUnnestSQL, ids, contents, vecs, metas)
	return err
}

// FetchParents implements ragy.HierarchyRetriever. Loads child rows by id, reads metadata[ragy.ParentDocumentIDKey], then loads parent rows by those IDs.
func (s *Store) FetchParents(ctx context.Context, childIDs []string) ([]ragy.Document, error) {
	if len(childIDs) == 0 {
		return nil, nil
	}
	var b strings.Builder
	b.WriteString("SELECT ")
	b.WriteString(string(s.idCol))
	b.WriteString(", ")
	b.WriteString(string(s.contentCol))
	b.WriteString(", ")
	b.WriteString(string(s.metaCol))
	b.WriteString(" FROM ")
	b.WriteString(string(s.table))
	b.WriteString(" WHERE ")
	b.WriteString(string(s.idCol))
	b.WriteString(" = ANY($1::text[])")
	rows, err := s.pool.Query(ctx, b.String(), childIDs)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var children []ragy.Document
	for rows.Next() {
		var id, content string
		var metaJSON []byte
		if err := rows.Scan(&id, &content, &metaJSON); err != nil {
			return nil, err
		}
		meta := make(map[string]any)
		if len(metaJSON) > 0 {
			if err := json.Unmarshal(metaJSON, &meta); err != nil {
				return nil, err
			}
		}
		children = append(children, ragy.Document{ID: id, Content: content, Metadata: meta})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	parentSeen := make(map[string]struct{})
	var parentIDs []string
	for _, ch := range children {
		raw, ok := ch.Metadata[ragy.ParentDocumentIDKey]
		if !ok || raw == nil {
			continue
		}
		pid, ok := raw.(string)
		if !ok || pid == "" {
			continue
		}
		if _, dup := parentSeen[pid]; dup {
			continue
		}
		parentSeen[pid] = struct{}{}
		parentIDs = append(parentIDs, pid)
	}
	if len(parentIDs) == 0 {
		return nil, nil
	}
	var b2 strings.Builder
	b2.WriteString("SELECT ")
	b2.WriteString(string(s.idCol))
	b2.WriteString(", ")
	b2.WriteString(string(s.contentCol))
	b2.WriteString(", ")
	b2.WriteString(string(s.metaCol))
	b2.WriteString(" FROM ")
	b2.WriteString(string(s.table))
	b2.WriteString(" WHERE ")
	b2.WriteString(string(s.idCol))
	b2.WriteString(" = ANY($1::text[])")
	rows2, err := s.pool.Query(ctx, b2.String(), parentIDs)
	if err != nil {
		return nil, err
	}
	defer rows2.Close()
	var out []ragy.Document
	for rows2.Next() {
		var id, content string
		var metaJSON []byte
		if err := rows2.Scan(&id, &content, &metaJSON); err != nil {
			return nil, err
		}
		meta := make(map[string]any)
		if len(metaJSON) > 0 {
			if err := json.Unmarshal(metaJSON, &meta); err != nil {
				return nil, err
			}
		}
		out = append(out, ragy.Document{ID: id, Content: content, Metadata: meta})
	}
	return out, rows2.Err()
}

// DeleteByFilter implements ragy.VectorStore.
func (s *Store) DeleteByFilter(ctx context.Context, f filter.Expr) error {
	if f == nil {
		return nil
	}
	v := NewSQLFilterVisitor(s.metaCol)
	whereSQL, whereArgs, err := v.ToSQL(f, 1)
	if err != nil {
		return err
	}
	var b strings.Builder
	b.WriteString("DELETE FROM ")
	b.WriteString(string(s.table))
	b.WriteString(" WHERE ")
	b.WriteString(whereSQL)
	_, err = s.pool.Exec(ctx, b.String(), whereArgs...)
	return err
}

var (
	_ ragy.VectorStore         = (*Store)(nil)
	_ ragy.HierarchyRetriever = (*Store)(nil)
)
