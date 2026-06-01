package pgvector

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/documents"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/internal/contracttest"
	"github.com/skosovsky/ragy/retrieval"
)

type fakeRow struct {
	id        string
	content   string
	attrsJSON []byte
	relevance float64
}

type fakeRows struct {
	rows  []fakeRow
	index int
}

func (r *fakeRows) Next() bool {
	if r.index >= len(r.rows) {
		return false
	}

	r.index++
	return true
}

func (r *fakeRows) Scan(dest ...any) error {
	if r.index == 0 || r.index > len(r.rows) {
		return errors.New("scan called out of bounds")
	}

	row := r.rows[r.index-1]
	switch len(dest) {
	case 4:
		*(dest[0].(*string)) = row.id
		*(dest[1].(*string)) = row.content
		*(dest[2].(*[]byte)) = append([]byte(nil), row.attrsJSON...)
		*(dest[3].(*float64)) = row.relevance
	case 3:
		*(dest[0].(*string)) = row.id
		*(dest[1].(*string)) = row.content
		*(dest[2].(*[]byte)) = append([]byte(nil), row.attrsJSON...)
	default:
		return fmt.Errorf("unexpected scan arity %d", len(dest))
	}

	return nil
}

func (r *fakeRows) Err() error   { return nil }
func (r *fakeRows) Close() error { return nil }

type fakeResult struct{ rows int64 }

func (r fakeResult) RowsAffected() int64 { return r.rows }

type fakeDB struct {
	query     string
	args      []any
	queryRows Rows
	execSQL   string
	execArgs  []any
	execCalls int
}

func (db *fakeDB) Query(_ context.Context, sql string, args ...any) (Rows, error) {
	db.query = sql
	db.args = args
	if db.queryRows != nil {
		return db.queryRows, nil
	}
	return &fakeRows{}, nil
}

func (db *fakeDB) Exec(_ context.Context, sql string, args ...any) (Result, error) {
	db.execSQL = sql
	db.execArgs = append([]any(nil), args...)
	db.execCalls++
	return fakeResult{}, nil
}

func emptySchema(t *testing.T) filter.Schema {
	t.Helper()

	schema, err := filter.NewSchema().Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	return schema
}

func tenantSchema(t *testing.T) filter.Schema {
	t.Helper()

	builder := filter.NewSchema()
	if _, err := builder.String("tenant"); err != nil {
		t.Fatalf("builder.String(): %v", err)
	}

	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	return schema
}

func ageSchema(t *testing.T) filter.Schema {
	t.Helper()

	builder := filter.NewSchema()
	if _, err := builder.Int("age"); err != nil {
		t.Fatalf("builder.Int(): %v", err)
	}

	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	return schema
}

func ageScoreSchema(t *testing.T) filter.Schema {
	t.Helper()

	builder := filter.NewSchema()
	if _, err := builder.Int("age"); err != nil {
		t.Fatalf("builder.Int(): %v", err)
	}
	if _, err := builder.Float("score"); err != nil {
		t.Fatalf("builder.Float(): %v", err)
	}

	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	return schema
}

func TestRenderSearchUsesFetchLimit(t *testing.T) {
	t.Parallel()

	store := &Store[contracttest.Meta]{table: "docs", schema: emptySchema(t)}
	sql, args, err := store.renderSearch(retrieval.RetrieveOptions{
		Vector:     []float32{1},
		FetchLimit: 50,
		TopK:       10,
	})
	if err != nil {
		t.Fatalf("renderSearch(): %v", err)
	}
	if !containsLimit(sql) {
		t.Fatalf("query = %q, want LIMIT clause", sql)
	}
	if len(args) != 2 || args[1] != 50 {
		t.Fatalf("args = %#v, want vector and fetch_limit 50", args)
	}
}

func TestRenderSearchFallsBackToTopKWhenFetchLimitZero(t *testing.T) {
	t.Parallel()

	store := &Store[contracttest.Meta]{table: "docs", schema: emptySchema(t)}
	sql, args, err := store.renderSearch(retrieval.RetrieveOptions{
		Vector: []float32{1},
		TopK:   15,
	})
	if err != nil {
		t.Fatalf("renderSearch(): %v", err)
	}
	if !containsLimit(sql) {
		t.Fatalf("query = %q, want LIMIT clause", sql)
	}
	if len(args) != 2 || args[1] != 15 {
		t.Fatalf("args = %#v, want vector and top_k fallback 15", args)
	}
}

func TestRetrieveHasNoImplicitLimitAndReturnsNilOnNoRows(t *testing.T) {
	db := &fakeDB{}
	store, err := New[contracttest.Meta](db, Config{Table: "docs", Schema: emptySchema(t)})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
		Vector: []float32{1, 0},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}

	if out != nil {
		t.Fatalf("Retrieve() out = %#v, want nil", out)
	}

	if got := db.query; got == "" || containsLimit(got) {
		t.Fatalf("query = %q, want no LIMIT clause", got)
	}
}

func TestDenseIndexConformance(t *testing.T) {
	contracttest.RunDenseIndexSuite(t, func(t *testing.T) dense.Index[contracttest.Meta] {
		t.Helper()
		store, err := New[contracttest.Meta](&fakeDB{}, Config{
			Table:  "docs",
			Schema: contracttest.TenantAgeSchema(t),
		})
		if err != nil {
			t.Fatalf("New(): %v", err)
		}
		return store
	})
}

func containsLimit(query string) bool {
	return strings.Contains(query, " LIMIT ") || strings.Contains(query, " OFFSET ")
}

func TestUpsertValidatesRecord(t *testing.T) {
	db := &fakeDB{}
	store, err := New[contracttest.Meta](db, Config{Table: "docs", Schema: emptySchema(t)})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	if err := store.Upsert(context.Background(), []dense.Record[contracttest.Meta]{{Content: "broken"}}); err == nil {
		t.Fatal("Upsert() error = nil, want error")
	}
}

func TestRetrieveProjectsCanonicalShape(t *testing.T) {
	db := &fakeDB{
		queryRows: &fakeRows{
			rows: []fakeRow{{
				id:        "doc-1",
				content:   "hello",
				attrsJSON: []byte("{}"),
				relevance: 0.75,
			}},
		},
	}

	store, err := New[contracttest.Meta](db, Config{Table: "docs", Schema: emptySchema(t)})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
		Vector: []float32{1, 0},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}

	if len(out) != 1 {
		t.Fatalf("len(out) = %d, want 1", len(out))
	}

	if out[0].Score != 0.75 {
		t.Fatalf("out[0].Score = %f, want 0.75", out[0].Score)
	}

	if len(out[0].Meta) != 0 {
		t.Fatalf("out[0].Meta = %#v, want empty", out[0].Meta)
	}

	if !strings.Contains(db.query, "1 / (1 + (vector <=> $1)) AS relevance") {
		t.Fatalf("query = %q, want similarity expression", db.query)
	}
}

func TestRetrieveRejectsInvalidBackendPayload(t *testing.T) {
	db := &fakeDB{
		queryRows: &fakeRows{
			rows: []fakeRow{{
				id:        "",
				content:   "broken",
				attrsJSON: nil,
				relevance: 0.75,
			}},
		},
	}

	store, err := New[contracttest.Meta](db, Config{Table: "docs", Schema: emptySchema(t)})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	if _, err := store.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
		Vector: []float32{1, 0},
	}); err == nil {
		t.Fatal("Retrieve() error = nil, want error")
	}
}

func TestFindByIDsRejectsInvalidBackendPayload(t *testing.T) {
	db := &fakeDB{
		queryRows: &fakeRows{
			rows: []fakeRow{{
				id:      "",
				content: "broken",
			}},
		},
	}

	store, err := New[contracttest.Meta](db, Config{Table: "docs", Schema: emptySchema(t)})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	if _, err := store.FindByIDs(context.Background(), []string{"doc-1"}); err == nil {
		t.Fatal("FindByIDs() error = nil, want error")
	}
}

func TestDeleteByFilterRejectsEmpty(t *testing.T) {
	db := &fakeDB{}
	store, err := New[contracttest.Meta](db, Config{Table: "docs", Schema: emptySchema(t)})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	if _, err := store.DeleteByFilter(context.Background(), filter.Condition{}); err == nil {
		t.Fatal("DeleteByFilter() error = nil, want error")
	}

	if db.execCalls != 0 {
		t.Fatalf("execCalls = %d, want 0", db.execCalls)
	}
}

func TestNewRejectsInvalidSQLIdentifier(t *testing.T) {
	if _, err := New[contracttest.Meta](&fakeDB{}, Config{Table: "1bad", Schema: emptySchema(t)}); err == nil {
		t.Fatal("New() error = nil, want error")
	}
}

func TestRetrieveRendersAttributeFilterAgainstAttributesJSON(t *testing.T) {
	db := &fakeDB{}
	builder := filter.NewSchema()
	tenant, err := builder.String("tenant")
	if err != nil {
		t.Fatalf("builder.String(tenant): %v", err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}
	store, err := New[contracttest.Meta](db, Config{Table: "docs", Schema: schema})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	filterBuilder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatalf("NewBuilder(): %v", err)
	}
	cond, err := filter.Eq(filterBuilder, tenant, "acme").Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	_, err = store.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
		Vector:  []float32{1},
		Filters: cond,
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}

	if !strings.Contains(db.query, "attributes->>'tenant' = $2") {
		t.Fatalf("query = %q, want attributes JSON filter", db.query)
	}
}

func TestUpsertRejectsWrongMetaTypeBeforeExec(t *testing.T) {
	db := &fakeDB{}
	store, err := New[contracttest.Meta](db, Config{Table: "docs", Schema: ageSchema(t)})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	err = store.Upsert(context.Background(), []dense.Record[contracttest.Meta]{{
		ID:      "doc-1",
		Content: "hello",
		Vector:  []float32{1},
		Meta:    contracttest.Meta{"age": "old"},
	}})
	if err == nil {
		t.Fatal("Upsert() error = nil, want error")
	}
	if db.execCalls != 0 {
		t.Fatalf("execCalls = %d, want 0", db.execCalls)
	}
}

func TestUpsertCanonicalizesMetaBeforeExec(t *testing.T) {
	db := &fakeDB{}
	store, err := New[contracttest.Meta](db, Config{Table: "docs", Schema: ageScoreSchema(t)})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	err = store.Upsert(context.Background(), []dense.Record[contracttest.Meta]{{
		ID:      "doc-1",
		Content: "hello",
		Vector:  []float32{1},
		Meta: contracttest.Meta{
			"age":   int(7),
			"score": float32(1.5),
		},
	}})
	if err != nil {
		t.Fatalf("Upsert(): %v", err)
	}

	if db.execCalls != 1 {
		t.Fatalf("execCalls = %d, want 1", db.execCalls)
	}

	attrsJSON, ok := db.execArgs[2].([]byte)
	if !ok {
		t.Fatalf("execArgs[2] type = %T, want []byte", db.execArgs[2])
	}

	var attrs map[string]json.Number
	if err := json.Unmarshal(attrsJSON, &attrs); err != nil {
		t.Fatalf("Unmarshal(attrsJSON): %v", err)
	}

	if got := attrs["age"].String(); got != "7" {
		t.Fatalf("age JSON = %q, want 7", got)
	}
	if got := attrs["score"].String(); got != "1.5" {
		t.Fatalf("score JSON = %q, want 1.5", got)
	}
}

func TestUpsertCanonicalizesEmptyMetaToNullJSON(t *testing.T) {
	db := &fakeDB{}
	store, err := New[contracttest.Meta](db, Config{Table: "docs", Schema: emptySchema(t)})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	err = store.Upsert(context.Background(), []dense.Record[contracttest.Meta]{{
		ID:      "doc-1",
		Content: "hello",
		Vector:  []float32{1},
	}})
	if err != nil {
		t.Fatalf("Upsert(): %v", err)
	}

	attrsJSON, ok := db.execArgs[2].([]byte)
	if !ok {
		t.Fatalf("execArgs[2] type = %T, want []byte", db.execArgs[2])
	}
	if string(attrsJSON) != "null" {
		t.Fatalf("attrs JSON = %q, want null", string(attrsJSON))
	}
}

func TestRetrieveRejectsUndeclaredFilterFieldBeforeQuery(t *testing.T) {
	db := &fakeDB{}
	store, err := New[contracttest.Meta](db, Config{Table: "docs", Schema: emptySchema(t)})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	foreign := filter.NewSchema()
	tenant, err := foreign.String("other_tenant")
	if err != nil {
		t.Fatalf("foreign.String(): %v", err)
	}
	foreignSchema, err := foreign.Build()
	if err != nil {
		t.Fatalf("foreign.Build(): %v", err)
	}
	filterBuilder, err := filter.NewBuilder(foreignSchema)
	if err != nil {
		t.Fatalf("NewBuilder(): %v", err)
	}
	cond, err := filter.Eq(filterBuilder, tenant, "acme").Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	_, err = store.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
		Vector:  []float32{1},
		Filters: cond,
	})
	if err == nil {
		t.Fatal("Retrieve() error = nil, want error")
	}
	if db.query != "" {
		t.Fatalf("query = %q, want no query execution", db.query)
	}
}

type documentsDB struct {
	docs map[string]retrieval.Document[contracttest.Meta]
}

func newDocumentsDB(docs []retrieval.Document[contracttest.Meta]) *documentsDB {
	out := make(map[string]retrieval.Document[contracttest.Meta], len(docs))
	for _, doc := range docs {
		out[doc.ID] = retrieval.Document[contracttest.Meta]{
			ID:      doc.ID,
			Content: doc.Content,
			Meta:    doc.Meta,
			Score:   doc.Score,
		}
	}

	return &documentsDB{docs: out}
}

func (db *documentsDB) Query(_ context.Context, sql string, args ...any) (Rows, error) {
	switch {
	case strings.Contains(sql, "SELECT id, content, attributes FROM docs WHERE id IN"):
		rows := make([]fakeRow, 0, len(args))
		for _, arg := range args {
			id, ok := arg.(string)
			if !ok {
				return nil, fmt.Errorf("unexpected id arg type %T", arg)
			}
			doc, ok := db.docs[id]
			if !ok {
				continue
			}

			attrs, err := json.Marshal(doc.Meta)
			if err != nil {
				return nil, err
			}
			rows = append(rows, fakeRow{
				id:        doc.ID,
				content:   doc.Content,
				attrsJSON: attrs,
			})
		}
		return &fakeRows{rows: rows}, nil
	default:
		return nil, fmt.Errorf("unexpected query %q", sql)
	}
}

func (db *documentsDB) Exec(_ context.Context, sql string, args ...any) (Result, error) {
	switch {
	case strings.Contains(sql, "DELETE FROM docs WHERE attributes->>'tenant' = $1"):
		tenant, ok := args[0].(string)
		if !ok {
			return nil, fmt.Errorf("unexpected tenant arg type %T", args[0])
		}
		deleted := int64(0)
		for id, doc := range db.docs {
			if value, ok := doc.Meta["tenant"].(string); ok && value == tenant {
				delete(db.docs, id)
				deleted++
			}
		}
		return fakeResult{rows: deleted}, nil
	default:
		return nil, fmt.Errorf("unexpected exec %q", sql)
	}
}

func TestDocumentsStoreConformance(t *testing.T) {
	contracttest.RunDocumentsStoreSuite(
		t,
		func(t *testing.T, docs []retrieval.Document[contracttest.Meta]) documents.Store[contracttest.Meta] {
			t.Helper()

			store, err := New[contracttest.Meta](newDocumentsDB(docs), Config{Table: "docs", Schema: tenantSchema(t)})
			if err != nil {
				t.Fatalf("New(): %v", err)
			}

			return store
		},
	)
}

type tenantMeta struct {
	Tenant string `json:"tenant"`
}

func TestRetrieveUnmarshalsStructMeta(t *testing.T) {
	t.Parallel()

	db := &fakeDB{
		queryRows: &fakeRows{
			rows: []fakeRow{{
				id:        "doc-1",
				content:   "hello",
				attrsJSON: []byte(`{"tenant":"acme"}`),
				relevance: 0.8,
			}},
		},
	}

	store, err := New[tenantMeta](db, Config{Table: "docs", Schema: tenantSchema(t)})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
		Vector: []float32{1},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if len(out) != 1 || out[0].Meta.Tenant != "acme" {
		t.Fatalf("Retrieve() = %#v, want tenant acme", out)
	}
}

func TestRetrieveRejectsIncompatibleStructMeta(t *testing.T) {
	t.Parallel()

	db := &fakeDB{
		queryRows: &fakeRows{
			rows: []fakeRow{{
				id:        "doc-1",
				content:   "hello",
				attrsJSON: []byte(`{"tenant":123}`),
				relevance: 0.8,
			}},
		},
	}

	store, err := New[tenantMeta](db, Config{Table: "docs", Schema: tenantSchema(t)})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	_, err = store.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
		Vector: []float32{1},
	})
	if err == nil {
		t.Fatal("Retrieve() error = nil, want decode error")
	}
}

func TestFindByIDsUnmarshalsStructMeta(t *testing.T) {
	t.Parallel()

	db := &fakeDB{
		queryRows: &fakeRows{
			rows: []fakeRow{{
				id:        "doc-1",
				content:   "hello",
				attrsJSON: []byte(`{"tenant":"acme"}`),
			}},
		},
	}

	store, err := New[tenantMeta](db, Config{Table: "docs", Schema: tenantSchema(t)})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.FindByIDs(context.Background(), []string{"doc-1"})
	if err != nil {
		t.Fatalf("FindByIDs(): %v", err)
	}
	if len(out) != 1 || out[0].Meta.Tenant != "acme" {
		t.Fatalf("FindByIDs() = %#v, want tenant acme", out)
	}
}

func TestFindByIDsRejectsIncompatibleStructMeta(t *testing.T) {
	t.Parallel()

	db := &fakeDB{
		queryRows: &fakeRows{
			rows: []fakeRow{{
				id:        "doc-1",
				content:   "hello",
				attrsJSON: []byte(`{"tenant":123}`),
			}},
		},
	}

	store, err := New[tenantMeta](db, Config{Table: "docs", Schema: tenantSchema(t)})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	_, err = store.FindByIDs(context.Background(), []string{"doc-1"})
	if err == nil {
		t.Fatal("FindByIDs() error = nil, want decode error")
	}
}
