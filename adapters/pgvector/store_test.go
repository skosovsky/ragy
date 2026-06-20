package pgvector

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/documents"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/internal/contracttest"
	"github.com/skosovsky/ragy/retrieval"
)

func retrieveStore[TMeta any](
	ctx context.Context,
	store *Store[TMeta],
	text string,
	opts retrieval.RetrieveOptions,
) (retrieval.ResultSet[TMeta], error) {
	return store.Retrieve(ctx, retrieval.Query[struct{}]{Text: text, Options: opts})
}

type fakeRow struct {
	id        string
	content   string
	attrsJSON []byte
	relevance float64
	scanErr   error
}

type fakeRows struct {
	rows    []fakeRow
	index   int
	rowsErr error
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
	if row.scanErr != nil {
		return row.scanErr
	}
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

func (r *fakeRows) Err() error {
	if r.rowsErr != nil {
		return r.rowsErr
	}
	return nil
}
func (r *fakeRows) Close() error { return nil }

type fakeResult struct{ rows int64 }

func (r fakeResult) RowsAffected() int64 { return r.rows }

type fakeDB struct {
	query     string
	args      []any
	queryRows Rows
	queryErr  error
	execSQL   string
	execArgs  []any
	execCalls int
	execErr   error
}

func (db *fakeDB) Query(_ context.Context, sql string, args ...any) (Rows, error) {
	db.query = sql
	db.args = args
	if db.queryErr != nil {
		return nil, db.queryErr
	}
	if db.queryRows != nil {
		return db.queryRows, nil
	}
	return &fakeRows{}, nil
}

func (db *fakeDB) Exec(_ context.Context, sql string, args ...any) (Result, error) {
	db.execSQL = sql
	db.execArgs = append([]any(nil), args...)
	db.execCalls++
	if db.execErr != nil {
		return nil, db.execErr
	}
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

func newStore(t *testing.T, db DB, schema filter.Schema) *Store[contracttest.StructMeta] {
	t.Helper()
	codec := contracttest.JSONCodec[contracttest.StructMeta](t, schema)
	store, err := New(db, Config[contracttest.StructMeta]{Table: "docs", Schema: schema}, codec)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	return store
}

func newStoreEmptySchema(t *testing.T, db DB) *Store[contracttest.StructMeta] {
	return newStore(t, db, emptySchema(t))
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

	store := &Store[contracttest.StructMeta]{table: "docs", schema: emptySchema(t)}
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

	store := &Store[contracttest.StructMeta]{table: "docs", schema: emptySchema(t)}
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

func TestRetrieveRejectsZeroTopKAndFetchLimit(t *testing.T) {
	db := &fakeDB{}
	store := newStoreEmptySchema(t, db)

	out, err := retrieveStore(context.Background(), store, "", retrieval.RetrieveOptions{
		Vector: []float32{1, 0},
	})
	contracttest.RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Retrieve() error = %v, want invalid argument", err)
	}
}

func TestRetrieveReturnsEmptyWithValidTopK(t *testing.T) {
	db := &fakeDB{}
	store := newStoreEmptySchema(t, db)

	out, err := retrieveStore(context.Background(), store, "", retrieval.RetrieveOptions{
		Vector: []float32{1, 0},
		TopK:   10,
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if out == nil {
		t.Fatal("Retrieve() out = nil, want non-nil empty ResultSet")
	}
	if !out.IsEmpty() {
		t.Fatalf("Retrieve() out = %#v, want empty", out.Documents())
	}
	if got := db.query; got == "" || !containsLimit(got) {
		t.Fatalf("query = %q, want LIMIT clause", got)
	}
}

func TestDecodeStoredMetaWrapsCorruptJSON(t *testing.T) {
	db := &fakeDB{
		queryRows: &fakeRows{
			rows: []fakeRow{
				{id: "bad", content: "bad", attrsJSON: []byte(`{`), relevance: 0.5},
			},
		},
	}
	store := newStore(t, db, tenantSchema(t))

	out, err := retrieveStore(context.Background(), store, "", retrieval.RetrieveOptions{
		Vector: []float32{1, 0},
		TopK:   5,
	})
	if err == nil {
		t.Fatal("Retrieve() error = nil, want projection error")
	}
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Retrieve() error = %v, want protocol", err)
	}
	if out == nil || !out.IsEmpty() {
		t.Fatalf("Retrieve() out = %#v, want empty on error", out)
	}
}

func TestDenseIndexConformance(t *testing.T) {
	contracttest.RunDenseIndexSuite(t, func(t *testing.T) dense.Index[contracttest.StructMeta] {
		t.Helper()
		schema := contracttest.TenantAgeSchema(t)
		store, err := New[contracttest.StructMeta](&fakeDB{}, Config[contracttest.StructMeta]{
			Table:  "docs",
			Schema: schema,
		}, contracttest.JSONCodec[contracttest.StructMeta](t, schema))
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
	store := newStoreEmptySchema(t, db)

	if err := store.Upsert(
		context.Background(),
		[]dense.Record[contracttest.StructMeta]{{Content: "broken"}},
	); err == nil {
		t.Fatal("Upsert() error = nil, want error")
	} else if !errors.Is(err, ragy.ErrMissingID) {
		t.Fatalf("Upsert() error = %v, want missing id", err)
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

	store := newStoreEmptySchema(t, db)

	out, err := retrieveStore(context.Background(), store, "", retrieval.RetrieveOptions{
		Vector: []float32{1, 0},
		TopK:   10,
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}

	docs := out.Documents()
	if len(docs) != 1 {
		t.Fatalf("len(docs) = %d, want 1", len(docs))
	}

	if docs[0].Score != 0.75 {
		t.Fatalf("docs[0].Score = %f, want 0.75", docs[0].Score)
	}

	if docs[0].Meta != (contracttest.StructMeta{}) {
		t.Fatalf("docs[0].Meta = %#v, want empty", docs[0].Meta)
	}

	if !strings.Contains(db.query, "1 / (1 + (vector <=> $1)) AS relevance") {
		t.Fatalf("query = %q, want similarity expression", db.query)
	}
}

func TestNewRejectsInvalidSQLIdentifier(t *testing.T) {
	if _, err := New(
		&fakeDB{},
		Config[contracttest.StructMeta]{Table: "1bad", Schema: emptySchema(t)},
		contracttest.JSONCodec[contracttest.StructMeta](t, emptySchema(t)),
	); err == nil {
		t.Fatal("New() error = nil, want error")
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("New() error = %v, want invalid argument", err)
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
	store, err := New[contracttest.StructMeta](
		db,
		Config[contracttest.StructMeta]{Table: "docs", Schema: schema},
		retrieval.NewJSONCodec[contracttest.StructMeta](schema),
	)
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

	_, err = retrieveStore(context.Background(), store, "", retrieval.RetrieveOptions{
		Vector:  []float32{1},
		TopK:    10,
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
	store, err := New[contracttest.StructMeta](
		db,
		Config[contracttest.StructMeta]{Table: "docs", Schema: ageSchema(t)},
		contracttest.JSONCodec[contracttest.StructMeta](t, ageSchema(t)),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	err = store.Upsert(context.Background(), []dense.Record[contracttest.StructMeta]{{
		ID:      "doc-1",
		Content: "hello",
		Vector:  []float32{1},
		Meta:    contracttest.StructMeta{Tenant: "acme"},
	}})
	if err == nil {
		t.Fatal("Upsert() error = nil, want error")
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Upsert() error = %v, want invalid argument", err)
	}
	if db.execCalls != 0 {
		t.Fatalf("execCalls = %d, want 0", db.execCalls)
	}
}

func TestUpsertCanonicalizesMetaBeforeExec(t *testing.T) {
	db := &fakeDB{}
	store, err := New[contracttest.StructMeta](
		db,
		Config[contracttest.StructMeta]{Table: "docs", Schema: ageScoreSchema(t)},
		contracttest.JSONCodec[contracttest.StructMeta](t, ageScoreSchema(t)),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	err = store.Upsert(context.Background(), []dense.Record[contracttest.StructMeta]{{
		ID:      "doc-1",
		Content: "hello",
		Vector:  []float32{1},
		Meta: contracttest.StructMeta{
			Age:   7,
			Score: 1.5,
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
	store := newStoreEmptySchema(t, db)

	err := store.Upsert(context.Background(), []dense.Record[contracttest.StructMeta]{{
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

func TestUpsertFindByIDsRoundTrip(t *testing.T) {
	t.Parallel()

	db := &fakeDB{}
	schema := tenantSchema(t)
	store, err := New[contracttest.TenantOnlyMeta](
		db,
		Config[contracttest.TenantOnlyMeta]{Table: "docs", Schema: schema},
		contracttest.JSONCodec[contracttest.TenantOnlyMeta](t, schema),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	err = store.Upsert(context.Background(), []dense.Record[contracttest.TenantOnlyMeta]{{
		ID:      "doc-1",
		Content: "hello",
		Vector:  []float32{1},
		Meta:    contracttest.TenantOnlyMeta{Tenant: "acme"},
	}})
	if err != nil {
		t.Fatalf("Upsert(): %v", err)
	}

	attrsJSON, ok := db.execArgs[2].([]byte)
	if !ok {
		t.Fatalf("execArgs[2] type = %T, want []byte", db.execArgs[2])
	}
	db.queryRows = &fakeRows{
		rows: []fakeRow{{
			id:        "doc-1",
			content:   "hello",
			attrsJSON: attrsJSON,
		}},
	}

	out, err := store.FindByIDs(context.Background(), []string{"doc-1"})
	if err != nil {
		t.Fatalf("FindByIDs(): %v", err)
	}
	if len(out) != 1 || out[0].Meta.Tenant != "acme" || out[0].Content != "hello" {
		t.Fatalf("FindByIDs() = %#v, want round-trip doc", out)
	}
}

func TestRetrieveEmptyVectorReturnsNonNilResultSet(t *testing.T) {
	t.Parallel()

	store := newStoreEmptySchema(t, &fakeDB{})
	out, err := retrieveStore(context.Background(), store, "q", retrieval.RetrieveOptions{TopK: 1})
	contracttest.RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrEmptyVector) {
		t.Fatalf("Retrieve() error = %v, want empty vector", err)
	}
}

func TestRetrieveRejectsUndeclaredFilterFieldBeforeQuery(t *testing.T) {
	db := &fakeDB{}
	store := newStoreEmptySchema(t, db)

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

	out, err := retrieveStore(context.Background(), store, "", retrieval.RetrieveOptions{
		Vector:  []float32{1},
		TopK:    10,
		Filters: cond,
	})
	contracttest.RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Retrieve() error = %v, want invalid argument", err)
	}
	if db.query != "" {
		t.Fatalf("query = %q, want no query execution", db.query)
	}
}

type documentsDB struct {
	docs map[string]retrieval.Document[contracttest.StructMeta]
}

func newDocumentsDB(docs []retrieval.Document[contracttest.StructMeta]) *documentsDB {
	out := make(map[string]retrieval.Document[contracttest.StructMeta], len(docs))
	for _, doc := range docs {
		out[doc.ID] = retrieval.Document[contracttest.StructMeta]{
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
			if doc.Meta.Tenant == tenant {
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
	contracttest.RunDocumentsStructStoreSuite(
		t,
		func(t *testing.T, docs []retrieval.Document[contracttest.StructMeta]) documents.Store[contracttest.StructMeta] {
			t.Helper()

			schema := tenantSchema(t)
			store, err := New[contracttest.StructMeta](
				newDocumentsDB(docs),
				Config[contracttest.StructMeta]{Table: "docs", Schema: schema},
				contracttest.JSONCodec[contracttest.StructMeta](t, schema),
			)
			if err != nil {
				t.Fatalf("New(): %v", err)
			}

			return store
		},
	)
}

func TestDocumentsPartialFindByIDsConformance(t *testing.T) {
	contracttest.RunDocumentsPartialFindByIDsSuite(t, func(t *testing.T) documents.Store[contracttest.StructMeta] {
		t.Helper()

		db := &fakeDB{
			queryRows: &fakeRows{
				rows: []fakeRow{
					{id: "ok", content: "good", attrsJSON: []byte(`{"tenant":"acme"}`)},
					{id: "bad", content: "bad", attrsJSON: []byte(`{`)},
				},
			},
		}
		return newStore(t, db, tenantSchema(t))
	})
}

func TestRetrievePartialProjectionConformance(t *testing.T) {
	contracttest.RunRetrievePartialProjectionSuite(
		t,
		func(t *testing.T) retrieval.Backend[struct{}, contracttest.StructMeta] {
			t.Helper()

			db := &fakeDB{
				queryRows: &fakeRows{
					rows: []fakeRow{
						{id: "ok", content: "good", attrsJSON: []byte(`{"tenant":"acme"}`), relevance: 0.9},
						{id: "bad", content: "bad", attrsJSON: []byte(`{`), relevance: 0.5},
					},
				},
			}
			return newStore(t, db, tenantSchema(t))
		},
		func(t *testing.T) retrieval.Backend[struct{}, contracttest.StructMeta] {
			t.Helper()

			resolver := contentMergeResolver[contracttest.StructMeta]{}
			db := &fakeDB{
				queryRows: &fakeRows{
					rows: []fakeRow{
						{id: "ok", content: "merge-key", attrsJSON: []byte(`{"tenant":"acme"}`), relevance: 0.9},
						{id: "bad", content: "bad", attrsJSON: []byte(`{`), relevance: 0.5},
					},
				},
			}
			store, err := New[contracttest.StructMeta](
				db,
				Config[contracttest.StructMeta]{
					Table:    "docs",
					Schema:   tenantSchema(t),
					Resolver: resolver,
				},
				contracttest.JSONCodec[contracttest.StructMeta](t, tenantSchema(t)),
			)
			if err != nil {
				t.Fatalf("New(): %v", err)
			}
			return store
		})
}

type contentMergeResolver[TMeta any] = contracttest.ContentMergeResolver[TMeta]

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

	store, err := New[contracttest.TenantOnlyMeta](
		db,
		Config[contracttest.TenantOnlyMeta]{Table: "docs", Schema: tenantSchema(t)},
		contracttest.JSONCodec[contracttest.TenantOnlyMeta](t, tenantSchema(t)),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := retrieveStore(context.Background(), store, "", retrieval.RetrieveOptions{
		Vector: []float32{1},
		TopK:   10,
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	docs := out.Documents()
	if len(docs) != 1 || docs[0].Meta.Tenant != "acme" {
		t.Fatalf("Retrieve() = %#v, want tenant acme", docs)
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

	store, err := New[contracttest.TenantOnlyMeta](
		db,
		Config[contracttest.TenantOnlyMeta]{Table: "docs", Schema: tenantSchema(t)},
		contracttest.JSONCodec[contracttest.TenantOnlyMeta](t, tenantSchema(t)),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := retrieveStore(context.Background(), store, "", retrieval.RetrieveOptions{
		Vector: []float32{1},
		TopK:   10,
	})
	contracttest.RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Retrieve() error = %v, want protocol", err)
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

	store, err := New[contracttest.TenantOnlyMeta](
		db,
		Config[contracttest.TenantOnlyMeta]{Table: "docs", Schema: tenantSchema(t)},
		contracttest.JSONCodec[contracttest.TenantOnlyMeta](t, tenantSchema(t)),
	)
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

	store, err := New[contracttest.TenantOnlyMeta](
		db,
		Config[contracttest.TenantOnlyMeta]{Table: "docs", Schema: tenantSchema(t)},
		contracttest.JSONCodec[contracttest.TenantOnlyMeta](t, tenantSchema(t)),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	_, err = store.FindByIDs(context.Background(), []string{"doc-1"})
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("FindByIDs() error = %v, want protocol", err)
	}
}

func newDenseStructBackend(
	t *testing.T,
	docs []retrieval.Document[contracttest.StructMeta],
) retrieval.Backend[struct{}, contracttest.StructMeta] {
	t.Helper()

	schema := contracttest.TenantAgeSchema(t)
	codec := retrieval.NewJSONCodec[contracttest.StructMeta](schema)
	rows := make([]fakeRow, 0, len(docs))
	for _, doc := range docs {
		attrs, err := codec.Encode(doc.Meta)
		if err != nil {
			t.Fatalf("Encode(): %v", err)
		}
		attrsJSON, err := json.Marshal(attrs)
		if err != nil {
			t.Fatalf("Marshal(attrs): %v", err)
		}
		rows = append(rows, fakeRow{
			id:        doc.ID,
			content:   doc.Content,
			attrsJSON: attrsJSON,
			relevance: 0.1,
		})
	}

	db := &fakeDB{queryRows: &fakeRows{rows: rows}}
	store, err := New[contracttest.StructMeta](
		db,
		Config[contracttest.StructMeta]{Table: "docs", Schema: schema},
		codec,
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	return store
}

func TestDenseStructBackendConformance(t *testing.T) {
	contracttest.RunDenseStructBackendSuite(t, newDenseStructBackend)
}

func TestRetrieveOptionsInvalidConformance(t *testing.T) {
	contracttest.RunRetrieveOptionsInvalidSuite(
		t,
		func(t *testing.T) retrieval.Backend[struct{}, contracttest.StructMeta] {
			t.Helper()
			return newStoreEmptySchema(t, &fakeDB{})
		},
	)
}

func TestNewRejectsNilCodec(t *testing.T) {
	if _, err := New[contracttest.StructMeta](
		&fakeDB{},
		Config[contracttest.StructMeta]{Table: "docs", Schema: emptySchema(t)},
		nil,
	); err == nil {
		t.Fatal("New() error = nil, want error")
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("New() error = %v, want invalid argument", err)
	}
}

func TestRetrievePreservesPartialScanError(t *testing.T) {
	t.Parallel()

	db := &fakeDB{
		queryRows: &fakeRows{
			rows: []fakeRow{
				{id: "ok", content: "good", attrsJSON: nil, relevance: 0.9},
				{id: "", content: "bad", attrsJSON: nil, relevance: 0.5},
			},
		},
	}
	store := newStoreEmptySchema(t, db)

	out, err := retrieveStore(context.Background(), store, "", retrieval.RetrieveOptions{
		Vector: []float32{1},
		TopK:   5,
	})
	if !errors.Is(err, ragy.ErrMissingID) {
		t.Fatalf("Retrieve() error = %v, want missing id", err)
	}
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Retrieve() error = %v, want protocol", err)
	}
	if out.Len() != 1 || out.Documents()[0].ID != "ok" {
		t.Fatalf("Documents() = %#v, want partial ok row", out.Documents())
	}
}

func TestRetrieveQueryErrorReturnsEmptyResultSet(t *testing.T) {
	t.Parallel()

	db := &fakeDB{queryErr: ragy.ErrUnavailable}
	store := newStoreEmptySchema(t, db)

	out, err := retrieveStore(context.Background(), store, "", retrieval.RetrieveOptions{
		Vector: []float32{1},
		TopK:   5,
	})
	contracttest.RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want unavailable", err)
	}
}

func TestRetrieveWrapsRawQueryError(t *testing.T) {
	t.Parallel()

	raw := errors.New("upstream")
	db := &fakeDB{queryErr: raw}
	store := newStoreEmptySchema(t, db)

	out, err := retrieveStore(context.Background(), store, "", retrieval.RetrieveOptions{
		Vector: []float32{1},
		TopK:   5,
	})
	contracttest.RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want unavailable", err)
	}
	if !errors.Is(err, raw) {
		t.Fatalf("error chain lost upstream: %v", err)
	}
}

func TestRetrievePreservesPartialOnRowsError(t *testing.T) {
	t.Parallel()

	db := &fakeDB{
		queryRows: &fakeRows{
			rows: []fakeRow{
				{id: "ok", content: "good", attrsJSON: nil, relevance: 0.9},
			},
			rowsErr: ragy.ErrUnavailable,
		},
	}
	store := newStoreEmptySchema(t, db)

	out, err := retrieveStore(context.Background(), store, "", retrieval.RetrieveOptions{
		Vector: []float32{1},
		TopK:   5,
	})
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want unavailable", err)
	}
	if out.Len() != 1 || out.Documents()[0].ID != "ok" {
		t.Fatalf("Documents() = %#v, want partial ok row", out.Documents())
	}
}

func TestFindByIDsPreservesPartialOnDecodeError(t *testing.T) {
	t.Parallel()

	db := &fakeDB{
		queryRows: &fakeRows{
			rows: []fakeRow{
				{id: "ok", content: "good", attrsJSON: []byte(`{"tenant":"acme"}`)},
				{id: "bad", content: "bad", attrsJSON: []byte(`{`)},
			},
		},
	}
	store := newStore(t, db, tenantSchema(t))

	docs, err := store.FindByIDs(context.Background(), []string{"ok", "bad"})
	if err == nil {
		t.Fatal("FindByIDs() error = nil, want error")
	}
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("FindByIDs() error = %v, want protocol", err)
	}
	if len(docs) != 1 || docs[0].ID != "ok" {
		t.Fatalf("FindByIDs() = %#v, want partial ok doc", docs)
	}
}

func TestFindByIDsPreservesPartialOnRowsError(t *testing.T) {
	t.Parallel()

	db := &fakeDB{
		queryRows: &fakeRows{
			rows: []fakeRow{
				{id: "ok", content: "good", attrsJSON: nil},
			},
			rowsErr: ragy.ErrUnavailable,
		},
	}
	store := newStoreEmptySchema(t, db)

	docs, err := store.FindByIDs(context.Background(), []string{"ok"})
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("FindByIDs() error = %v, want unavailable", err)
	}
	if len(docs) != 1 || docs[0].ID != "ok" {
		t.Fatalf("FindByIDs() = %#v, want partial ok doc", docs)
	}
}

func TestFindByIDsPreservesPartialOnScanError(t *testing.T) {
	t.Parallel()

	db := &fakeDB{
		queryRows: &fakeRows{
			rows: []fakeRow{
				{id: "ok", content: "good", attrsJSON: nil},
				{scanErr: ragy.ErrUnavailable},
			},
		},
	}
	store := newStoreEmptySchema(t, db)

	docs, err := store.FindByIDs(context.Background(), []string{"ok", "bad"})
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("FindByIDs() error = %v, want unavailable", err)
	}
	if len(docs) != 1 || docs[0].ID != "ok" {
		t.Fatalf("FindByIDs() = %#v, want partial ok doc", docs)
	}
}

func TestFindByIDsWrapsRawQueryError(t *testing.T) {
	t.Parallel()

	raw := errors.New("upstream")
	db := &fakeDB{queryErr: raw}
	store := newStoreEmptySchema(t, db)

	_, err := store.FindByIDs(context.Background(), []string{"a"})
	if err == nil {
		t.Fatal("FindByIDs() error = nil, want error")
	}
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("FindByIDs() error = %v, want unavailable", err)
	}
	if !errors.Is(err, raw) {
		t.Fatalf("error chain lost upstream: %v", err)
	}
}

func TestDeleteByIDsWrapsRawQueryError(t *testing.T) {
	t.Parallel()

	raw := errors.New("upstream")
	db := &fakeDB{execErr: raw}
	store := newStoreEmptySchema(t, db)

	_, err := store.DeleteByIDs(context.Background(), []string{"a"})
	if err == nil {
		t.Fatal("DeleteByIDs() error = nil, want error")
	}
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("DeleteByIDs() error = %v, want unavailable", err)
	}
	if !errors.Is(err, raw) {
		t.Fatalf("error chain lost upstream: %v", err)
	}
}

func TestUpsertWrapsRawQueryError(t *testing.T) {
	t.Parallel()

	raw := errors.New("upstream")
	db := &fakeDB{execErr: raw}
	store := newStoreEmptySchema(t, db)

	err := store.Upsert(context.Background(), []dense.Record[contracttest.StructMeta]{{
		ID:      "d1",
		Content: "c",
		Vector:  []float32{1},
	}})
	if err == nil {
		t.Fatal("Upsert() error = nil, want error")
	}
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Upsert() error = %v, want unavailable", err)
	}
	if !errors.Is(err, raw) {
		t.Fatalf("error chain lost upstream: %v", err)
	}
}

func TestDeleteByFilterWrapsRawQueryError(t *testing.T) {
	t.Parallel()

	raw := errors.New("upstream")
	db := &fakeDB{execErr: raw}

	builder := filter.NewSchema()
	tenant, err := builder.String("tenant")
	if err != nil {
		t.Fatalf("builder.String(tenant): %v", err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}
	store, err := New[contracttest.StructMeta](
		db,
		Config[contracttest.StructMeta]{Table: "docs", Schema: schema},
		retrieval.NewJSONCodec[contracttest.StructMeta](schema),
	)
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

	_, err = store.DeleteByFilter(context.Background(), cond)
	if err == nil {
		t.Fatal("DeleteByFilter() error = nil, want error")
	}
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("DeleteByFilter() error = %v, want unavailable", err)
	}
	if !errors.Is(err, raw) {
		t.Fatalf("error chain lost upstream: %v", err)
	}
}
