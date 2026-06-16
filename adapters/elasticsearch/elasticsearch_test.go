package elasticsearch

import (
	"context"
	"errors"
	"maps"
	"slices"
	"sort"
	"strings"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/internal/contracttest"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/retrieval"
)

type fakeClient struct {
	body      map[string]any
	hits      []Hit
	searchErr error
}

func (f *fakeClient) Search(_ context.Context, _ string, body map[string]any) ([]Hit, error) {
	f.body = body
	if f.searchErr != nil {
		return nil, f.searchErr
	}
	return f.hits, nil
}

func schemaWithContent(t *testing.T) filter.Schema {
	t.Helper()
	return filter.EmptySchema()
}

func structMetaCodec(t *testing.T, schema filter.Schema) retrieval.MetadataCodec[contracttest.StructMeta] {
	t.Helper()
	return contracttest.JSONCodec[contracttest.StructMeta](t, schema)
}

func schemaWithContentAndTenant(t *testing.T) filter.Schema {
	t.Helper()

	builder := filter.NewSchema()
	if _, err := builder.String("tenant"); err != nil {
		t.Fatalf("builder.String(tenant): %v", err)
	}

	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	return schema
}

func TestRetrieveProjectsCanonicalDocumentShape(t *testing.T) {
	client := &fakeClient{
		hits: []Hit{{
			ID:    "doc-1",
			Score: 4,
			Source: map[string]any{
				"content": "hello",
				"tenant":  "acme",
			},
		}},
	}

	store, err := New[contracttest.StructMeta](client, Config[contracttest.StructMeta]{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schemaWithContentAndTenant(t),
	}, structMetaCodec(t, schemaWithContentAndTenant(t)))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	schema := schemaWithContentAndTenant(t)
	tenant, err := schema.StringField("tenant")
	if err != nil {
		t.Fatalf("schema.StringField(tenant): %v", err)
	}
	filterBuilder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatalf("NewBuilder(): %v", err)
	}
	cond, err := filter.Eq(filterBuilder, tenant, "acme").Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{
		TopK:    10,
		Filters: cond,
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}

	assertESFilterTenantEq(t, client.body, "acme")

	docs := out.Documents()
	if len(docs) != 1 {
		t.Fatalf("len(docs) = %d, want 1", len(docs))
	}

	if docs[0].Meta.Tenant != "acme" {
		t.Fatalf("document tenant = %#v, want acme", docs[0].Meta.Tenant)
	}
}

func TestRetrieveUsesFetchLimitForSize(t *testing.T) {
	t.Parallel()

	client := &fakeClient{
		hits: []Hit{{
			ID:     "doc-1",
			Score:  4,
			Source: map[string]any{"content": "hello"},
		}},
	}

	store, err := New[contracttest.StructMeta](client, Config[contracttest.StructMeta]{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schemaWithContent(t),
	}, structMetaCodec(t, schemaWithContent(t)))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	_, err = store.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{
		FetchLimit: 30,
		TopK:       10,
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}

	size, ok := client.body["size"].(int)
	if !ok || size != 30 {
		t.Fatalf("body[size] = %#v, want 30", client.body["size"])
	}
}

func TestRetrieveFallsBackToTopKWhenFetchLimitZero(t *testing.T) {
	t.Parallel()

	client := &fakeClient{
		hits: []Hit{{
			ID:     "doc-1",
			Score:  4,
			Source: map[string]any{"content": "hello"},
		}},
	}

	store, err := New[contracttest.StructMeta](client, Config[contracttest.StructMeta]{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schemaWithContent(t),
	}, structMetaCodec(t, schemaWithContent(t)))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	_, err = store.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{
		TopK: 18,
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}

	size, ok := client.body["size"].(int)
	if !ok || size != 18 {
		t.Fatalf("body[size] = %#v, want 18", client.body["size"])
	}
}

func TestRetrieveSetsSizeWhenTopKPositive(t *testing.T) {
	t.Parallel()

	client := &fakeClient{
		hits: []Hit{{
			ID:     "doc-1",
			Score:  4,
			Source: map[string]any{"content": "hello"},
		}},
	}

	store, err := New[contracttest.StructMeta](client, Config[contracttest.StructMeta]{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schemaWithContent(t),
	}, structMetaCodec(t, schemaWithContent(t)))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	const topK = 12
	_, err = store.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{
		TopK: topK,
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}

	size, ok := client.body["size"].(int)
	if !ok || size != topK {
		t.Fatalf("body[size] = %#v, want %d", client.body["size"], topK)
	}
}

func TestRetrieveReturnsNilMetaWhenOnlyContentIsPresent(t *testing.T) {
	client := &fakeClient{
		hits: []Hit{{
			ID:    "doc-1",
			Score: 4,
			Source: map[string]any{
				"content": "hello",
			},
		}},
	}

	store, err := New[contracttest.StructMeta](client, Config[contracttest.StructMeta]{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schemaWithContent(t),
	}, structMetaCodec(t, schemaWithContent(t)))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{TopK: 10})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}

	docs := out.Documents()
	if len(docs) != 1 {
		t.Fatalf("len(docs) = %d, want 1", len(docs))
	}

	if docs[0].Meta != (contracttest.StructMeta{}) {
		t.Fatalf("document meta = %#v, want empty", docs[0].Meta)
	}
}

func TestRetrieveRejectsMissingContent(t *testing.T) {
	client := &fakeClient{
		hits: []Hit{{
			ID:     "doc-1",
			Score:  4,
			Source: map[string]any{"tenant": "acme"},
		}},
	}

	store, err := New[contracttest.StructMeta](client, Config[contracttest.StructMeta]{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schemaWithContentAndTenant(t),
	}, structMetaCodec(t, schemaWithContentAndTenant(t)))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{TopK: 10})
	contracttest.RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Retrieve() error = %v, want protocol error", err)
	}
}

func TestRetrieveRejectsNonStringContent(t *testing.T) {
	client := &fakeClient{
		hits: []Hit{{
			ID:    "doc-1",
			Score: 4,
			Source: map[string]any{
				"content": 7,
			},
		}},
	}

	store, err := New[contracttest.StructMeta](client, Config[contracttest.StructMeta]{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schemaWithContent(t),
	}, structMetaCodec(t, schemaWithContent(t)))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{TopK: 10})
	contracttest.RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Retrieve() error = %v, want protocol error", err)
	}
}

func TestRetrieveRejectsUndeclaredFilterField(t *testing.T) {
	client := &fakeClient{}
	store, err := New[contracttest.StructMeta](client, Config[contracttest.StructMeta]{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schemaWithContent(t),
	}, structMetaCodec(t, schemaWithContent(t)))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	foreign := filter.NewSchema()
	tenant, err := foreign.String("tenant")
	if err != nil {
		t.Fatalf("foreign.String(tenant): %v", err)
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

	out, err := store.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{
		TopK:    10,
		Filters: cond,
	})
	contracttest.RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Retrieve() error = %v, want invalid argument", err)
	}
	if client.body != nil {
		t.Fatalf("body = %#v, want no backend call", client.body)
	}
}

func TestRetrieveEmptyQueryReturnsNonNilResultSet(t *testing.T) {
	t.Parallel()

	store, err := New[contracttest.StructMeta](
		&fakeClient{},
		Config[contracttest.StructMeta]{Index: "docs", SearchFields: []string{"content"}, Schema: schemaWithContent(t)},
		structMetaCodec(t, schemaWithContent(t)),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "   ", retrieval.RetrieveOptions{TopK: 1})
	contracttest.RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrEmptyText) {
		t.Fatalf("Retrieve() error = %v, want empty text", err)
	}
}

func TestNewRejectsInvalidIndexName(t *testing.T) {
	if _, err := New[contracttest.StructMeta](&fakeClient{}, Config[contracttest.StructMeta]{
		Index:        "1Bad",
		SearchFields: []string{"content"},
		Schema:       schemaWithContent(t),
	}, structMetaCodec(t, schemaWithContent(t))); err == nil {
		t.Fatal("New() error = nil, want error")
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("New() error = %v, want invalid argument", err)
	}
}

func TestNewRejectsInvalidSearchField(t *testing.T) {
	if _, err := New[contracttest.StructMeta](&fakeClient{}, Config[contracttest.StructMeta]{
		Index:        "docs",
		SearchFields: []string{"1bad"},
		Schema:       schemaWithContent(t),
	}, structMetaCodec(t, schemaWithContent(t))); err == nil {
		t.Fatal("New() error = nil, want error")
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("New() error = %v, want invalid argument", err)
	}
}

func TestNewRejectsDuplicateSearchField(t *testing.T) {
	if _, err := New[contracttest.StructMeta](&fakeClient{}, Config[contracttest.StructMeta]{
		Index:        "docs",
		SearchFields: []string{"content", "content"},
		Schema:       schemaWithContent(t),
	}, structMetaCodec(t, schemaWithContent(t))); err == nil {
		t.Fatal("New() error = nil, want error")
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("New() error = %v, want invalid argument", err)
	}
}

func TestShadowSearchMetaOnlyParityBM25AndES(t *testing.T) {
	t.Parallel()

	schema := schemaWithContentAndTenant(t)
	searchFields := []string{"tenant"}

	idx, err := lexical.NewBM25Index[contracttest.StructMeta](
		schema,
		lexical.Config[contracttest.StructMeta]{SearchFields: searchFields},
		lexical.DefaultTokenizer{},
		nil,
	)
	if err != nil {
		t.Fatalf("NewBM25Index(): %v", err)
	}
	if indexErr := idx.Index([]retrieval.Document[contracttest.StructMeta]{
		{ID: "1", Content: "secret keyword", Meta: contracttest.StructMeta{Tenant: "acme"}},
	}); indexErr != nil {
		t.Fatalf("Index(): %v", indexErr)
	}

	byContent, err := idx.Retrieve(context.Background(), "secret", retrieval.RetrieveOptions{TopK: 1})
	if err != nil {
		t.Fatalf("BM25 Retrieve(by content): %v", err)
	}
	if !byContent.IsEmpty() {
		t.Fatalf("BM25 by content = %#v, want no hits without content field", byContent.Documents())
	}

	bm25rs, err := idx.Retrieve(context.Background(), "acme", retrieval.RetrieveOptions{TopK: 1})
	if err != nil {
		t.Fatalf("BM25 Retrieve(by tenant): %v", err)
	}

	client := &fakeClient{
		hits: []Hit{{ID: "1", Score: 2, Source: map[string]any{"tenant": "acme"}}},
	}
	store, err := New[contracttest.StructMeta](client, Config[contracttest.StructMeta]{
		Index:        "docs",
		SearchFields: searchFields,
		Schema:       schema,
	}, structMetaCodec(t, schema))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	esrs, err := store.Retrieve(context.Background(), "acme", retrieval.RetrieveOptions{TopK: 5})
	if err != nil {
		t.Fatalf("ES Retrieve(): %v", err)
	}
	assertShadowRankParity(t, bm25rs, esrs)

	fields := multiMatchFields(t, client.body)
	if len(fields) != 1 || fields[0] != "tenant" {
		t.Fatalf("multi_match fields = %#v, want [tenant]", fields)
	}
}

func TestElasticsearchRetrieveMetaOnlyHitWithoutContent(t *testing.T) {
	t.Parallel()

	schema := schemaWithContentAndTenant(t)
	client := &fakeClient{
		hits: []Hit{{
			ID:     "doc-1",
			Score:  2,
			Source: map[string]any{"tenant": "acme"},
		}},
	}
	store, err := New[contracttest.StructMeta](client, Config[contracttest.StructMeta]{
		Index:        "docs",
		SearchFields: []string{"tenant"},
		Schema:       schema,
	}, structMetaCodec(t, schema))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	rs, err := store.Retrieve(context.Background(), "acme", retrieval.RetrieveOptions{TopK: 5})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 1 {
		t.Fatalf("Documents() = %#v, want one hit", rs.Documents())
	}
	if rs.Documents()[0].Content != "" {
		t.Fatalf("Content = %q, want empty when content omitted from _source", rs.Documents()[0].Content)
	}
}

func docIDs(rs retrieval.ResultSet[contracttest.StructMeta]) []string {
	docs := rs.Documents()
	ids := make([]string, len(docs))
	for i, d := range docs {
		ids[i] = d.ID
	}
	return ids
}

func assertShadowRankParity(t *testing.T, bm25rs, esrs retrieval.ResultSet[contracttest.StructMeta]) {
	t.Helper()
	// Rank-only parity by design: ES logistic score vs BM25 ClampScore differ; document IDs and order must match.
	bm25IDs := docIDs(bm25rs)
	esIDs := docIDs(esrs)
	if len(bm25IDs) != len(esIDs) {
		t.Fatalf("rank len bm25=%d es=%d", len(bm25IDs), len(esIDs))
	}
	for i := range bm25IDs {
		if bm25IDs[i] != esIDs[i] {
			t.Fatalf("rank[%d]: bm25=%q es=%q", i, bm25IDs[i], esIDs[i])
		}
	}
}

func TestShadowSearchMultiDocRankParityBM25AndES(t *testing.T) {
	t.Parallel()

	schema := schemaWithContent(t)
	searchFields := []string{"content"}

	idx, err := lexical.NewBM25Index[contracttest.StructMeta](
		schema,
		lexical.Config[contracttest.StructMeta]{SearchFields: searchFields},
		lexical.DefaultTokenizer{},
		nil,
	)
	if err != nil {
		t.Fatalf("NewBM25Index(): %v", err)
	}
	if indexErr := idx.Index([]retrieval.Document[contracttest.StructMeta]{
		{ID: "1", Content: "alpha beta gamma"},
		{ID: "2", Content: "alpha beta delta"},
		{ID: "3", Content: "alpha beta epsilon"},
	}); indexErr != nil {
		t.Fatalf("Index(): %v", indexErr)
	}

	bm25rs, err := idx.Retrieve(context.Background(), "alpha beta", retrieval.RetrieveOptions{TopK: 3})
	if err != nil {
		t.Fatalf("BM25 Retrieve(): %v", err)
	}

	client := &fakeClient{
		hits: []Hit{
			{ID: "1", Score: 3, Source: map[string]any{"content": "alpha beta gamma"}},
			{ID: "2", Score: 2, Source: map[string]any{"content": "alpha beta delta"}},
			{ID: "3", Score: 1, Source: map[string]any{"content": "alpha beta epsilon"}},
		},
	}
	store, err := New[contracttest.StructMeta](client, Config[contracttest.StructMeta]{
		Index:        "docs",
		SearchFields: searchFields,
		Schema:       schema,
	}, structMetaCodec(t, schema))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	esrs, err := store.Retrieve(context.Background(), "alpha beta", retrieval.RetrieveOptions{TopK: 3})
	if err != nil {
		t.Fatalf("ES Retrieve(): %v", err)
	}
	if bm25rs.Len() != 3 || esrs.Len() != 3 {
		t.Fatalf("Len bm25=%d es=%d, want 3", bm25rs.Len(), esrs.Len())
	}
	assertShadowRankParity(t, bm25rs, esrs)
}

func TestShadowSearchContentParityBM25AndES(t *testing.T) {
	t.Parallel()

	schema := schemaWithContent(t)
	searchFields := []string{"content"}

	idx, err := lexical.NewBM25Index[contracttest.StructMeta](
		schema,
		lexical.Config[contracttest.StructMeta]{SearchFields: searchFields},
		lexical.DefaultTokenizer{},
		nil,
	)
	if err != nil {
		t.Fatalf("NewBM25Index(): %v", err)
	}
	if indexErr := idx.Index([]retrieval.Document[contracttest.StructMeta]{
		{ID: "1", Content: "hello world"},
	}); indexErr != nil {
		t.Fatalf("Index(): %v", indexErr)
	}

	bm25rs, err := idx.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{TopK: 5})
	if err != nil {
		t.Fatalf("BM25 Retrieve(): %v", err)
	}

	client := &fakeClient{
		hits: []Hit{
			{ID: "1", Score: 2, Source: map[string]any{"content": "hello world"}},
		},
	}
	store, err := New[contracttest.StructMeta](client, Config[contracttest.StructMeta]{
		Index:        "docs",
		SearchFields: searchFields,
		Schema:       schema,
	}, structMetaCodec(t, schema))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	esrs, err := store.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{TopK: 5})
	if err != nil {
		t.Fatalf("ES Retrieve(): %v", err)
	}
	assertShadowRankParity(t, bm25rs, esrs)
	fields := multiMatchFields(t, client.body)
	if len(fields) != 1 || fields[0] != "content" {
		t.Fatalf("multi_match fields = %#v, want [content]", fields)
	}
}

func TestElasticsearchRetrieveExpandsSynonyms(t *testing.T) {
	t.Parallel()

	client := &fakeClient{}
	store, err := New[contracttest.StructMeta](client, Config[contracttest.StructMeta]{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schemaWithContent(t),
		Synonyms:     lexical.SynonymMap{"car": {"automobile"}},
	}, structMetaCodec(t, schemaWithContent(t)))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	if _, err := store.Retrieve(context.Background(), "car", retrieval.RetrieveOptions{TopK: 5}); err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}

	queryRoot, ok := client.body["query"].(map[string]any)
	if !ok {
		t.Fatalf("body[query] = %#v, want map", client.body["query"])
	}
	multiMatch, ok := queryRoot["multi_match"].(map[string]any)
	if !ok {
		t.Fatalf("query[multi_match] = %#v, want map", queryRoot["multi_match"])
	}
	queryText, ok := multiMatch["query"].(string)
	if !ok {
		t.Fatalf("multi_match[query] = %#v, want string", multiMatch["query"])
	}
	if !strings.Contains(queryText, "automobile") {
		t.Fatalf("multi_match query = %q, want synonym expansion", queryText)
	}
}

func newLexicalStructBackend(
	t *testing.T,
	docs []retrieval.Document[contracttest.StructMeta],
) retrieval.Backend[contracttest.StructMeta] {
	t.Helper()

	schema := contracttest.TenantAgeSchema(t)
	codec := retrieval.NewJSONCodec[contracttest.StructMeta](schema)
	hits := make([]Hit, 0, len(docs))
	for _, doc := range docs {
		attrs, err := codec.Encode(doc.Meta)
		if err != nil {
			t.Fatalf("Encode(): %v", err)
		}
		source := map[string]any{"content": doc.Content}
		maps.Copy(source, attrs)
		hits = append(hits, Hit{ID: doc.ID, Score: 1, Source: source})
	}

	client := &fakeClient{hits: hits}
	store, err := New[contracttest.StructMeta](client, Config[contracttest.StructMeta]{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schema,
	}, codec)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	return store
}

func TestLexicalStructBackendConformance(t *testing.T) {
	contracttest.RunLexicalStructBackendSuite(t, newLexicalStructBackend)
}

func TestRetrieveOptionsInvalidConformance(t *testing.T) {
	contracttest.RunRetrieveOptionsInvalidSuite(t, func(t *testing.T) retrieval.Backend[contracttest.StructMeta] {
		t.Helper()
		store, err := New[contracttest.StructMeta](&fakeClient{}, Config[contracttest.StructMeta]{
			Index:        "docs",
			SearchFields: []string{"content"},
			Schema:       schemaWithContent(t),
		}, structMetaCodec(t, schemaWithContent(t)))
		if err != nil {
			t.Fatalf("New(): %v", err)
		}
		return store
	}, contracttest.RetrieveOptionsInvalidConfig{Query: "hello", Vector: nil})
}

func TestNewRejectsNilCodec(t *testing.T) {
	if _, err := New[contracttest.StructMeta](
		&fakeClient{},
		Config[contracttest.StructMeta]{Index: "docs", SearchFields: []string{"content"}, Schema: schemaWithContent(t)},
		nil,
	); err == nil {
		t.Fatal("New() error = nil, want error")
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("New() error = %v, want invalid argument", err)
	}
}

func TestElasticsearchRetrieveNormalizesQueryWithoutSynonyms(t *testing.T) {
	t.Parallel()

	client := &fakeClient{}
	store, err := New[contracttest.StructMeta](client, Config[contracttest.StructMeta]{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schemaWithContent(t),
	}, structMetaCodec(t, schemaWithContent(t)))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	if _, err := store.Retrieve(context.Background(), "Hello!", retrieval.RetrieveOptions{TopK: 5}); err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}

	queryText := multiMatchQuery(t, client.body)
	if queryText != "hello" {
		t.Fatalf("multi_match query = %q, want hello", queryText)
	}
}

func TestElasticsearchRetrieveEmptyTokensReturnsEmptyResultSet(t *testing.T) {
	t.Parallel()

	client := &fakeClient{}
	store, err := New[contracttest.StructMeta](client, Config[contracttest.StructMeta]{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schemaWithContent(t),
	}, structMetaCodec(t, schemaWithContent(t)))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "!!!", retrieval.RetrieveOptions{TopK: 5})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if !out.IsEmpty() {
		t.Fatalf("Documents() = %#v, want empty", out.Documents())
	}
	if client.body != nil {
		t.Fatalf("body = %#v, want no backend call", client.body)
	}
}

func TestRetrievePartialProjectionConformance(t *testing.T) {
	contracttest.RunRetrievePartialProjectionSuite(t, func(t *testing.T) retrieval.Backend[contracttest.StructMeta] {
		t.Helper()

		client := &fakeClient{
			hits: []Hit{
				{ID: "ok", Score: 1, Source: map[string]any{"content": "good"}},
				{ID: "bad", Score: 1, Source: map[string]any{"tenant": "acme"}},
			},
		}
		store, err := New[contracttest.StructMeta](client, Config[contracttest.StructMeta]{
			Index:        "docs",
			SearchFields: []string{"content"},
			Schema:       schemaWithContentAndTenant(t),
		}, structMetaCodec(t, schemaWithContentAndTenant(t)))
		if err != nil {
			t.Fatalf("New(): %v", err)
		}
		return store
	}, func(t *testing.T) retrieval.Backend[contracttest.StructMeta] {
		t.Helper()

		client := &fakeClient{
			hits: []Hit{
				{ID: "ok", Score: 1, Source: map[string]any{"content": "merge-key", "tenant": "acme"}},
				{ID: "bad", Score: 1, Source: map[string]any{"content": "bad", "tenant": 123}},
			},
		}
		schema := schemaWithContentAndTenant(t)
		store, err := New[contracttest.StructMeta](client, Config[contracttest.StructMeta]{
			Index:        "docs",
			SearchFields: []string{"content"},
			Schema:       schema,
			Resolver:     contracttest.ContentMergeResolver[contracttest.StructMeta]{},
		}, structMetaCodec(t, schema))
		if err != nil {
			t.Fatalf("New(): %v", err)
		}
		return store
	})
}

func TestRetrieveSearchErrorReturnsEmptyResultSet(t *testing.T) {
	t.Parallel()

	client := &fakeClient{searchErr: ragy.ErrUnavailable}
	store, err := New[contracttest.StructMeta](client, Config[contracttest.StructMeta]{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schemaWithContent(t),
	}, structMetaCodec(t, schemaWithContent(t)))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{TopK: 5})
	contracttest.RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want unavailable", err)
	}
}

func TestRetrieveWrapsRawSearchError(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name string
		raw  error
	}{
		{name: "upstream", raw: errors.New("upstream")},
		{name: "transport", raw: errors.New("connection reset by peer")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			client := &fakeClient{searchErr: tc.raw}
			store, err := New[contracttest.StructMeta](client, Config[contracttest.StructMeta]{
				Index:        "docs",
				SearchFields: []string{"content"},
				Schema:       schemaWithContent(t),
			}, structMetaCodec(t, schemaWithContent(t)))
			if err != nil {
				t.Fatalf("New(): %v", err)
			}

			out, err := store.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{TopK: 5})
			contracttest.RequireErrorResultSet(t, out, err)
			if !errors.Is(err, ragy.ErrUnavailable) {
				t.Fatalf("Retrieve() error = %v, want unavailable", err)
			}
			if !errors.Is(err, tc.raw) {
				t.Fatalf("error chain lost upstream: %v", err)
			}
		})
	}
}

func TestNewRejectsUndeclaredSearchField(t *testing.T) {
	t.Parallel()

	schema := contracttest.TenantAgeSchema(t)
	if _, err := New[contracttest.StructMeta](&fakeClient{}, Config[contracttest.StructMeta]{
		Index:        "docs",
		SearchFields: []string{"unknown_field"},
		Schema:       schema,
	}, contracttest.JSONCodec[contracttest.StructMeta](t, schema)); err == nil {
		t.Fatal("New() error = nil, want error")
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("New() error = %v, want invalid argument", err)
	}
}

func TestShadowSearchSynonymParityBM25AndES(t *testing.T) {
	t.Parallel()

	synonyms := lexical.SynonymMap{"car": {"automobile"}, "automobile": {"car"}}
	schema := schemaWithContent(t)
	searchFields := []string{"content"}

	idx, err := lexical.NewBM25Index[contracttest.StructMeta](
		schema,
		lexical.Config[contracttest.StructMeta]{SearchFields: searchFields},
		lexical.DefaultTokenizer{},
		synonyms,
	)
	if err != nil {
		t.Fatalf("NewBM25Index(): %v", err)
	}
	if indexErr := idx.Index([]retrieval.Document[contracttest.StructMeta]{
		{ID: "1", Content: "fast automobile"},
	}); indexErr != nil {
		t.Fatalf("Index(): %v", indexErr)
	}
	bm25rs, err := idx.Retrieve(context.Background(), "car", retrieval.RetrieveOptions{TopK: 1})
	if err != nil {
		t.Fatalf("BM25 Retrieve(): %v", err)
	}

	client := &fakeClient{
		hits: []Hit{{ID: "1", Score: 1, Source: map[string]any{"content": "fast automobile"}}},
	}
	store, err := New[contracttest.StructMeta](client, Config[contracttest.StructMeta]{
		Index:        "docs",
		SearchFields: searchFields,
		Schema:       schema,
		Synonyms:     synonyms,
	}, structMetaCodec(t, schema))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	esrs, err := store.Retrieve(context.Background(), "car", retrieval.RetrieveOptions{TopK: 5})
	if err != nil {
		t.Fatalf("ES Retrieve(): %v", err)
	}
	assertShadowRankParity(t, bm25rs, esrs)
	queryText := multiMatchQuery(t, client.body)
	if !strings.Contains(queryText, "automobile") {
		t.Fatalf("ES query = %q, want automobile synonym", queryText)
	}
}

func TestShadowSearchSynonymMultiDocRankParityBM25AndES(t *testing.T) {
	t.Parallel()

	synonyms := lexical.SynonymMap{"car": {"automobile"}, "automobile": {"car"}}
	schema := schemaWithContent(t)
	searchFields := []string{"content"}

	idx, err := lexical.NewBM25Index[contracttest.StructMeta](
		schema,
		lexical.Config[contracttest.StructMeta]{SearchFields: searchFields},
		lexical.DefaultTokenizer{},
		synonyms,
	)
	if err != nil {
		t.Fatalf("NewBM25Index(): %v", err)
	}
	if indexErr := idx.Index([]retrieval.Document[contracttest.StructMeta]{
		{ID: "1", Content: "fast automobile"},
		{ID: "2", Content: "slow car"},
	}); indexErr != nil {
		t.Fatalf("Index(): %v", indexErr)
	}

	bm25rs, err := idx.Retrieve(context.Background(), "car", retrieval.RetrieveOptions{TopK: 2})
	if err != nil {
		t.Fatalf("BM25 Retrieve(): %v", err)
	}

	client := &fakeClient{
		hits: []Hit{
			{ID: "1", Score: 2, Source: map[string]any{"content": "fast automobile"}},
			{ID: "2", Score: 1, Source: map[string]any{"content": "slow car"}},
		},
	}
	store, err := New[contracttest.StructMeta](client, Config[contracttest.StructMeta]{
		Index:        "docs",
		SearchFields: searchFields,
		Schema:       schema,
		Synonyms:     synonyms,
	}, structMetaCodec(t, schema))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	esrs, err := store.Retrieve(context.Background(), "car", retrieval.RetrieveOptions{TopK: 2})
	if err != nil {
		t.Fatalf("ES Retrieve(): %v", err)
	}
	if bm25rs.Len() != 2 || esrs.Len() != 2 {
		t.Fatalf("Len bm25=%d es=%d, want 2", bm25rs.Len(), esrs.Len())
	}
	assertShadowRankParity(t, bm25rs, esrs)
	queryText := multiMatchQuery(t, client.body)
	if !strings.Contains(queryText, "automobile") {
		t.Fatalf("ES query = %q, want automobile synonym", queryText)
	}
}

func TestShadowSearchFilterParityBM25AndES(t *testing.T) {
	t.Parallel()

	schema := schemaWithContentAndTenant(t)
	searchFields := []string{"content"}

	idx, err := lexical.NewBM25Index[contracttest.StructMeta](
		schema,
		lexical.Config[contracttest.StructMeta]{SearchFields: searchFields},
		lexical.DefaultTokenizer{},
		nil,
	)
	if err != nil {
		t.Fatalf("NewBM25Index(): %v", err)
	}
	if indexErr := idx.Index([]retrieval.Document[contracttest.StructMeta]{
		{ID: "a", Content: "secret alpha", Meta: contracttest.StructMeta{Tenant: "acme"}},
		{ID: "b", Content: "secret beta", Meta: contracttest.StructMeta{Tenant: "other"}},
	}); indexErr != nil {
		t.Fatalf("Index(): %v", indexErr)
	}

	tenant, err := schema.StringField("tenant")
	if err != nil {
		t.Fatalf("schema.StringField(tenant): %v", err)
	}
	filterBuilder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatalf("NewBuilder(): %v", err)
	}
	cond, err := filter.Eq(filterBuilder, tenant, "acme").Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	unfiltered, err := idx.Retrieve(context.Background(), "secret", retrieval.RetrieveOptions{TopK: 2})
	if err != nil {
		t.Fatalf("BM25 Retrieve(unfiltered): %v", err)
	}
	if unfiltered.Len() != 2 {
		t.Fatalf("unfiltered Len() = %d, want 2", unfiltered.Len())
	}

	bm25rs, err := idx.Retrieve(context.Background(), "secret", retrieval.RetrieveOptions{
		TopK: 2, Filters: cond,
	})
	if err != nil {
		t.Fatalf("BM25 Retrieve(filtered): %v", err)
	}
	if bm25rs.Len() != 1 || bm25rs.Documents()[0].ID != "a" {
		t.Fatalf("BM25 filtered = %#v, want [a]", bm25rs.Documents())
	}

	client := &fakeClient{
		hits: []Hit{{
			ID:     "a",
			Score:  2,
			Source: map[string]any{"content": "secret alpha", "tenant": "acme"},
		}},
	}
	store, err := New[contracttest.StructMeta](client, Config[contracttest.StructMeta]{
		Index:        "docs",
		SearchFields: searchFields,
		Schema:       schema,
	}, structMetaCodec(t, schema))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	esrs, err := store.Retrieve(context.Background(), "secret", retrieval.RetrieveOptions{
		TopK: 2, Filters: cond,
	})
	if err != nil {
		t.Fatalf("ES Retrieve(): %v", err)
	}
	assertShadowRankParity(t, bm25rs, esrs)
	assertESFilterTenantEq(t, client.body, "acme")
}

func TestShadowSearchMultiFieldRankParityBM25AndES(t *testing.T) {
	t.Parallel()

	schema := schemaWithContentAndTenant(t)
	searchFields := []string{"content", "tenant"}

	idx, err := lexical.NewBM25Index[contracttest.StructMeta](
		schema,
		lexical.Config[contracttest.StructMeta]{SearchFields: searchFields},
		lexical.DefaultTokenizer{},
		nil,
	)
	if err != nil {
		t.Fatalf("NewBM25Index(): %v", err)
	}
	if indexErr := idx.Index([]retrieval.Document[contracttest.StructMeta]{
		{ID: "1", Content: "alpha beta gamma", Meta: contracttest.StructMeta{Tenant: "zzz"}},
		{ID: "2", Content: "alpha beta delta", Meta: contracttest.StructMeta{Tenant: "alpha"}},
	}); indexErr != nil {
		t.Fatalf("Index(): %v", indexErr)
	}

	bm25rs, err := idx.Retrieve(context.Background(), "alpha beta", retrieval.RetrieveOptions{TopK: 2})
	if err != nil {
		t.Fatalf("BM25 Retrieve(): %v", err)
	}
	if bm25rs.Len() != 2 {
		t.Fatalf("BM25 Len() = %d, want 2", bm25rs.Len())
	}

	hits := make([]Hit, len(bm25rs.Documents()))
	for i, doc := range bm25rs.Documents() {
		hits[i] = Hit{
			ID:    doc.ID,
			Score: float64(len(bm25rs.Documents()) - i),
			Source: map[string]any{
				"content": doc.Content,
				"tenant":  doc.Meta.Tenant,
			},
		}
	}

	client := &fakeClient{hits: hits}
	store, err := New[contracttest.StructMeta](client, Config[contracttest.StructMeta]{
		Index:        "docs",
		SearchFields: searchFields,
		Schema:       schema,
	}, structMetaCodec(t, schema))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	esrs, err := store.Retrieve(context.Background(), "alpha beta", retrieval.RetrieveOptions{TopK: 2})
	if err != nil {
		t.Fatalf("ES Retrieve(): %v", err)
	}
	assertShadowRankParity(t, bm25rs, esrs)

	fields := multiMatchFields(t, client.body)
	sort.Strings(fields)
	want := []string{"content", "tenant"}
	sort.Strings(want)
	if !slices.Equal(fields, want) {
		t.Fatalf("multi_match fields = %#v, want %#v", fields, want)
	}
}

func assertESFilterTenantEq(t *testing.T, body map[string]any, wantTenant string) {
	t.Helper()

	queryRoot, ok := body["query"].(map[string]any)
	if !ok {
		t.Fatalf("body[query] = %#v, want map", body["query"])
	}
	boolQ, ok := queryRoot["bool"].(map[string]any)
	if !ok {
		t.Fatalf("query = %#v, want bool wrapper when Filters set", queryRoot)
	}
	filterClause, ok := boolQ["filter"].([]any)
	if !ok || len(filterClause) == 0 {
		t.Fatalf("bool.filter = %#v, want non-empty filter clause", boolQ["filter"])
	}
	term, ok := filterClause[0].(map[string]any)["term"].(map[string]any)
	if !ok {
		t.Fatalf("filter[0] = %#v, want term query", filterClause[0])
	}
	got, ok := term["tenant"].(string)
	if !ok || got != wantTenant {
		t.Fatalf("term[tenant] = %#v, want %q", term["tenant"], wantTenant)
	}
}

func multiMatchQuery(t *testing.T, body map[string]any) string {
	t.Helper()
	queryRoot, ok := body["query"].(map[string]any)
	if !ok {
		t.Fatalf("body[query] = %#v, want map", body["query"])
	}
	multiMatch, ok := queryRoot["multi_match"].(map[string]any)
	if !ok {
		t.Fatalf("query[multi_match] = %#v, want map", queryRoot["multi_match"])
	}
	queryText, ok := multiMatch["query"].(string)
	if !ok {
		t.Fatalf("multi_match[query] = %#v, want string", multiMatch["query"])
	}
	return queryText
}

func multiMatchFields(t *testing.T, body map[string]any) []string {
	t.Helper()
	queryRoot, ok := body["query"].(map[string]any)
	if !ok {
		t.Fatalf("body[query] = %#v, want map", body["query"])
	}
	multiMatch, ok := queryRoot["multi_match"].(map[string]any)
	if !ok {
		t.Fatalf("query[multi_match] = %#v, want map", queryRoot["multi_match"])
	}
	rawFields, ok := multiMatch["fields"].([]string)
	if !ok {
		if anyFields, ok := multiMatch["fields"].([]any); ok {
			out := make([]string, 0, len(anyFields))
			for _, field := range anyFields {
				s, ok := field.(string)
				if !ok {
					t.Fatalf("multi_match[fields] entry = %#v, want string", field)
				}
				out = append(out, s)
			}
			return out
		}
		t.Fatalf("multi_match[fields] = %#v, want []string", multiMatch["fields"])
	}
	return rawFields
}
