package lexical

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/internal/contracttest"
	"github.com/skosovsky/ragy/retrieval"
)

func retrieveBM25[TMeta any](
	ctx context.Context,
	idx *BM25Index[TMeta],
	text string,
	opts retrieval.RetrieveOptions,
) (retrieval.ResultSet[TMeta], error) {
	return idx.Retrieve(ctx, retrieval.Query[struct{}]{Text: text, Options: opts})
}

func TestBM25RetrieveRace(t *testing.T) {
	builder := filter.NewSchema()
	if _, err := builder.String("tenant"); err != nil {
		t.Fatalf("String(tenant): %v", err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	idx, err := NewBM25Index[struct{}](
		schema,
		Config[struct{}]{SearchFields: []string{"content"}},
		DefaultTokenizer{},
		nil,
	)
	if err != nil {
		t.Fatalf("NewBM25Index(): %v", err)
	}

	for _, doc := range []retrieval.Document[struct{}]{
		{ID: "1", Content: "hello world"},
		{ID: "2", Content: "hello again"},
	} {
		if err := idx.Upsert(doc); err != nil {
			t.Fatalf("Upsert(): %v", err)
		}
	}

	const workers = 8
	var wg sync.WaitGroup
	wg.Add(workers)
	for range workers {
		go func() {
			defer wg.Done()
			rs, err := retrieveBM25(context.Background(), idx, "hello", retrieval.RetrieveOptions{TopK: 2})
			if err != nil {
				t.Error(err)
				return
			}
			if rs.IsEmpty() {
				t.Error("expected hits")
			}
		}()
	}
	wg.Wait()
}

func TestBM25RetrieveUsesPlannedExpandedText(t *testing.T) {
	t.Parallel()

	schema, err := filter.NewSchema().Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}
	idx, err := NewBM25Index[struct{}](
		schema,
		Config[struct{}]{SearchFields: []string{"content"}},
		DefaultTokenizer{},
		nil,
	)
	if err != nil {
		t.Fatalf("NewBM25Index(): %v", err)
	}
	if upsertErr := idx.Upsert(retrieval.Document[struct{}]{
		ID:      "planned",
		Content: "expanded token",
	}); upsertErr != nil {
		t.Fatalf("Upsert(): %v", upsertErr)
	}

	rs, err := idx.Retrieve(context.Background(), retrieval.Query[struct{}]{
		Text: "missing",
		Plan: &retrieval.PlannedQuery[struct{}]{
			ExpandedText: "expanded",
		},
		Options: retrieval.RetrieveOptions{TopK: 1},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "planned" {
		t.Fatalf("Documents() = %#v, want planned text hit", rs.Documents())
	}
}

func TestBM25RetrieveMarksRankedDocsScorePresent(t *testing.T) {
	t.Parallel()

	schema, err := filter.NewSchema().Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}
	idx, err := NewBM25Index[struct{}](
		schema,
		Config[struct{}]{SearchFields: []string{"content"}},
		DefaultTokenizer{},
		nil,
	)
	if err != nil {
		t.Fatalf("NewBM25Index(): %v", err)
	}
	if upsertErr := idx.Upsert(retrieval.Document[struct{}]{
		ID:         "rank-only-source",
		Content:    "alpha beta",
		ScoreState: retrieval.ScoreAbsent,
		Rank:       1,
	}); upsertErr != nil {
		t.Fatalf("Upsert(): %v", upsertErr)
	}

	rs, err := idx.Retrieve(context.Background(), retrieval.Query[struct{}]{
		Text:    "alpha",
		Options: retrieval.RetrieveOptions{TopK: 1},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	docs := rs.Documents()
	if len(docs) != 1 {
		t.Fatalf("len(Documents()) = %d, want 1", len(docs))
	}
	if docs[0].ScoreState != retrieval.ScorePresent || docs[0].Score <= 0 || docs[0].Rank != 1 {
		t.Fatalf("score/rank = (%v,%v,%d), want present positive score with rank 1",
			docs[0].ScoreState, docs[0].Score, docs[0].Rank)
	}
}

func TestBM25AppliesDeclaredFilter(t *testing.T) {
	builder := filter.NewSchema()
	if _, err := builder.String("tenant"); err != nil {
		t.Fatalf("String(tenant): %v", err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	idx, err := NewBM25Index[struct {
		Tenant string `json:"tenant"`
	}](
		schema,
		Config[struct {
			Tenant string `json:"tenant"`
		}]{SearchFields: []string{"content"}},
		DefaultTokenizer{},
		nil,
	)
	if err != nil {
		t.Fatalf("NewBM25Index(): %v", err)
	}

	if indexErr := idx.Index([]retrieval.Document[struct {
		Tenant string `json:"tenant"`
	}]{
		{ID: "1", Content: "hello world", Meta: struct {
			Tenant string `json:"tenant"`
		}{Tenant: "acme"}},
		{ID: "2", Content: "hello again", Meta: struct {
			Tenant string `json:"tenant"`
		}{Tenant: "other"}},
	}); indexErr != nil {
		t.Fatalf("Index(): %v", indexErr)
	}

	filterBuilder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatalf("NewBuilder(): %v", err)
	}
	tenant, err := schema.StringField("tenant")
	if err != nil {
		t.Fatalf("StringField(tenant): %v", err)
	}
	cond, err := filter.Eq(filterBuilder, tenant, "acme").Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := retrieveBM25(context.Background(), idx, "hello", retrieval.RetrieveOptions{
		TopK:    10,
		Filters: cond,
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	docs := rs.Documents()
	if len(docs) != 1 || docs[0].ID != "1" {
		t.Fatalf("Documents() = %#v, want doc 1", docs)
	}
}

func TestBM25SynonymsAffectScoring(t *testing.T) {
	schema, err := filter.NewSchema().Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	idx, err := NewBM25Index[struct{}](
		schema,
		Config[struct{}]{SearchFields: []string{"content"}},
		DefaultTokenizer{},
		SynonymMap{"automobile": {"car"}},
	)
	if err != nil {
		t.Fatalf("NewBM25Index(): %v", err)
	}

	if indexErr := idx.Index([]retrieval.Document[struct{}]{
		{ID: "1", Content: "fast car"},
		{ID: "2", Content: "slow boat"},
	}); indexErr != nil {
		t.Fatalf("Index(): %v", indexErr)
	}

	rs, err := retrieveBM25(context.Background(), idx, "automobile", retrieval.RetrieveOptions{TopK: 1})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Documents()[0].ID != "1" {
		t.Fatalf("Documents() = %#v, want doc 1", rs.Documents())
	}
}

func TestBM25RankPreservesTieOrder(t *testing.T) {
	t.Parallel()

	schema, err := filter.NewSchema().Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	idx, err := NewBM25Index[struct{}](
		schema,
		Config[struct{}]{SearchFields: []string{"content"}},
		DefaultTokenizer{},
		nil,
	)
	if err != nil {
		t.Fatalf("NewBM25Index(): %v", err)
	}

	// Same token bag → equal BM25 raw score; IDs chosen so map iteration ≠ sorted materialization order.
	if indexErr := idx.Index([]retrieval.Document[struct{}]{
		{ID: "doc-z", Content: "tie query terms"},
		{ID: "doc-a", Content: "tie query terms"},
	}); indexErr != nil {
		t.Fatalf("Index(): %v", indexErr)
	}

	rs, err := retrieveBM25(context.Background(), idx, "tie query terms", retrieval.RetrieveOptions{TopK: 2})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	docs := rs.Documents()
	if len(docs) != 2 {
		t.Fatalf("Documents() len = %d, want 2", len(docs))
	}
	if docs[0].Score != docs[1].Score {
		t.Fatalf("Documents() scores = %v, %v, want equal scores", docs[0].Score, docs[1].Score)
	}
	// Sorted doc-ID materialization + score-only stable sort (no ID tie-break comparator).
	if docs[0].ID != "doc-a" || docs[1].ID != "doc-z" {
		t.Fatalf("Documents() = %#v, want doc-a then doc-z on score tie", docs)
	}
}

func TestBM25UpsertZeroTokensReturnsError(t *testing.T) {
	schema, err := filter.NewSchema().Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	idx, err := NewBM25Index[struct{}](
		schema,
		Config[struct{}]{SearchFields: []string{"content"}},
		DefaultTokenizer{},
		nil,
	)
	if err != nil {
		t.Fatalf("NewBM25Index(): %v", err)
	}

	if upsertErr := idx.Upsert(retrieval.Document[struct{}]{ID: "1", Content: "hello"}); upsertErr != nil {
		t.Fatalf("Upsert(): %v", upsertErr)
	}

	err = idx.Upsert(retrieval.Document[struct{}]{ID: "1", Content: "   "})
	if !errors.Is(err, ragy.ErrEmptyText) {
		t.Fatalf("Upsert() error = %v, want empty text", err)
	}

	rs, err := retrieveBM25(context.Background(), idx, "hello", retrieval.RetrieveOptions{TopK: 1})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if len(rs.Documents()) != 1 || rs.Documents()[0].Content != "hello" {
		t.Fatalf("Documents() = %#v, want original doc preserved", rs.Documents())
	}
}

func TestBM25IndexZeroTokensReturnsError(t *testing.T) {
	schema, err := filter.NewSchema().Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	idx, err := NewBM25Index[struct{}](
		schema,
		Config[struct{}]{SearchFields: []string{"content"}},
		DefaultTokenizer{},
		nil,
	)
	if err != nil {
		t.Fatalf("NewBM25Index(): %v", err)
	}

	err = idx.Index([]retrieval.Document[struct{}]{
		{ID: "1", Content: "valid"},
		{ID: "2", Content: ""},
	})
	if !errors.Is(err, ragy.ErrEmptyText) {
		t.Fatalf("Index() error = %v, want empty text", err)
	}
}

type bm25ContractBackend struct {
	*BM25Index[contracttest.StructMeta]

	staged   []retrieval.Document[contracttest.StructMeta]
	resolver retrieval.IdentityResolver[contracttest.StructMeta]
}

func (b *bm25ContractBackend) Retrieve(
	ctx context.Context,
	req retrieval.Query[struct{}],
) (retrieval.ResultSet[contracttest.StructMeta], error) {
	for _, doc := range b.staged {
		if err := retrieval.ValidateDocument(doc); err != nil {
			return retrieval.NewResultSet[contracttest.StructMeta](
				nil,
				b.resolver,
			), ragy.WrapProjectionError(err, "bm25 contract validate")
		}
	}
	return b.BM25Index.Retrieve(ctx, req)
}

func newBM25LexicalStructBackend(
	t *testing.T,
	docs []retrieval.Document[contracttest.StructMeta],
) retrieval.Backend[struct{}, contracttest.StructMeta] {
	t.Helper()

	schema := contracttest.TenantAgeSchema(t)
	cfg := Config[contracttest.StructMeta]{SearchFields: []string{"content"}}
	idx, err := NewBM25Index[contracttest.StructMeta](
		schema,
		cfg,
		DefaultTokenizer{},
		nil,
	)
	if err != nil {
		t.Fatalf("NewBM25Index(): %v", err)
	}

	toIndex := make([]retrieval.Document[contracttest.StructMeta], 0, len(docs))
	for _, doc := range docs {
		// Invalid docs are excluded from the index and left staged for projection-error paths.
		if err := retrieval.ValidateDocument(doc); err != nil {
			continue
		}
		indexDoc := doc
		if strings.TrimSpace(indexDoc.Content) == "" {
			indexDoc.Content = "seed"
		}
		toIndex = append(toIndex, indexDoc)
	}
	if len(toIndex) > 0 {
		if err := idx.Index(toIndex); err != nil {
			t.Fatalf("Index(): %v", err)
		}
	}
	return &bm25ContractBackend{
		BM25Index: idx,
		staged:    docs,
		resolver:  retrieval.DefaultResolver(cfg.Resolver),
	}
}

func TestBM25LexicalStructBackendConformance(t *testing.T) {
	contracttest.RunLexicalStructBackendSuite(t, newBM25LexicalStructBackend)
}

func TestRetrieveOptionsInvalidConformance(t *testing.T) {
	contracttest.RunRetrieveOptionsInvalidSuite(
		t,
		func(t *testing.T) retrieval.Backend[struct{}, contracttest.StructMeta] {
			t.Helper()
			schema, err := filter.NewSchema().Build()
			if err != nil {
				t.Fatalf("Build(): %v", err)
			}
			idx, err := NewBM25Index[contracttest.StructMeta](
				schema,
				Config[contracttest.StructMeta]{SearchFields: []string{"content"}},
				DefaultTokenizer{},
				nil,
			)
			if err != nil {
				t.Fatalf("NewBM25Index(): %v", err)
			}
			return idx
		},
		contracttest.RetrieveOptionsInvalidConfig{Query: "hello", Vector: nil},
	)
}

type failingAfterFirstEncodeCodec[TMeta any] struct {
	inner retrieval.MetadataCodec[TMeta]
	calls int
}

func (c *failingAfterFirstEncodeCodec[TMeta]) Encode(meta TMeta) (filter.RawAttributes, error) {
	c.calls++
	if c.calls > 1 {
		return nil, fmt.Errorf("%w: injected encode failure", ragy.ErrInvalidArgument)
	}
	return c.inner.Encode(meta)
}

func (c *failingAfterFirstEncodeCodec[TMeta]) Decode(attrs filter.RawAttributes) (TMeta, error) {
	return c.inner.Decode(attrs)
}

type bm25PartialProjectionBackend struct {
	*BM25Index[contracttest.StructMeta]

	filters filter.Condition
}

func (b *bm25PartialProjectionBackend) Retrieve(
	ctx context.Context,
	req retrieval.Query[struct{}],
) (retrieval.ResultSet[contracttest.StructMeta], error) {
	opts := req.Options
	opts.Filters = b.filters
	req.Options = opts
	return b.BM25Index.Retrieve(ctx, req)
}

func TestRetrievePartialProjectionConformance(t *testing.T) {
	contracttest.RunRetrievePartialProjectionSuite(
		t,
		func(t *testing.T) retrieval.Backend[struct{}, contracttest.StructMeta] {
			t.Helper()

			schema := contracttest.TenantAgeSchema(t)
			innerCodec := retrieval.NewJSONCodec[contracttest.StructMeta](schema)
			idx, err := NewBM25Index[contracttest.StructMeta](
				schema,
				Config[contracttest.StructMeta]{
					SearchFields: []string{"content"},
					Codec:        &failingAfterFirstEncodeCodec[contracttest.StructMeta]{inner: innerCodec},
				},
				DefaultTokenizer{},
				nil,
			)
			if err != nil {
				t.Fatalf("NewBM25Index(): %v", err)
			}

			tenant, err := schema.StringField("tenant")
			if err != nil {
				t.Fatalf("Schema().StringField(tenant): %v", err)
			}
			builder, err := filter.NewBuilder(schema)
			if err != nil {
				t.Fatalf("NewBuilder(): %v", err)
			}
			cond, err := filter.In(builder, tenant, "acme", "globex").Build()
			if err != nil {
				t.Fatalf("Build(): %v", err)
			}

			if err := idx.Index([]retrieval.Document[contracttest.StructMeta]{
				{ID: "ok", Content: "q good", Meta: contracttest.StructMeta{Tenant: "acme"}},
				{ID: "zzbad", Content: "q bad", Meta: contracttest.StructMeta{Tenant: "globex"}},
			}); err != nil {
				t.Fatalf("Index(): %v", err)
			}

			return &bm25PartialProjectionBackend{BM25Index: idx, filters: cond}
		},
		func(t *testing.T) retrieval.Backend[struct{}, contracttest.StructMeta] {
			t.Helper()

			schema := contracttest.TenantAgeSchema(t)
			innerCodec := retrieval.NewJSONCodec[contracttest.StructMeta](schema)
			resolver := contracttest.ContentMergeResolver[contracttest.StructMeta]{}
			idx, err := NewBM25Index[contracttest.StructMeta](
				schema,
				Config[contracttest.StructMeta]{
					SearchFields: []string{"content"},
					Codec:        &failingAfterFirstEncodeCodec[contracttest.StructMeta]{inner: innerCodec},
					Resolver:     resolver,
				},
				DefaultTokenizer{},
				nil,
			)
			if err != nil {
				t.Fatalf("NewBM25Index(): %v", err)
			}

			tenant, err := schema.StringField("tenant")
			if err != nil {
				t.Fatalf("Schema().StringField(tenant): %v", err)
			}
			builder, err := filter.NewBuilder(schema)
			if err != nil {
				t.Fatalf("NewBuilder(): %v", err)
			}
			cond, err := filter.In(builder, tenant, "acme", "globex").Build()
			if err != nil {
				t.Fatalf("Build(): %v", err)
			}

			if indexErr := idx.Index([]retrieval.Document[contracttest.StructMeta]{
				{ID: "ok", Content: "q merge-key", Meta: contracttest.StructMeta{Tenant: "acme"}},
				{ID: "zzbad", Content: "q bad", Meta: contracttest.StructMeta{Tenant: "globex"}},
			}); indexErr != nil {
				t.Fatalf("Index(): %v", indexErr)
			}

			return &bm25PartialProjectionBackend{BM25Index: idx, filters: cond}
		})
}

func TestBM25RetrieveErrorReturnsNonNilResultSet(t *testing.T) {
	t.Parallel()

	schema, err := filter.NewSchema().Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}
	idx, err := NewBM25Index[struct{}](
		schema,
		Config[struct{}]{SearchFields: []string{"content"}},
		DefaultTokenizer{},
		nil,
	)
	if err != nil {
		t.Fatalf("NewBM25Index(): %v", err)
	}

	out, err := retrieveBM25(context.Background(), idx, "   ", retrieval.RetrieveOptions{TopK: 1})
	contracttest.RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrEmptyText) {
		t.Fatalf("Retrieve() error = %v, want empty text", err)
	}
}

func TestBM25UpsertRejectsEncodeFailure(t *testing.T) {
	t.Parallel()

	type wrongTenantMeta struct {
		Tenant int `json:"tenant"`
	}

	builder := filter.NewSchema()
	if _, err := builder.String("tenant"); err != nil {
		t.Fatalf("String(tenant): %v", err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}
	idx, err := NewBM25Index[wrongTenantMeta](
		schema,
		Config[wrongTenantMeta]{SearchFields: []string{"tenant"}},
		DefaultTokenizer{},
		nil,
	)
	if err != nil {
		t.Fatalf("NewBM25Index(): %v", err)
	}

	doc := retrieval.Document[wrongTenantMeta]{
		ID:      "doc-1",
		Content: "hello",
		Score:   1,
		Meta:    wrongTenantMeta{Tenant: 1},
	}
	if err := idx.Upsert(doc); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Upsert() error = %v, want invalid argument from encode", err)
	}
}

func TestBM25RejectsUnknownSearchField(t *testing.T) {
	t.Parallel()

	schema, err := filter.NewSchema().Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}
	_, err = NewBM25Index[struct{}](
		schema,
		Config[struct{}]{SearchFields: []string{"missing_field"}},
		DefaultTokenizer{},
		nil,
	)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("NewBM25Index() error = %v, want invalid argument", err)
	}
}

type failingAfterFirstCodec[TMeta any] struct {
	calls int
	inner retrieval.MetadataCodec[TMeta]
}

func (c *failingAfterFirstCodec[TMeta]) Encode(meta TMeta) (filter.RawAttributes, error) {
	c.calls++
	if c.calls > 1 {
		return nil, ragy.ErrInvalidArgument
	}
	return c.inner.Encode(meta)
}

func (c *failingAfterFirstCodec[TMeta]) Decode(attrs filter.RawAttributes) (TMeta, error) {
	return c.inner.Decode(attrs)
}

func TestBM25FilterPreservesPartialScoresOnMatchError(t *testing.T) {
	t.Parallel()

	builder := filter.NewSchema()
	tenant, err := builder.String("tenant")
	if err != nil {
		t.Fatalf("String(tenant): %v", err)
	}
	schemaWithTenant, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}
	idx, err := NewBM25Index[contracttest.TenantOnlyMeta](
		schemaWithTenant,
		Config[contracttest.TenantOnlyMeta]{SearchFields: []string{"content"}},
		DefaultTokenizer{},
		nil,
	)
	if err != nil {
		t.Fatalf("NewBM25Index(): %v", err)
	}

	snapshot := bm25Snapshot[contracttest.TenantOnlyMeta]{
		docs: map[string]retrieval.Document[contracttest.TenantOnlyMeta]{
			"a": {ID: "a", Content: "hello world", Meta: contracttest.TenantOnlyMeta{Tenant: "acme"}},
			"b": {ID: "b", Content: "hello again", Meta: contracttest.TenantOnlyMeta{Tenant: "other"}},
		},
	}
	scores := map[string]float64{"a": 1.0, "b": 0.5}
	codec := &failingAfterFirstCodec[contracttest.TenantOnlyMeta]{
		inner: retrieval.NewJSONCodec[contracttest.TenantOnlyMeta](schemaWithTenant),
	}
	filterBuilder, err := filter.NewBuilder(schemaWithTenant)
	if err != nil {
		t.Fatalf("NewBuilder(): %v", err)
	}
	cond, err := filter.Eq(filterBuilder, tenant, "acme").Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	filtered, err := idx.filterScoredDocs(snapshot, scores, cond, codec)
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("filterScoredDocs() error = %v, want protocol", err)
	}
	if len(filtered) != 1 || filtered["a"] != 1.0 {
		t.Fatalf("filtered scores = %#v, want partial entry for doc a", filtered)
	}
}

func TestBM25SearchFieldsExcludesContentWhenNotDeclared(t *testing.T) {
	t.Parallel()

	builder := filter.NewSchema()
	if _, err := builder.String("tenant"); err != nil {
		t.Fatalf("String(tenant): %v", err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}
	idx, err := NewBM25Index[contracttest.TenantOnlyMeta](
		schema,
		Config[contracttest.TenantOnlyMeta]{SearchFields: []string{"tenant"}},
		DefaultTokenizer{},
		nil,
	)
	if err != nil {
		t.Fatalf("NewBM25Index(): %v", err)
	}
	if indexErr := idx.Index([]retrieval.Document[contracttest.TenantOnlyMeta]{
		{ID: "1", Content: "secret keyword", Meta: contracttest.TenantOnlyMeta{Tenant: "acme"}},
	}); indexErr != nil {
		t.Fatalf("Index(): %v", indexErr)
	}

	byContent, err := retrieveBM25(context.Background(), idx, "secret", retrieval.RetrieveOptions{TopK: 1})
	if err != nil {
		t.Fatalf("Retrieve(by content): %v", err)
	}
	if !byContent.IsEmpty() {
		t.Fatalf("Retrieve(by content) = %#v, want no hits without content field", byContent.Documents())
	}

	byTenant, err := retrieveBM25(context.Background(), idx, "acme", retrieval.RetrieveOptions{TopK: 1})
	if err != nil {
		t.Fatalf("Retrieve(by tenant): %v", err)
	}
	if byTenant.Len() != 1 || byTenant.Documents()[0].ID != "1" {
		t.Fatalf("Retrieve(by tenant) = %#v, want doc 1", byTenant.Documents())
	}
}

func TestBM25FieldValueUsesConfigCodec(t *testing.T) {
	t.Parallel()

	builder := filter.NewSchema()
	if _, err := builder.String("tenant"); err != nil {
		t.Fatalf("String(tenant): %v", err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	markerCodec := &markerTenantCodec{}
	idx, err := NewBM25Index[contracttest.TenantOnlyMeta](
		schema,
		Config[contracttest.TenantOnlyMeta]{
			SearchFields: []string{"tenant"},
			Codec:        markerCodec,
		},
		DefaultTokenizer{},
		nil,
	)
	if err != nil {
		t.Fatalf("NewBM25Index(): %v", err)
	}
	if indexErr := idx.Index([]retrieval.Document[contracttest.TenantOnlyMeta]{
		{ID: "1", Content: "body", Meta: contracttest.TenantOnlyMeta{Tenant: "acme"}},
	}); indexErr != nil {
		t.Fatalf("Index(): %v", indexErr)
	}

	byMarker, err := retrieveBM25(context.Background(), idx, "encvacme", retrieval.RetrieveOptions{TopK: 1})
	if err != nil {
		t.Fatalf("Retrieve(marker token): %v", err)
	}
	if byMarker.Len() != 1 || byMarker.Documents()[0].ID != "1" {
		t.Fatalf("Retrieve(marker token) = %#v, want doc 1 via custom codec", byMarker.Documents())
	}

	byPlain, err := retrieveBM25(context.Background(), idx, "acme", retrieval.RetrieveOptions{TopK: 1})
	if err != nil {
		t.Fatalf("Retrieve(plain token): %v", err)
	}
	if !byPlain.IsEmpty() {
		t.Fatalf("Retrieve(plain token) = %#v, want no hits when fieldValue uses custom codec", byPlain.Documents())
	}
}

type markerTenantCodec struct{}

func (markerTenantCodec) Encode(meta contracttest.TenantOnlyMeta) (filter.RawAttributes, error) {
	return filter.RawAttributes{"tenant": "encv" + meta.Tenant}, nil
}

func (markerTenantCodec) Decode(attrs filter.RawAttributes) (contracttest.TenantOnlyMeta, error) {
	raw, ok := attrs["tenant"].(string)
	if !ok {
		return contracttest.TenantOnlyMeta{}, ragy.ErrInvalidArgument
	}
	const prefix = "encv"
	if !strings.HasPrefix(raw, prefix) {
		return contracttest.TenantOnlyMeta{}, ragy.ErrInvalidArgument
	}
	return contracttest.TenantOnlyMeta{Tenant: strings.TrimPrefix(raw, prefix)}, nil
}

func TestNewBM25RejectsDuplicateSearchField(t *testing.T) {
	t.Parallel()

	schema, err := filter.NewSchema().Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}
	_, err = NewBM25Index[struct{}](
		schema,
		Config[struct{}]{SearchFields: []string{"content", "content"}},
		DefaultTokenizer{},
		nil,
	)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("NewBM25Index() error = %v, want invalid argument", err)
	}
}

func TestNewBM25RejectsInvalidSearchField(t *testing.T) {
	t.Parallel()

	schema, err := filter.NewSchema().Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}
	_, err = NewBM25Index[struct{}](
		schema,
		Config[struct{}]{SearchFields: []string{"1bad"}},
		DefaultTokenizer{},
		nil,
	)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("NewBM25Index() error = %v, want invalid argument", err)
	}
}
