package qdrant

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/documents"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/internal/contracttest"
	"github.com/skosovsky/ragy/retrieval"
)

type fakeClient struct {
	cond              Condition
	searchLimit       int
	searchPoints      []Point
	searchErr         error
	getPoints         []Point
	upsertPoints      []Point
	upsertCalls       int
	deleteCalls       int
	getErr            error
	deleteByIDsErr    error
	upsertErr         error
	deleteByFilterErr error
}

func (c *fakeClient) Upsert(_ context.Context, _ string, points []Point) error {
	c.upsertCalls++
	c.upsertPoints = append([]Point(nil), points...)
	if c.upsertErr != nil {
		return c.upsertErr
	}
	return nil
}

func (c *fakeClient) Search(_ context.Context, _ string, _ []float32, cond Condition, limit int) ([]Point, error) {
	c.cond = cond
	c.searchLimit = limit
	if c.searchErr != nil {
		return nil, c.searchErr
	}
	return c.searchPoints, nil
}

func (c *fakeClient) Get(_ context.Context, _ string, _ []string) ([]Point, error) {
	if c.getErr != nil {
		return nil, c.getErr
	}
	return c.getPoints, nil
}

func (c *fakeClient) DeleteByIDs(_ context.Context, _ string, _ []string) (int, error) {
	if c.deleteByIDsErr != nil {
		return 0, c.deleteByIDsErr
	}
	return 0, nil
}

func (c *fakeClient) DeleteByFilter(_ context.Context, _ string, cond Condition) (int, error) {
	c.deleteCalls++
	c.cond = cond
	if c.deleteByFilterErr != nil {
		return 0, c.deleteByFilterErr
	}
	return 0, nil
}

func emptySchema(t *testing.T) filter.Schema {
	t.Helper()

	schema, err := filter.NewSchema().Build()
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

func TestRetrievePreservesTypedFilterValue(t *testing.T) {
	client := &fakeClient{}
	builder := filter.NewSchema()
	tenant, err := builder.Int("tenant")
	if err != nil {
		t.Fatalf("builder.Int(): %v", err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}
	store, err := New[contracttest.StructMeta](
		client,
		Config[contracttest.StructMeta]{Collection: "docs", Schema: schema},
		retrieval.NewJSONCodec[contracttest.StructMeta](schema),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	filterBuilder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatalf("NewBuilder(): %v", err)
	}
	cond, err := filter.Eq(filterBuilder, tenant, int64(7)).Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	_, err = store.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
		Vector:  []float32{1},
		TopK:    10,
		Filters: cond,
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}

	eq, ok := client.cond.(EqCondition)
	if !ok {
		t.Fatalf("condition type = %T, want EqCondition", client.cond)
	}

	if _, ok := eq.Value.(int64); !ok {
		t.Fatalf("condition value type = %T, want int64", eq.Value)
	}
}

func TestRetrieveUsesFetchLimitForSearch(t *testing.T) {
	t.Parallel()

	client := &fakeClient{}
	store, err := New[contracttest.StructMeta](
		client,
		Config[contracttest.StructMeta]{Collection: "docs", Schema: emptySchema(t)},
		contracttest.JSONCodec[contracttest.StructMeta](t, emptySchema(t)),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	_, err = store.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
		Vector:     []float32{1},
		FetchLimit: 25,
		TopK:       10,
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if client.searchLimit != 25 {
		t.Fatalf("searchLimit = %d, want 25", client.searchLimit)
	}
}

func TestRetrieveFallsBackToTopKWhenFetchLimitZero(t *testing.T) {
	t.Parallel()

	client := &fakeClient{}
	store, err := New[contracttest.StructMeta](
		client,
		Config[contracttest.StructMeta]{Collection: "docs", Schema: emptySchema(t)},
		contracttest.JSONCodec[contracttest.StructMeta](t, emptySchema(t)),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	_, err = store.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
		Vector: []float32{1},
		TopK:   12,
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if client.searchLimit != 12 {
		t.Fatalf("searchLimit = %d, want 12", client.searchLimit)
	}
}

func TestDenseIndexConformance(t *testing.T) {
	contracttest.RunDenseIndexSuite(t, func(t *testing.T) dense.Index[contracttest.StructMeta] {
		t.Helper()
		schema := contracttest.TenantAgeSchema(t)
		store, err := New[contracttest.StructMeta](&fakeClient{}, Config[contracttest.StructMeta]{
			Collection: "docs",
			Schema:     schema,
		}, contracttest.JSONCodec[contracttest.StructMeta](t, schema))
		if err != nil {
			t.Fatalf("New(): %v", err)
		}
		return store
	})
}

func TestRetrieveReturnsNilMetaWhenPayloadEmpty(t *testing.T) {
	client := &fakeClient{
		searchPoints: []Point{{
			ID:         "doc-1",
			Content:    "hello",
			Attributes: filter.RawAttributes{},
			Score:      3,
		}},
	}

	store, err := New[contracttest.StructMeta](
		client,
		Config[contracttest.StructMeta]{Collection: "docs", Schema: emptySchema(t)},
		contracttest.JSONCodec[contracttest.StructMeta](t, emptySchema(t)),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
		Vector: []float32{1},
		TopK:   10,
	})
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

func TestNewRejectsInvalidCollectionName(t *testing.T) {
	if _, err := New(
		&fakeClient{},
		Config[contracttest.StructMeta]{Collection: "1bad", Schema: emptySchema(t)},
		contracttest.JSONCodec[contracttest.StructMeta](t, emptySchema(t)),
	); err == nil {
		t.Fatal("New() error = nil, want error")
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("New() error = %v, want invalid argument", err)
	}
}

func TestFindByIDsWrapsClientErrorWithErrUnavailable(t *testing.T) {
	raw := errors.New("upstream")
	client := &fakeClient{getErr: raw}
	store, err := New[contracttest.StructMeta](
		client,
		Config[contracttest.StructMeta]{Collection: "docs", Schema: emptySchema(t)},
		contracttest.JSONCodec[contracttest.StructMeta](t, emptySchema(t)),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, err = store.FindByIDs(context.Background(), []string{"a"})
	if err == nil {
		t.Fatal("FindByIDs() error = nil, want error")
	}
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("FindByIDs() error = %v, want errors.Is(..., ErrUnavailable)", err)
	}
	if !errors.Is(err, raw) {
		t.Fatalf("error chain lost upstream: %v", err)
	}
}

func TestDeleteByIDsWrapsClientErrorWithErrUnavailable(t *testing.T) {
	raw := errors.New("upstream")
	client := &fakeClient{deleteByIDsErr: raw}
	store, err := New[contracttest.StructMeta](
		client,
		Config[contracttest.StructMeta]{Collection: "docs", Schema: emptySchema(t)},
		contracttest.JSONCodec[contracttest.StructMeta](t, emptySchema(t)),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, err = store.DeleteByIDs(context.Background(), []string{"a"})
	if err == nil {
		t.Fatal("DeleteByIDs() error = nil, want error")
	}
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("DeleteByIDs() error = %v, want errors.Is(..., ErrUnavailable)", err)
	}
	if !errors.Is(err, raw) {
		t.Fatalf("error chain lost upstream: %v", err)
	}
}

func TestUpsertWrapsRawClientError(t *testing.T) {
	t.Parallel()

	raw := errors.New("upstream")
	client := &fakeClient{upsertErr: raw}
	store, err := New[contracttest.StructMeta](
		client,
		Config[contracttest.StructMeta]{Collection: "docs", Schema: emptySchema(t)},
		contracttest.JSONCodec[contracttest.StructMeta](t, emptySchema(t)),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	err = store.Upsert(context.Background(), []dense.Record[contracttest.StructMeta]{{
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

func TestDeleteByFilterWrapsRawClientError(t *testing.T) {
	t.Parallel()

	raw := errors.New("upstream")
	client := &fakeClient{deleteByFilterErr: raw}

	builder := filter.NewSchema()
	tenant, err := builder.Int("tenant")
	if err != nil {
		t.Fatalf("builder.Int(tenant): %v", err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}
	store, err := New[contracttest.StructMeta](
		client,
		Config[contracttest.StructMeta]{Collection: "docs", Schema: schema},
		retrieval.NewJSONCodec[contracttest.StructMeta](schema),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	filterBuilder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatalf("NewBuilder(): %v", err)
	}
	cond, err := filter.Eq(filterBuilder, tenant, int64(7)).Build()
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

func TestUpsertRejectsWrongMetaTypeBeforeWrite(t *testing.T) {
	client := &fakeClient{}
	store, err := New[contracttest.StructMeta](
		client,
		Config[contracttest.StructMeta]{Collection: "docs", Schema: ageSchema(t)},
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
	if client.upsertCalls != 0 {
		t.Fatalf("upsertCalls = %d, want 0", client.upsertCalls)
	}
}

func TestUpsertCanonicalizesMetaBeforeClientCall(t *testing.T) {
	client := &fakeClient{}
	store, err := New[contracttest.StructMeta](
		client,
		Config[contracttest.StructMeta]{Collection: "docs", Schema: ageScoreSchema(t)},
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

	if client.upsertCalls != 1 {
		t.Fatalf("upsertCalls = %d, want 1", client.upsertCalls)
	}
	if len(client.upsertPoints) != 1 {
		t.Fatalf("len(upsertPoints) = %d, want 1", len(client.upsertPoints))
	}

	if value, ok := client.upsertPoints[0].Attributes["age"].(int64); !ok || value != 7 {
		t.Fatalf("age attr = %#v, want int64(7)", client.upsertPoints[0].Attributes["age"])
	}
	if value, ok := client.upsertPoints[0].Attributes["score"].(float64); !ok || value != 1.5 {
		t.Fatalf("score attr = %#v, want float64(1.5)", client.upsertPoints[0].Attributes["score"])
	}
}

func TestUpsertFindByIDsRoundTrip(t *testing.T) {
	t.Parallel()

	client := &fakeClient{}
	schema := contracttest.TenantSchema(t)
	store, err := New[contracttest.TenantOnlyMeta](
		client,
		Config[contracttest.TenantOnlyMeta]{Collection: "docs", Schema: schema},
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

	client.getPoints = client.upsertPoints
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

	store, err := New[contracttest.StructMeta](
		&fakeClient{},
		Config[contracttest.StructMeta]{Collection: "docs", Schema: emptySchema(t)},
		contracttest.JSONCodec[contracttest.StructMeta](t, emptySchema(t)),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "q", retrieval.RetrieveOptions{TopK: 1})
	contracttest.RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrEmptyVector) {
		t.Fatalf("Retrieve() error = %v, want empty vector", err)
	}
}

func TestUpsertCanonicalizesEmptyMetaToNil(t *testing.T) {
	client := &fakeClient{}
	store, err := New[contracttest.StructMeta](
		client,
		Config[contracttest.StructMeta]{Collection: "docs", Schema: emptySchema(t)},
		contracttest.JSONCodec[contracttest.StructMeta](t, emptySchema(t)),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	err = store.Upsert(context.Background(), []dense.Record[contracttest.StructMeta]{{
		ID:      "doc-1",
		Content: "hello",
		Vector:  []float32{1},
	}})
	if err != nil {
		t.Fatalf("Upsert(): %v", err)
	}

	if got := client.upsertPoints[0].Attributes; got != nil {
		t.Fatalf("Attributes = %#v, want nil", got)
	}
}

func TestRetrieveRejectsUndeclaredFilterField(t *testing.T) {
	client := &fakeClient{}
	store, err := New[contracttest.StructMeta](
		client,
		Config[contracttest.StructMeta]{Collection: "docs", Schema: emptySchema(t)},
		contracttest.JSONCodec[contracttest.StructMeta](t, emptySchema(t)),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	foreign := filter.NewSchema()
	tenant, err := foreign.Int("other")
	if err != nil {
		t.Fatalf("foreign.Int(other): %v", err)
	}
	foreignSchema, err := foreign.Build()
	if err != nil {
		t.Fatalf("foreign.Build(): %v", err)
	}
	filterBuilder, err := filter.NewBuilder(foreignSchema)
	if err != nil {
		t.Fatalf("NewBuilder(): %v", err)
	}
	cond, err := filter.Eq(filterBuilder, tenant, int64(7)).Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
		Vector:  []float32{1},
		Filters: cond,
	})
	contracttest.RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Retrieve() error = %v, want invalid argument", err)
	}
	if client.cond != nil {
		t.Fatalf("condition = %#v, want no backend search call", client.cond)
	}
}

type documentsClient struct {
	docs map[string]retrieval.Document[contracttest.StructMeta]
}

func newDocumentsClient(docs []retrieval.Document[contracttest.StructMeta]) *documentsClient {
	out := make(map[string]retrieval.Document[contracttest.StructMeta], len(docs))
	for _, doc := range docs {
		out[doc.ID] = retrieval.Document[contracttest.StructMeta]{
			ID:      doc.ID,
			Content: doc.Content,
			Meta:    doc.Meta,
			Score:   doc.Score,
		}
	}
	return &documentsClient{docs: out}
}

func (c *documentsClient) Upsert(_ context.Context, _ string, _ []Point) error { return nil }

func (c *documentsClient) Search(_ context.Context, _ string, _ []float32, _ Condition, _ int) ([]Point, error) {
	return nil, nil
}

func (c *documentsClient) Get(_ context.Context, _ string, ids []string) ([]Point, error) {
	points := make([]Point, 0, len(ids))
	for _, id := range ids {
		doc, ok := c.docs[id]
		if !ok {
			continue
		}
		attrs, err := structMetaToRawAttributes(doc.Meta)
		if err != nil {
			return nil, err
		}
		points = append(points, Point{
			ID:         doc.ID,
			Content:    doc.Content,
			Attributes: attrs,
		})
	}
	if len(points) == 0 {
		return nil, nil
	}
	return points, nil
}

func (c *documentsClient) DeleteByIDs(_ context.Context, _ string, ids []string) (int, error) {
	deleted := 0
	for _, id := range ids {
		if _, ok := c.docs[id]; !ok {
			continue
		}
		delete(c.docs, id)
		deleted++
	}
	return deleted, nil
}

func (c *documentsClient) DeleteByFilter(_ context.Context, _ string, cond Condition) (int, error) {
	deleted := 0
	for id, doc := range c.docs {
		matched, err := matchesCondition(doc, cond)
		if err != nil {
			return 0, err
		}
		if !matched {
			continue
		}
		delete(c.docs, id)
		deleted++
	}
	return deleted, nil
}

func TestDocumentsStoreConformance(t *testing.T) {
	contracttest.RunDocumentsStructStoreSuite(
		t,
		func(t *testing.T, docs []retrieval.Document[contracttest.StructMeta]) documents.Store[contracttest.StructMeta] {
			t.Helper()
			builder := filter.NewSchema()
			if _, err := builder.String("tenant"); err != nil {
				t.Fatalf("builder.String(tenant): %v", err)
			}
			schema, err := builder.Build()
			if err != nil {
				t.Fatalf("Build(): %v", err)
			}
			store, err := New[contracttest.StructMeta](
				newDocumentsClient(docs),
				Config[contracttest.StructMeta]{Collection: "docs", Schema: schema},
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

		client := &fakeClient{
			getPoints: []Point{
				{ID: "ok", Content: "good", Attributes: filter.RawAttributes{}},
				{ID: "", Content: "bad", Attributes: filter.RawAttributes{}},
			},
		}
		store, err := New[contracttest.StructMeta](
			client,
			Config[contracttest.StructMeta]{Collection: "docs", Schema: emptySchema(t)},
			contracttest.JSONCodec[contracttest.StructMeta](t, emptySchema(t)),
		)
		if err != nil {
			t.Fatalf("New(): %v", err)
		}
		return store
	})
}

func TestRetrievePartialProjectionConformance(t *testing.T) {
	contracttest.RunRetrievePartialProjectionSuite(t, func(t *testing.T) retrieval.Backend[contracttest.StructMeta] {
		t.Helper()

		schema := emptySchema(t)
		client := &fakeClient{
			searchPoints: []Point{
				{ID: "ok", Content: "good", Attributes: filter.RawAttributes{}, Score: 0.9},
				{ID: "", Content: "bad", Attributes: filter.RawAttributes{}, Score: 0.5},
			},
		}
		store, err := New[contracttest.StructMeta](
			client,
			Config[contracttest.StructMeta]{Collection: "docs", Schema: schema},
			contracttest.JSONCodec[contracttest.StructMeta](t, schema),
		)
		if err != nil {
			t.Fatalf("New(): %v", err)
		}
		return store
	}, func(t *testing.T) retrieval.Backend[contracttest.StructMeta] {
		t.Helper()

		schema := contracttest.TenantSchema(t)
		resolver := contracttest.ContentMergeResolver[contracttest.StructMeta]{}
		client := &fakeClient{
			searchPoints: []Point{
				{ID: "ok", Content: "merge-key", Attributes: filter.RawAttributes{"tenant": "acme"}, Score: 0.9},
				{ID: "bad", Content: "bad", Attributes: filter.RawAttributes{"tenant": 123}, Score: 0.5},
			},
		}
		store, err := New[contracttest.StructMeta](
			client,
			Config[contracttest.StructMeta]{
				Collection: "docs",
				Schema:     schema,
				Resolver:   resolver,
			},
			contracttest.JSONCodec[contracttest.StructMeta](t, schema),
		)
		if err != nil {
			t.Fatalf("New(): %v", err)
		}
		return store
	})
}

func matchesCondition(doc retrieval.Document[contracttest.StructMeta], cond Condition) (bool, error) {
	switch node := cond.(type) {
	case MatchAllCondition:
		return true, nil
	case EqCondition:
		return matchesEquality(doc, node.Field, node.Value)
	case NeqCondition:
		return matchesInequality(doc, node.Field, node.Value)
	case RangeCondition:
		return matchesRange(doc, node)
	case InCondition:
		return matchesIn(doc, node)
	case GroupCondition:
		return matchesGroup(doc, node)
	case NotCondition:
		return matchesNot(doc, node.Item)
	default:
		return false, fmt.Errorf("unsupported qdrant condition %T", cond)
	}
}

func matchesEquality(doc retrieval.Document[contracttest.StructMeta], field string, expected any) (bool, error) {
	value, ok := documentField(doc, field)
	return ok && value == expected, nil
}

func matchesInequality(doc retrieval.Document[contracttest.StructMeta], field string, expected any) (bool, error) {
	value, ok := documentField(doc, field)
	return !ok || value != expected, nil
}

func matchesRange(doc retrieval.Document[contracttest.StructMeta], cond RangeCondition) (bool, error) {
	value, ok := documentField(doc, cond.Field)
	if !ok {
		return false, nil
	}
	return compareRange(value, cond.Value, cond.Op)
}

func matchesIn(doc retrieval.Document[contracttest.StructMeta], cond InCondition) (bool, error) {
	value, ok := documentField(doc, cond.Field)
	if !ok {
		return false, nil
	}
	return slices.Contains(cond.Values, value), nil
}

func matchesGroup(doc retrieval.Document[contracttest.StructMeta], cond GroupCondition) (bool, error) {
	switch cond.Op {
	case "and":
		return matchesAll(doc, cond.Items)
	case "or":
		return matchesAny(doc, cond.Items)
	default:
		return false, fmt.Errorf("unknown group op %q", cond.Op)
	}
}

func matchesAll(doc retrieval.Document[contracttest.StructMeta], items []Condition) (bool, error) {
	for _, item := range items {
		matched, err := matchesCondition(doc, item)
		if err != nil || !matched {
			return matched, err
		}
	}
	return true, nil
}

func matchesAny(doc retrieval.Document[contracttest.StructMeta], items []Condition) (bool, error) {
	for _, item := range items {
		matched, err := matchesCondition(doc, item)
		if err != nil {
			return false, err
		}
		if matched {
			return true, nil
		}
	}
	return false, nil
}

func matchesNot(doc retrieval.Document[contracttest.StructMeta], cond Condition) (bool, error) {
	matched, err := matchesCondition(doc, cond)
	return !matched, err
}

func structMetaToRawAttributes(meta contracttest.StructMeta) (filter.RawAttributes, error) {
	data, err := json.Marshal(meta)
	if err != nil {
		return nil, err
	}
	var attrs filter.RawAttributes
	if err := json.Unmarshal(data, &attrs); err != nil {
		return nil, err
	}
	if len(attrs) == 0 {
		return filter.RawAttributes{}, nil
	}
	return attrs, nil
}

func documentField(doc retrieval.Document[contracttest.StructMeta], field string) (any, bool) {
	switch field {
	case "id":
		return doc.ID, true
	case "content":
		return doc.Content, true
	case "tenant":
		if doc.Meta.Tenant == "" {
			return nil, false
		}
		return doc.Meta.Tenant, true
	case "age":
		if doc.Meta.Age == 0 {
			return nil, false
		}
		return doc.Meta.Age, true
	case "score":
		if doc.Meta.Score == 0 {
			return nil, false
		}
		return doc.Meta.Score, true
	case "kind":
		if doc.Meta.Kind == "" {
			return nil, false
		}
		return doc.Meta.Kind, true
	default:
		return nil, false
	}
}

func compareRange(left any, right any, op string) (bool, error) {
	lv, lok := toFloat(left)
	rv, rok := toFloat(right)
	if !lok || !rok {
		return false, nil
	}
	switch op {
	case "gt":
		return lv > rv, nil
	case "gte":
		return lv >= rv, nil
	case "lt":
		return lv < rv, nil
	case "lte":
		return lv <= rv, nil
	default:
		return false, fmt.Errorf("unknown range op %q", op)
	}
}

func toFloat(value any) (float64, bool) {
	switch v := value.(type) {
	case int:
		return float64(v), true
	case int64:
		return float64(v), true
	case float64:
		return v, true
	default:
		return 0, false
	}
}

func TestRetrieveUnmarshalsStructMeta(t *testing.T) {
	t.Parallel()

	client := &fakeClient{
		searchPoints: []Point{{
			ID:         "doc-1",
			Content:    "hello",
			Attributes: filter.RawAttributes{"tenant": "acme"},
			Vector:     []float32{1},
			Score:      0.9,
		}},
	}

	schema := contracttest.TenantSchema(t)
	store, err := New[contracttest.TenantOnlyMeta](client, Config[contracttest.TenantOnlyMeta]{
		Collection: "docs",
		Schema:     schema,
	}, contracttest.JSONCodec[contracttest.TenantOnlyMeta](t, schema))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
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

	client := &fakeClient{
		searchPoints: []Point{{
			ID:         "doc-1",
			Content:    "hello",
			Attributes: filter.RawAttributes{"tenant": 123},
			Vector:     []float32{1},
			Score:      0.9,
		}},
	}

	schema := contracttest.TenantSchema(t)
	store, err := New[contracttest.TenantOnlyMeta](client, Config[contracttest.TenantOnlyMeta]{
		Collection: "docs",
		Schema:     schema,
	}, contracttest.JSONCodec[contracttest.TenantOnlyMeta](t, schema))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
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

	client := &fakeClient{
		getPoints: []Point{{
			ID:         "doc-1",
			Content:    "hello",
			Attributes: filter.RawAttributes{"tenant": "acme"},
		}},
	}

	schema := contracttest.TenantSchema(t)
	store, err := New[contracttest.TenantOnlyMeta](client, Config[contracttest.TenantOnlyMeta]{
		Collection: "docs",
		Schema:     schema,
	}, contracttest.JSONCodec[contracttest.TenantOnlyMeta](t, schema))
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

	client := &fakeClient{
		getPoints: []Point{{
			ID:         "doc-1",
			Content:    "hello",
			Attributes: filter.RawAttributes{"tenant": 123},
		}},
	}

	schema := contracttest.TenantSchema(t)
	store, err := New[contracttest.TenantOnlyMeta](client, Config[contracttest.TenantOnlyMeta]{
		Collection: "docs",
		Schema:     schema,
	}, contracttest.JSONCodec[contracttest.TenantOnlyMeta](t, schema))
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
) retrieval.Backend[contracttest.StructMeta] {
	t.Helper()

	schema := contracttest.TenantAgeSchema(t)
	codec := retrieval.NewJSONCodec[contracttest.StructMeta](schema)
	points := make([]Point, 0, len(docs))
	for _, doc := range docs {
		attrs, err := codec.Encode(doc.Meta)
		if err != nil {
			t.Fatalf("Encode(): %v", err)
		}
		points = append(points, Point{
			ID:         doc.ID,
			Content:    doc.Content,
			Attributes: attrs,
			Score:      1,
		})
	}

	client := &fakeClient{searchPoints: points}
	store, err := New[contracttest.StructMeta](
		client,
		Config[contracttest.StructMeta]{Collection: "docs", Schema: schema},
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
	contracttest.RunRetrieveOptionsInvalidSuite(t, func(t *testing.T) retrieval.Backend[contracttest.StructMeta] {
		t.Helper()
		store, err := New[contracttest.StructMeta](
			&fakeClient{},
			Config[contracttest.StructMeta]{Collection: "docs", Schema: emptySchema(t)},
			contracttest.JSONCodec[contracttest.StructMeta](t, emptySchema(t)),
		)
		if err != nil {
			t.Fatalf("New(): %v", err)
		}
		return store
	})
}

func TestNewRejectsNilCodec(t *testing.T) {
	if _, err := New[contracttest.StructMeta](
		&fakeClient{},
		Config[contracttest.StructMeta]{Collection: "docs", Schema: emptySchema(t)},
		nil,
	); err == nil {
		t.Fatal("New() error = nil, want error")
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("New() error = %v, want invalid argument", err)
	}
}

func TestFindByIDsPreservesPartialOnProjectError(t *testing.T) {
	t.Parallel()

	client := &fakeClient{
		getPoints: []Point{
			{ID: "ok", Content: "good", Attributes: filter.RawAttributes{}},
			{ID: "", Content: "bad", Attributes: filter.RawAttributes{}},
		},
	}
	store, err := New[contracttest.StructMeta](
		client,
		Config[contracttest.StructMeta]{Collection: "docs", Schema: emptySchema(t)},
		contracttest.JSONCodec[contracttest.StructMeta](t, emptySchema(t)),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.FindByIDs(context.Background(), []string{"ok", "bad"})
	if !errors.Is(err, ragy.ErrMissingID) {
		t.Fatalf("FindByIDs() error = %v, want missing id", err)
	}
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("FindByIDs() error = %v, want protocol", err)
	}
	if len(out) != 1 || out[0].ID != "ok" {
		t.Fatalf("FindByIDs() = %#v, want partial ok point", out)
	}
}

func TestRetrieveSearchErrorReturnsEmptyResultSet(t *testing.T) {
	t.Parallel()

	client := &fakeClient{searchErr: ragy.ErrUnavailable}
	store, err := New[contracttest.StructMeta](
		client,
		Config[contracttest.StructMeta]{Collection: "docs", Schema: emptySchema(t)},
		contracttest.JSONCodec[contracttest.StructMeta](t, emptySchema(t)),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
		Vector: []float32{1},
		TopK:   5,
	})
	contracttest.RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want unavailable", err)
	}
}

func TestRetrieveWrapsRawClientError(t *testing.T) {
	t.Parallel()

	raw := errors.New("upstream")
	client := &fakeClient{searchErr: raw}
	store, err := New[contracttest.StructMeta](
		client,
		Config[contracttest.StructMeta]{Collection: "docs", Schema: emptySchema(t)},
		contracttest.JSONCodec[contracttest.StructMeta](t, emptySchema(t)),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
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
