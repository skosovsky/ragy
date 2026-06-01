package qdrant

import (
	"context"
	"errors"
	"fmt"
	"maps"
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
	cond           Condition
	searchPoints   []Point
	getPoints      []Point
	upsertPoints   []Point
	upsertCalls    int
	deleteCalls    int
	getErr         error
	deleteByIDsErr error
}

func (c *fakeClient) Upsert(_ context.Context, _ string, points []Point) error {
	c.upsertCalls++
	c.upsertPoints = append([]Point(nil), points...)
	return nil
}

func (c *fakeClient) Search(_ context.Context, _ string, _ []float32, cond Condition, _ int) ([]Point, error) {
	c.cond = cond
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
	store, err := New[contracttest.Meta](client, Config{Collection: "docs", Schema: schema})
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

func TestDenseIndexConformance(t *testing.T) {
	contracttest.RunDenseIndexSuite(t, func(t *testing.T) dense.Index[contracttest.Meta] {
		t.Helper()
		store, err := New[contracttest.Meta](&fakeClient{}, Config{
			Collection: "docs",
			Schema:     contracttest.TenantAgeSchema(t),
		})
		if err != nil {
			t.Fatalf("New(): %v", err)
		}
		return store
	})
}

func TestDeleteByFilterRejectsEmpty(t *testing.T) {
	client := &fakeClient{}
	store, err := New[contracttest.Meta](client, Config{Collection: "docs", Schema: emptySchema(t)})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	if _, err := store.DeleteByFilter(context.Background(), filter.Condition{}); err == nil {
		t.Fatal("DeleteByFilter() error = nil, want error")
	}

	if client.deleteCalls != 0 {
		t.Fatalf("deleteCalls = %d, want 0", client.deleteCalls)
	}
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

	store, err := New[contracttest.Meta](client, Config{Collection: "docs", Schema: emptySchema(t)})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
		Vector: []float32{1},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}

	if len(out) != 1 {
		t.Fatalf("len(out) = %d, want 1", len(out))
	}

	if len(out[0].Meta) != 0 {
		t.Fatalf("document meta = %#v, want empty", out[0].Meta)
	}
}

func TestNewRejectsInvalidCollectionName(t *testing.T) {
	if _, err := New[contracttest.Meta](&fakeClient{}, Config{Collection: "1bad", Schema: emptySchema(t)}); err == nil {
		t.Fatal("New() error = nil, want error")
	}
}

func TestFindByIDsRejectsInvalidBackendPayload(t *testing.T) {
	client := &fakeClient{
		getPoints: []Point{{
			ID:      "",
			Content: "broken",
		}},
	}
	store, err := New[contracttest.Meta](client, Config{Collection: "docs", Schema: emptySchema(t)})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	if _, err := store.FindByIDs(context.Background(), []string{"doc-1"}); err == nil {
		t.Fatal("FindByIDs() error = nil, want error")
	}
}

func TestFindByIDsWrapsClientErrorWithErrUnavailable(t *testing.T) {
	client := &fakeClient{getErr: errors.New("upstream")}
	store, err := New[contracttest.Meta](client, Config{Collection: "docs", Schema: emptySchema(t)})
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
}

func TestDeleteByIDsWrapsClientErrorWithErrUnavailable(t *testing.T) {
	client := &fakeClient{deleteByIDsErr: errors.New("upstream")}
	store, err := New[contracttest.Meta](client, Config{Collection: "docs", Schema: emptySchema(t)})
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
}

func TestUpsertRejectsWrongMetaTypeBeforeWrite(t *testing.T) {
	client := &fakeClient{}
	store, err := New[contracttest.Meta](client, Config{Collection: "docs", Schema: ageSchema(t)})
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
	if client.upsertCalls != 0 {
		t.Fatalf("upsertCalls = %d, want 0", client.upsertCalls)
	}
}

func TestUpsertCanonicalizesMetaBeforeClientCall(t *testing.T) {
	client := &fakeClient{}
	store, err := New[contracttest.Meta](client, Config{Collection: "docs", Schema: ageScoreSchema(t)})
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

func TestUpsertCanonicalizesEmptyMetaToNil(t *testing.T) {
	client := &fakeClient{}
	store, err := New[contracttest.Meta](client, Config{Collection: "docs", Schema: emptySchema(t)})
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

	if got := client.upsertPoints[0].Attributes; got != nil {
		t.Fatalf("Attributes = %#v, want nil", got)
	}
}

func TestRetrieveRejectsUndeclaredFilterField(t *testing.T) {
	client := &fakeClient{}
	store, err := New[contracttest.Meta](client, Config{Collection: "docs", Schema: emptySchema(t)})
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

	if _, err := store.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
		Vector:  []float32{1},
		Filters: cond,
	}); err == nil {
		t.Fatal("Retrieve() error = nil, want error")
	}
	if client.cond != nil {
		t.Fatalf("condition = %#v, want no backend search call", client.cond)
	}
}

type documentsClient struct {
	docs map[string]retrieval.Document[contracttest.Meta]
}

func newDocumentsClient(docs []retrieval.Document[contracttest.Meta]) *documentsClient {
	out := make(map[string]retrieval.Document[contracttest.Meta], len(docs))
	for _, doc := range docs {
		out[doc.ID] = retrieval.Document[contracttest.Meta]{
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
		attrs := filter.RawAttributes{}
		maps.Copy(attrs, doc.Meta)
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
	contracttest.RunDocumentsStoreSuite(
		t,
		func(t *testing.T, docs []retrieval.Document[contracttest.Meta]) documents.Store[contracttest.Meta] {
			t.Helper()
			builder := filter.NewSchema()
			if _, err := builder.String("tenant"); err != nil {
				t.Fatalf("builder.String(tenant): %v", err)
			}
			schema, err := builder.Build()
			if err != nil {
				t.Fatalf("Build(): %v", err)
			}
			store, err := New[contracttest.Meta](newDocumentsClient(docs), Config{Collection: "docs", Schema: schema})
			if err != nil {
				t.Fatalf("New(): %v", err)
			}
			return store
		},
	)
}

func matchesCondition(doc retrieval.Document[contracttest.Meta], cond Condition) (bool, error) {
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

func matchesEquality(doc retrieval.Document[contracttest.Meta], field string, expected any) (bool, error) {
	value, ok := documentField(doc, field)
	return ok && value == expected, nil
}

func matchesInequality(doc retrieval.Document[contracttest.Meta], field string, expected any) (bool, error) {
	value, ok := documentField(doc, field)
	return !ok || value != expected, nil
}

func matchesRange(doc retrieval.Document[contracttest.Meta], cond RangeCondition) (bool, error) {
	value, ok := documentField(doc, cond.Field)
	if !ok {
		return false, nil
	}
	return compareRange(value, cond.Value, cond.Op)
}

func matchesIn(doc retrieval.Document[contracttest.Meta], cond InCondition) (bool, error) {
	value, ok := documentField(doc, cond.Field)
	if !ok {
		return false, nil
	}
	return slices.Contains(cond.Values, value), nil
}

func matchesGroup(doc retrieval.Document[contracttest.Meta], cond GroupCondition) (bool, error) {
	switch cond.Op {
	case "and":
		return matchesAll(doc, cond.Items)
	case "or":
		return matchesAny(doc, cond.Items)
	default:
		return false, fmt.Errorf("unknown group op %q", cond.Op)
	}
}

func matchesAll(doc retrieval.Document[contracttest.Meta], items []Condition) (bool, error) {
	for _, item := range items {
		matched, err := matchesCondition(doc, item)
		if err != nil || !matched {
			return matched, err
		}
	}
	return true, nil
}

func matchesAny(doc retrieval.Document[contracttest.Meta], items []Condition) (bool, error) {
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

func matchesNot(doc retrieval.Document[contracttest.Meta], cond Condition) (bool, error) {
	matched, err := matchesCondition(doc, cond)
	return !matched, err
}

func documentField(doc retrieval.Document[contracttest.Meta], field string) (any, bool) {
	switch field {
	case "id":
		return doc.ID, true
	case "content":
		return doc.Content, true
	default:
		value, ok := doc.Meta[field]
		return value, ok
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

type tenantMeta struct {
	Tenant string `json:"tenant"`
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

	store, err := New[tenantMeta](client, Config{
		Collection: "docs",
		Schema:     contracttest.TenantSchema(t),
	})
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

	client := &fakeClient{
		searchPoints: []Point{{
			ID:         "doc-1",
			Content:    "hello",
			Attributes: filter.RawAttributes{"tenant": 123},
			Vector:     []float32{1},
			Score:      0.9,
		}},
	}

	store, err := New[tenantMeta](client, Config{
		Collection: "docs",
		Schema:     contracttest.TenantSchema(t),
	})
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

	client := &fakeClient{
		getPoints: []Point{{
			ID:         "doc-1",
			Content:    "hello",
			Attributes: filter.RawAttributes{"tenant": "acme"},
		}},
	}

	store, err := New[tenantMeta](client, Config{
		Collection: "docs",
		Schema:     contracttest.TenantSchema(t),
	})
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

	store, err := New[tenantMeta](client, Config{
		Collection: "docs",
		Schema:     contracttest.TenantSchema(t),
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	_, err = store.FindByIDs(context.Background(), []string{"doc-1"})
	if err == nil {
		t.Fatal("FindByIDs() error = nil, want decode error")
	}
}
