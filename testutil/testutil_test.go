package testutil

import (
	"context"
	"testing"

	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/documents"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/internal/contracttest"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/tensor"
)

func TestDocumentStoreConformance(t *testing.T) {
	contracttest.RunDocumentsStoreSuite(
		t,
		func(t *testing.T, docs []retrieval.Document[contracttest.Meta]) documents.Store[contracttest.Meta] {
			t.Helper()
			return &DocumentStore{Docs: docs, FilterSchema: tenantSchema(t)}
		},
	)
}

func TestBackendConformance(t *testing.T) {
	contracttest.RunDenseBackendSuite(
		t,
		func(t *testing.T, docs []retrieval.Document[contracttest.Meta]) retrieval.Backend[contracttest.Meta] {
			t.Helper()
			return &RetrievalBackend{Docs: docs, FilterSchema: tenantSchema(t), VectorRequired: true}
		},
	)

	contracttest.RunLexicalBackendSuite(
		t,
		func(t *testing.T, docs []retrieval.Document[contracttest.Meta]) retrieval.Backend[contracttest.Meta] {
			t.Helper()
			return &RetrievalBackend{Docs: docs, FilterSchema: tenantSchema(t), VectorRequired: false}
		},
	)
}

func TestIndexConformance(t *testing.T) {
	contracttest.RunDenseIndexSuite(t, func(t *testing.T) dense.Index[contracttest.Meta] {
		t.Helper()
		return &DenseIndex{FilterSchema: tenantAgeSchema(t)}
	})

	contracttest.RunTensorIndexSuite(t, func(t *testing.T) tensor.Index[contracttest.Meta] {
		t.Helper()
		return &TensorIndex{FilterSchema: tenantSchema(t)}
	})
}

func TestGraphStoreConformance(t *testing.T) {
	contracttest.RunGraphStoreSuite(
		t,
		func(t *testing.T, snapshot graph.Snapshot[contracttest.Meta], schema graph.Schema) graph.Store[contracttest.Meta] {
			t.Helper()
			return &GraphStore{Snapshot: snapshot, GraphSchema: schema}
		},
	)
}

func TestStructBackendConformance(t *testing.T) {
	contracttest.RunDenseStructBackendSuite(
		t,
		func(t *testing.T, docs []retrieval.Document[contracttest.StructMeta]) retrieval.Backend[contracttest.StructMeta] {
			t.Helper()
			return &StructRetrievalBackend{Docs: docs, FilterSchema: tenantSchema(t), VectorRequired: true}
		},
	)

	contracttest.RunLexicalStructBackendSuite(
		t,
		func(t *testing.T, docs []retrieval.Document[contracttest.StructMeta]) retrieval.Backend[contracttest.StructMeta] {
			t.Helper()
			return &StructRetrievalBackend{Docs: docs, FilterSchema: tenantSchema(t), VectorRequired: false}
		},
	)
}

func TestStructDocumentStoreConformance(t *testing.T) {
	contracttest.RunDocumentsStructStoreSuite(
		t,
		func(t *testing.T, docs []retrieval.Document[contracttest.StructMeta]) documents.Store[contracttest.StructMeta] {
			t.Helper()
			return &StructDocumentStore{Docs: docs, FilterSchema: tenantSchema(t)}
		},
	)
}

func TestBackendsRejectUnsetSchema(t *testing.T) {
	if _, err := (&RetrievalBackend{
		Docs:           []retrieval.Document[contracttest.Meta]{{ID: "doc-1"}},
		VectorRequired: true,
	}).Retrieve(context.Background(), "", retrieval.RetrieveOptions{
		Vector: []float32{1},
	}); err == nil {
		t.Fatal("RetrievalBackend.Retrieve(dense) error = nil, want schema error")
	}

	if _, err := (&RetrievalBackend{
		Docs:           []retrieval.Document[contracttest.Meta]{{ID: "doc-1"}},
		VectorRequired: false,
	}).Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{}); err == nil {
		t.Fatal("RetrievalBackend.Retrieve(lexical) error = nil, want schema error")
	}
}

func TestDocumentStoreRejectsUnsetSchema(t *testing.T) {
	store := &DocumentStore{Docs: []retrieval.Document[contracttest.Meta]{{
		ID:   "doc-1",
		Meta: contracttest.Meta{"tenant": "acme"},
	}}}

	cond := tenantFilterCondition(t)
	if _, err := store.DeleteByFilter(context.Background(), cond); err == nil {
		t.Fatal("DeleteByFilter() error = nil, want schema error")
	}
}

func TestGraphStoreRejectsUnsetSchema(t *testing.T) {
	store := &GraphStore{Snapshot: graph.Snapshot[contracttest.Meta]{
		Nodes: []graph.Node[contracttest.Meta]{{ID: "n1", Labels: []string{"Doc"}}},
	}}

	_, err := store.Traverse(context.Background(), graph.TraversalRequest{
		Seeds:     []string{"n1"},
		Direction: graph.DirectionOutbound,
		Depth:     1,
	})
	if err == nil {
		t.Fatal("Traverse() error = nil, want schema error")
	}
}

func TestDocumentStoreFindByIDsRejectsInvalidOutgoingDocuments(t *testing.T) {
	store := &DocumentStore{
		Docs: []retrieval.Document[contracttest.Meta]{{
			ID:      "doc-1",
			Content: "hello",
			Score:   1.5,
		}},
		FilterSchema: tenantSchema(t),
	}

	if _, err := store.FindByIDs(context.Background(), []string{"doc-1"}); err == nil {
		t.Fatal("FindByIDs() error = nil, want invalid document error")
	}
}

func TestDenseIndexRejectsUnsetSchema(t *testing.T) {
	index := &DenseIndex{}
	err := index.Upsert(context.Background(), []dense.Record[contracttest.Meta]{{ID: "doc-1", Vector: []float32{1}}})
	if err == nil {
		t.Fatal("Upsert() error = nil, want schema error")
	}
}

func TestDenseIndexCanonicalizesStoredAttributes(t *testing.T) {
	index := &DenseIndex{FilterSchema: tenantAgeSchema(t)}
	err := index.Upsert(context.Background(), []dense.Record[contracttest.Meta]{{
		ID:      "doc-1",
		Content: "hello",
		Meta:    contracttest.Meta{"tenant": "acme", "age": int(7)},
		Vector:  []float32{1},
	}})
	if err != nil {
		t.Fatalf("Upsert(): %v", err)
	}

	value, ok := index.Records[0][0].Meta["age"].(int64)
	if !ok || value != 7 {
		t.Fatalf("stored age = %#v, want int64(7)", index.Records[0][0].Meta["age"])
	}
}

func tenantSchema(t *testing.T) filter.Schema {
	t.Helper()
	return contracttest.TenantSchema(t)
}

func tenantAgeSchema(t *testing.T) filter.Schema {
	t.Helper()
	return contracttest.TenantAgeSchema(t)
}

func tenantFilterCondition(t *testing.T) filter.Condition {
	t.Helper()

	schema := tenantSchema(t)
	tenant, err := schema.StringField("tenant")
	if err != nil {
		t.Fatalf("schema.StringField(tenant): %v", err)
	}

	builder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatalf("NewBuilder(): %v", err)
	}

	cond, err := filter.Eq(builder, tenant, "acme").Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	return cond
}
