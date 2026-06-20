package testutil

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/documents"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/internal/contracttest"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/tensor"
)

func TestDocumentStoreConformance(t *testing.T) {
	contracttest.RunDocumentsStructStoreSuite(
		t,
		func(t *testing.T, docs []retrieval.Document[contracttest.StructMeta]) documents.Store[contracttest.StructMeta] {
			t.Helper()
			return &DocumentStore{Docs: docs, FilterSchema: tenantSchema(t)}
		},
	)
}

func TestBackendConformance(t *testing.T) {
	contracttest.RunDenseStructBackendSuite(
		t,
		func(t *testing.T, docs []retrieval.Document[contracttest.StructMeta]) retrieval.Backend[struct{}, contracttest.StructMeta] {
			t.Helper()
			return &RetrievalBackend{Docs: docs, FilterSchema: tenantSchema(t), VectorRequired: true}
		},
	)

	contracttest.RunLexicalStructBackendSuite(
		t,
		func(t *testing.T, docs []retrieval.Document[contracttest.StructMeta]) retrieval.Backend[struct{}, contracttest.StructMeta] {
			t.Helper()
			return &RetrievalBackend{Docs: docs, FilterSchema: tenantSchema(t), VectorRequired: false}
		},
	)
}

func TestRetrieveOptionsInvalidConformance(t *testing.T) {
	t.Parallel()

	contracttest.RunRetrieveOptionsInvalidSuite(
		t,
		func(t *testing.T) retrieval.Backend[struct{}, contracttest.StructMeta] {
			t.Helper()
			return &RetrievalBackend{FilterSchema: tenantSchema(t), VectorRequired: true}
		},
	)
}

func TestIndexConformance(t *testing.T) {
	contracttest.RunDenseIndexSuite(t, func(t *testing.T) dense.Index[contracttest.StructMeta] {
		t.Helper()
		return &DenseIndex{FilterSchema: tenantAgeSchema(t)}
	})

	contracttest.RunTensorIndexSuite(t, func(t *testing.T) tensor.Index[contracttest.StructMeta] {
		t.Helper()
		return &TensorIndex{FilterSchema: tenantSchema(t)}
	})
}

func TestGraphStoreConformance(t *testing.T) {
	contracttest.RunGraphStoreSuite(
		t,
		func(t *testing.T, snapshot graph.Snapshot[contracttest.StructMeta], schema graph.Schema) graph.Store[contracttest.StructMeta] {
			t.Helper()
			return &GraphStore{Snapshot: snapshot, GraphSchema: schema}
		},
	)
}

func TestBackendsRejectUnsetSchema(t *testing.T) {
	if _, err := (&RetrievalBackend{
		Docs:           []retrieval.Document[contracttest.StructMeta]{{ID: "doc-1"}},
		VectorRequired: true,
	}).Retrieve(context.Background(), retrieval.Query[struct{}]{
		Options: retrieval.RetrieveOptions{Vector: []float32{1}},
	}); err == nil {
		t.Fatal("RetrievalBackend.Retrieve(dense) error = nil, want schema error")
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("RetrievalBackend.Retrieve(dense) error = %v, want invalid argument", err)
	}

	if _, err := (&RetrievalBackend{
		Docs:           []retrieval.Document[contracttest.StructMeta]{{ID: "doc-1"}},
		VectorRequired: false,
	}).Retrieve(context.Background(), retrieval.Query[struct{}]{Text: "hello"}); err == nil {
		t.Fatal("RetrievalBackend.Retrieve(lexical) error = nil, want schema error")
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("RetrievalBackend.Retrieve(lexical) error = %v, want invalid argument", err)
	}
}

func TestDocumentStoreRejectsUnsetSchema(t *testing.T) {
	store := &StructDocumentStore{Docs: []retrieval.Document[contracttest.StructMeta]{{
		ID:   "doc-1",
		Meta: contracttest.StructMeta{Tenant: "acme"},
	}}}

	cond := tenantFilterCondition(t)
	if _, err := store.DeleteByFilter(context.Background(), cond); err == nil {
		t.Fatal("DeleteByFilter() error = nil, want schema error")
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("DeleteByFilter() error = %v, want invalid argument", err)
	}
}

func TestGraphStoreRejectsUnsetSchema(t *testing.T) {
	store := &GraphStore{Snapshot: graph.Snapshot[contracttest.StructMeta]{
		Nodes: []graph.Node[contracttest.StructMeta]{{ID: "n1", Labels: []string{"Doc"}}},
	}}

	_, err := store.Traverse(context.Background(), graph.TraversalRequest{
		Seeds:     []string{"n1"},
		Direction: graph.DirectionOutbound,
		Depth:     1,
	})
	if err == nil {
		t.Fatal("Traverse() error = nil, want schema error")
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Traverse() error = %v, want invalid argument", err)
	}
}

func TestDocumentStoreFindByIDsRejectsInvalidOutgoingDocuments(t *testing.T) {
	store := &StructDocumentStore{
		Docs: []retrieval.Document[contracttest.StructMeta]{{
			ID:      "doc-1",
			Content: "hello",
			Score:   1.5,
		}},
		FilterSchema: tenantSchema(t),
	}

	if _, err := store.FindByIDs(context.Background(), []string{"doc-1"}); err == nil {
		t.Fatal("FindByIDs() error = nil, want invalid document error")
	} else if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("FindByIDs() error = %v, want protocol", err)
	}
}

func TestDocumentsPartialFindByIDsConformance(t *testing.T) {
	contracttest.RunDocumentsPartialFindByIDsSuite(t, func(t *testing.T) documents.Store[contracttest.StructMeta] {
		return &DocumentStore{
			Docs: []retrieval.Document[contracttest.StructMeta]{
				{ID: "ok", Content: "good", Meta: contracttest.StructMeta{Tenant: "acme"}},
				{ID: "bad", Content: "bad", Score: 1.5},
			},
			FilterSchema: tenantSchema(t),
		}
	})
}

func TestRetrievePartialProjectionConformance(t *testing.T) {
	contracttest.RunRetrievePartialProjectionSuite(
		t,
		func(t *testing.T) retrieval.Backend[struct{}, contracttest.StructMeta] {
			t.Helper()

			return &StructRetrievalBackend{
				Docs: []retrieval.Document[contracttest.StructMeta]{
					{ID: "ok", Content: "good", Meta: contracttest.StructMeta{Tenant: "acme"}},
					{ID: "bad", Content: "bad", Score: 1.5},
				},
				FilterSchema: tenantSchema(t),
			}
		},
		func(t *testing.T) retrieval.Backend[struct{}, contracttest.StructMeta] {
			t.Helper()

			resolver := contracttest.ContentMergeResolver[contracttest.StructMeta]{}
			return &StructRetrievalBackend{
				Docs: []retrieval.Document[contracttest.StructMeta]{
					{ID: "ok", Content: "merge-key", Meta: contracttest.StructMeta{Tenant: "acme"}},
					{ID: "bad", Content: "bad", Score: 1.5},
				},
				FilterSchema: tenantSchema(t),
				Resolver:     resolver,
			}
		},
	)
}

func TestDenseIndexRejectsUnsetSchema(t *testing.T) {
	index := &DenseIndex{}
	err := index.Upsert(
		context.Background(),
		[]dense.Record[contracttest.StructMeta]{{ID: "doc-1", Vector: []float32{1}}},
	)
	if err == nil {
		t.Fatal("Upsert() error = nil, want schema error")
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Upsert() error = %v, want invalid argument", err)
	}
}

func TestDenseIndexCanonicalizesStoredAttributes(t *testing.T) {
	index := &DenseIndex{FilterSchema: tenantAgeSchema(t)}
	err := index.Upsert(context.Background(), []dense.Record[contracttest.StructMeta]{{
		ID:      "doc-1",
		Content: "hello",
		Meta:    contracttest.StructMeta{Tenant: "acme", Age: 7},
		Vector:  []float32{1},
	}})
	if err != nil {
		t.Fatalf("Upsert(): %v", err)
	}

	value := index.Records[0][0].Meta.Age
	if value != 7 {
		t.Fatalf("stored age = %#v, want int64(7)", index.Records[0][0].Meta.Age)
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

func TestRetrievalBackendFilterValidationReturnsNonNilResultSet(t *testing.T) {
	schema := tenantSchema(t)
	foreign := filter.NewSchema()
	other, err := foreign.String("missing")
	if err != nil {
		t.Fatalf("foreign.String(missing): %v", err)
	}
	foreignSchema, err := foreign.Build()
	if err != nil {
		t.Fatalf("foreign.Build(): %v", err)
	}
	filterBuilder, err := filter.NewBuilder(foreignSchema)
	if err != nil {
		t.Fatalf("NewBuilder(): %v", err)
	}
	cond, err := filter.Eq(filterBuilder, other, "acme").Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	backend := &RetrievalBackend{Docs: nil, FilterSchema: schema, VectorRequired: true}
	out, err := backend.Retrieve(context.Background(), retrieval.Query[struct{}]{
		Options: retrieval.RetrieveOptions{
			Vector:  []float32{1},
			Filters: cond,
		},
	})
	if err == nil {
		t.Fatal("Retrieve() error = nil, want error")
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Retrieve() error = %v, want invalid argument", err)
	}
	if out == nil {
		t.Fatal("Retrieve() out = nil, want non-nil empty ResultSet")
	}
}
