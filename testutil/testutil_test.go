package testutil

import (
	"context"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/documents"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/internal/contracttest"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/tensor"
)

func TestDocumentStoreConformance(t *testing.T) {
	contracttest.RunDocumentsStoreSuite(t, func(t *testing.T, docs []ragy.Document) documents.Store {
		t.Helper()
		return &DocumentStore{Docs: docs, FilterSchema: tenantSchema(t)}
	})
}

func TestSearcherConformance(t *testing.T) {
	contracttest.RunDenseSearcherSuite(t, func(t *testing.T, docs []ragy.Document) dense.Searcher {
		t.Helper()
		return &DenseSearcher{Docs: docs, FilterSchema: tenantSchema(t)}
	})

	contracttest.RunLexicalSearcherSuite(t, func(t *testing.T, docs []ragy.Document) lexical.Searcher {
		t.Helper()
		return &LexicalSearcher{Docs: docs, FilterSchema: tenantSchema(t)}
	})

	contracttest.RunTensorSearcherSuite(t, func(t *testing.T, docs []ragy.Document) tensor.Searcher {
		t.Helper()
		return &TensorSearcher{Docs: docs, FilterSchema: tenantSchema(t)}
	})
}

func TestIndexConformance(t *testing.T) {
	contracttest.RunDenseIndexSuite(t, func(t *testing.T) dense.Index {
		t.Helper()
		return &DenseIndex{FilterSchema: tenantAgeSchema(t)}
	})

	contracttest.RunTensorIndexSuite(t, func(t *testing.T) tensor.Index {
		t.Helper()
		return &TensorIndex{FilterSchema: tenantSchema(t)}
	})
}

func TestGraphStoreConformance(t *testing.T) {
	contracttest.RunGraphStoreSuite(
		t,
		func(t *testing.T, snapshot graph.Snapshot, schema graph.Schema) graph.Store {
			t.Helper()
			return &GraphStore{Snapshot: snapshot, GraphSchema: schema}
		},
	)
}

func TestSearchersRejectUnsetSchema(t *testing.T) {
	if _, err := (&DenseSearcher{Docs: []ragy.Document{{ID: "doc-1"}}}).Search(context.Background(), dense.Request{
		Vector: []float32{1},
	}); err == nil {
		t.Fatal("DenseSearcher.Search() error = nil, want schema error")
	}

	if _, err := (&LexicalSearcher{Docs: []ragy.Document{{ID: "doc-1"}}}).Search(context.Background(), lexical.Request{
		Text: "hello",
	}); err == nil {
		t.Fatal("LexicalSearcher.Search() error = nil, want schema error")
	}

	if _, err := (&TensorSearcher{Docs: []ragy.Document{{ID: "doc-1"}}}).Search(context.Background(), tensor.Request{
		Query: tensor.Tensor{{1}},
	}); err == nil {
		t.Fatal("TensorSearcher.Search() error = nil, want schema error")
	}
}

func TestDocumentStoreRejectsUnsetSchema(t *testing.T) {
	store := &DocumentStore{Docs: []ragy.Document{{ID: "doc-1", Attributes: ragy.Attributes{"tenant": "acme"}}}}

	expr := tenantFilterExpr(t)
	if _, err := store.DeleteByFilter(context.Background(), expr); err == nil {
		t.Fatal("DeleteByFilter() error = nil, want schema error")
	}
}

func TestGraphStoreRejectsUnsetSchema(t *testing.T) {
	store := &GraphStore{Snapshot: graph.Snapshot{
		Nodes: []graph.Node{{ID: "n1", Labels: []string{"Doc"}}},
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
		Docs: []ragy.Document{{
			ID:         "doc-1",
			Content:    "hello",
			Attributes: ragy.Attributes{"tags": []string{"x"}},
		}},
		FilterSchema: tenantSchema(t),
	}

	if _, err := store.FindByIDs(context.Background(), []string{"doc-1"}); err == nil {
		t.Fatal("FindByIDs() error = nil, want invalid document error")
	}
}

func TestDenseIndexRejectsUnsetSchema(t *testing.T) {
	index := &DenseIndex{}
	err := index.Upsert(context.Background(), []dense.Record{{ID: "doc-1", Vector: []float32{1}}})
	if err == nil {
		t.Fatal("Upsert() error = nil, want schema error")
	}
}

func TestDenseIndexCanonicalizesStoredAttributes(t *testing.T) {
	index := &DenseIndex{FilterSchema: tenantAgeSchema(t)}
	err := index.Upsert(context.Background(), []dense.Record{{
		ID:         "doc-1",
		Content:    "hello",
		Attributes: ragy.Attributes{"tenant": "acme", "age": int(7)},
		Vector:     []float32{1},
	}})
	if err != nil {
		t.Fatalf("Upsert(): %v", err)
	}

	value, ok := index.Records[0][0].Attributes["age"].(int64)
	if !ok || value != 7 {
		t.Fatalf("stored age = %#v, want int64(7)", index.Records[0][0].Attributes["age"])
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

func tenantFilterExpr(t *testing.T) filter.IR {
	t.Helper()

	tenant, err := tenantSchema(t).StringField("tenant")
	if err != nil {
		t.Fatalf("schema.StringField(tenant): %v", err)
	}
	expr, err := filter.Normalize(filter.Equal(tenant, "acme"))
	if err != nil {
		t.Fatalf("Normalize(): %v", err)
	}

	return expr
}
