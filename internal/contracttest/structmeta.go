package contracttest

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/documents"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/retrieval"
)

// TenantOnlyMeta is a minimal metadata fixture for decode and projection error tests.
type TenantOnlyMeta struct {
	Tenant string `json:"tenant"`
}

// StructMeta is a typed metadata fixture for struct-based contract tests.
type StructMeta struct {
	Tenant string  `json:"tenant,omitempty"`
	Age    int64   `json:"age,omitempty"`
	Score  float64 `json:"score,omitempty"`
	Kind   string  `json:"kind,omitempty"`
}

// ContentMergeResolver groups documents by content for merge-key contract tests.
type ContentMergeResolver[TMeta any] struct{}

func (ContentMergeResolver[TMeta]) Resolve(doc retrieval.Document[TMeta]) retrieval.Identity {
	return retrieval.Identity{DocumentID: doc.ID, MergeKey: doc.Content}
}

const wantedDocID = "doc-1"
const tenantAcme = "acme"
const partialProjectionTopK = 10

type DenseStructBackendFactory func(t *testing.T, docs []retrieval.Document[StructMeta]) retrieval.Backend[StructMeta]
type LexicalStructBackendFactory func(t *testing.T, docs []retrieval.Document[StructMeta]) retrieval.Backend[StructMeta]
type DocumentsStructStoreFactory func(t *testing.T, docs []retrieval.Document[StructMeta]) documents.Store[StructMeta]
type DocumentsStructStorePartialFactory func(t *testing.T) documents.Store[StructMeta]

func undeclaredStructFilter(t *testing.T) filter.Condition {
	t.Helper()

	builder := filter.NewSchema()
	missing, err := builder.String("missing")
	if err != nil {
		t.Fatalf("String(missing): %v", err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}
	filterBuilder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatalf("NewBuilder(): %v", err)
	}
	cond, err := filter.Eq(filterBuilder, missing, "value").Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}
	return cond
}

func requireUndeclaredFilterRetrieveRejects(
	t *testing.T,
	retrieve func(filter.Condition) (retrieval.ResultSet[StructMeta], error),
) {
	t.Helper()

	out, err := retrieve(undeclaredStructFilter(t))
	RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Retrieve() error = %v, want invalid argument", err)
	}
}

const partialMergeProbeScore = 0.1

func assertPartialProjectionResolverParity(t *testing.T, out retrieval.ResultSet[StructMeta]) {
	t.Helper()

	resolver := ContentMergeResolver[StructMeta]{}
	merged, mergeErr := retrieval.NewResultSet(
		[]retrieval.Document[StructMeta]{{
			ID:      "other",
			Content: out.Documents()[0].Content,
			Score:   partialMergeProbeScore,
		}},
		resolver,
	).Merge(out)
	if mergeErr != nil {
		t.Fatalf("Merge(): %v", mergeErr)
	}
	if merged.Len() != 1 {
		t.Fatalf("merged Len() = %d, want 1 doc under content merge key", merged.Len())
	}
}

func tenantStructCondition(t *testing.T, schema filter.Schema, value string) filter.Condition {
	t.Helper()

	tenant, err := schema.StringField("tenant")
	if err != nil {
		t.Fatalf("Schema().StringField(tenant): %v", err)
	}

	builder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatalf("NewBuilder(): %v", err)
	}

	cond, err := filter.Eq(builder, tenant, value).Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	return cond
}

type schemaProvider interface {
	Schema() filter.Schema
}

func schemaFromTypedBackend[TMeta any](t *testing.T, backend retrieval.Backend[TMeta]) filter.Schema {
	t.Helper()

	provider, ok := backend.(schemaProvider)
	if !ok {
		t.Fatal("backend does not expose Schema()")
	}
	return provider.Schema()
}

func requireOneDoc(t *testing.T, out retrieval.ResultSet[StructMeta], wantID string) {
	t.Helper()
	docs := out.Documents()
	if len(docs) != 1 || docs[0].ID != wantID {
		t.Fatalf("Documents() = %#v, want id %q", docs, wantID)
	}
}

func requireEmptyResultSet(t *testing.T, out retrieval.ResultSet[StructMeta]) {
	t.Helper()
	if out == nil {
		t.Fatal("Retrieve() = nil ResultSet, want non-nil empty set")
	}
	if !out.IsEmpty() {
		t.Fatalf("Retrieve() = %#v, want empty", out.Documents())
	}
}

// RequireErrorResultSet asserts error paths return a non-nil empty ResultSet.
func RequireErrorResultSet[TMeta any](t *testing.T, out retrieval.ResultSet[TMeta], err error) {
	t.Helper()
	if err == nil {
		t.Fatal("err = nil, want error")
	}
	if out == nil {
		t.Fatal("ResultSet = nil, want non-nil empty set")
	}
	if !out.IsEmpty() {
		t.Fatalf("ResultSet = %#v, want empty on error", out.Documents())
	}
}

// RunDenseStructBackendSuite checks typed struct metadata for dense backends.
func RunDenseStructBackendSuite(t *testing.T, factory DenseStructBackendFactory) {
	t.Helper()

	t.Run("valid docs pass through", func(t *testing.T) {
		backend := factory(t, []retrieval.Document[StructMeta]{{ID: wantedDocID, Content: "hello"}})
		out, err := backend.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
			Vector: []float32{1},
			TopK:   retrieveOptionsInvalidTopK,
		})
		if err != nil {
			t.Fatalf("Retrieve(): %v", err)
		}
		requireOneDoc(t, out, wantedDocID)
	})

	t.Run("declared filter built from builder passes", func(t *testing.T) {
		backend := factory(t, []retrieval.Document[StructMeta]{{
			ID:      wantedDocID,
			Content: "hello",
			Meta:    StructMeta{Tenant: tenantAcme},
		}})
		schema := schemaFromTypedBackend(t, backend)
		out, err := backend.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
			Vector:  []float32{1},
			TopK:    retrieveOptionsInvalidTopK,
			Filters: tenantStructCondition(t, schema, tenantAcme),
		})
		if err != nil {
			t.Fatalf("Retrieve(): %v", err)
		}
		docs := out.Documents()
		if len(docs) != 1 || docs[0].Meta.Tenant != tenantAcme {
			t.Fatalf("Documents() = %#v, want tenant acme", docs)
		}
	})

	t.Run("invalid docs reject", func(t *testing.T) {
		backend := factory(t, []retrieval.Document[StructMeta]{{Content: "broken"}})
		out, err := backend.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
			Vector: []float32{1},
			TopK:   retrieveOptionsInvalidTopK,
		})
		RequireErrorResultSet(t, out, err)
		if !errors.Is(err, ragy.ErrProtocol) {
			t.Fatalf("Retrieve() error = %v, want protocol", err)
		}
	})

	t.Run("no results returns empty set", func(t *testing.T) {
		backend := factory(t, nil)
		out, err := backend.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
			Vector: []float32{1},
			TopK:   retrieveOptionsInvalidTopK,
		})
		if err != nil {
			t.Fatalf("Retrieve(): %v", err)
		}
		requireEmptyResultSet(t, out)
	})

	t.Run("undeclared filter rejects", func(t *testing.T) {
		backend := factory(t, []retrieval.Document[StructMeta]{{ID: wantedDocID, Content: "hello"}})
		requireUndeclaredFilterRetrieveRejects(t, func(cond filter.Condition) (retrieval.ResultSet[StructMeta], error) {
			return backend.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
				Vector:  []float32{1},
				TopK:    retrieveOptionsInvalidTopK,
				Filters: cond,
			})
		})
	})
}

// RunLexicalStructBackendSuite checks typed struct metadata for lexical backends.
func RunLexicalStructBackendSuite(t *testing.T, factory LexicalStructBackendFactory) {
	t.Helper()

	t.Run("valid docs pass through", func(t *testing.T) {
		backend := factory(t, []retrieval.Document[StructMeta]{{ID: wantedDocID, Content: "hello"}})
		out, err := backend.Retrieve(context.Background(), "hello", DefaultRetrieveOptions())
		if err != nil {
			t.Fatalf("Retrieve(): %v", err)
		}
		requireOneDoc(t, out, wantedDocID)
	})

	t.Run("declared filter built from builder passes", func(t *testing.T) {
		backend := factory(t, []retrieval.Document[StructMeta]{{
			ID:      wantedDocID,
			Content: "hello",
			Meta:    StructMeta{Tenant: tenantAcme},
		}})
		schema := schemaFromTypedBackend(t, backend)
		out, err := backend.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{
			TopK:    retrieveOptionsInvalidTopK,
			Filters: tenantStructCondition(t, schema, tenantAcme),
		})
		if err != nil {
			t.Fatalf("Retrieve(): %v", err)
		}
		docs := out.Documents()
		if len(docs) != 1 || docs[0].Meta.Tenant != tenantAcme {
			t.Fatalf("Documents() = %#v, want tenant acme", docs)
		}
	})

	t.Run("invalid docs reject", func(t *testing.T) {
		backend := factory(t, []retrieval.Document[StructMeta]{{Content: "broken"}})
		out, err := backend.Retrieve(context.Background(), "hello", DefaultRetrieveOptions())
		RequireErrorResultSet(t, out, err)
		if !errors.Is(err, ragy.ErrProtocol) {
			t.Fatalf("Retrieve() error = %v, want protocol", err)
		}
	})

	t.Run("no results returns empty set", func(t *testing.T) {
		backend := factory(t, nil)
		out, err := backend.Retrieve(context.Background(), "hello", DefaultRetrieveOptions())
		if err != nil {
			t.Fatalf("Retrieve(): %v", err)
		}
		requireEmptyResultSet(t, out)
	})

	t.Run("undeclared filter rejects", func(t *testing.T) {
		backend := factory(t, []retrieval.Document[StructMeta]{{ID: wantedDocID, Content: "hello"}})
		requireUndeclaredFilterRetrieveRejects(t, func(cond filter.Condition) (retrieval.ResultSet[StructMeta], error) {
			return backend.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{
				TopK:    retrieveOptionsInvalidTopK,
				Filters: cond,
			})
		})
	})
}

// RunDocumentsStructStoreSuite checks typed struct metadata for document stores.
func RunDocumentsStructStoreSuite(t *testing.T, factory DocumentsStructStoreFactory) {
	t.Helper()

	t.Run("find missing returns nil", func(t *testing.T) {
		testDocumentsFindMissingReturnsNil(t, factory)
	})

	t.Run("delete empty filter rejects", func(t *testing.T) {
		testDocumentsDeleteEmptyFilterRejects(t, factory)
	})

	t.Run("find returns typed meta", func(t *testing.T) {
		store := factory(t, []retrieval.Document[StructMeta]{{
			ID:      "doc-1",
			Content: "hello",
			Meta:    StructMeta{Tenant: tenantAcme},
		}})

		docs, err := store.FindByIDs(context.Background(), []string{"doc-1"})
		if err != nil {
			t.Fatalf("FindByIDs(): %v", err)
		}
		if len(docs) != 1 || docs[0].Meta.Tenant != tenantAcme {
			t.Fatalf("FindByIDs() = %#v, want tenant acme", docs)
		}
	})

	t.Run("delete by filter mutates state", func(t *testing.T) {
		store := factory(t, []retrieval.Document[StructMeta]{
			{ID: "doc-1", Content: "hello", Meta: StructMeta{Tenant: tenantAcme}},
			{ID: "doc-2", Content: "world", Meta: StructMeta{Tenant: "globex"}},
		})

		tenant, err := store.Schema().StringField("tenant")
		if err != nil {
			t.Fatalf("Schema().StringField(tenant): %v", err)
		}
		builder, err := filter.NewBuilder(store.Schema())
		if err != nil {
			t.Fatalf("NewBuilder(): %v", err)
		}
		cond, err := filter.Eq(builder, tenant, tenantAcme).Build()
		if err != nil {
			t.Fatalf("Build(): %v", err)
		}

		result, err := store.DeleteByFilter(context.Background(), cond)
		if err != nil {
			t.Fatalf("DeleteByFilter(): %v", err)
		}
		if result.Deleted != 1 {
			t.Fatalf("DeleteResult.Deleted = %d, want 1", result.Deleted)
		}
	})

	t.Run("undeclared filter rejects", func(t *testing.T) {
		store := factory(t, []retrieval.Document[StructMeta]{{ID: "doc-1", Content: "hello"}})
		_, err := store.DeleteByFilter(context.Background(), undeclaredStructFilter(t))
		if err == nil {
			t.Fatal("DeleteByFilter() error = nil, want error")
		}
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("DeleteByFilter() error = %v, want invalid argument", err)
		}
	})
}

// RunDocumentsPartialFindByIDsSuite checks partial slice semantics on projection failure.
func RunDocumentsPartialFindByIDsSuite(t *testing.T, factory DocumentsStructStorePartialFactory) {
	t.Helper()

	t.Run("find preserves partial on projection error", func(t *testing.T) {
		store := factory(t)
		out, err := store.FindByIDs(context.Background(), []string{"ok", "bad"})
		if err == nil {
			t.Fatal("FindByIDs() error = nil, want error")
		}
		if !errors.Is(err, ragy.ErrProtocol) {
			t.Fatalf("FindByIDs() error = %v, want protocol", err)
		}
		if len(out) != 1 || out[0].ID != "ok" {
			t.Fatalf("FindByIDs() = %#v, want partial ok doc", out)
		}
	})
}

// RetrievePartialProjectionFactory builds a backend that fails projection on the second hit.
type RetrievePartialProjectionFactory func(t *testing.T) retrieval.Backend[StructMeta]

// RetrievePartialProjectionResolverFactory builds a backend with ContentMergeResolver for merge-key parity.
type RetrievePartialProjectionResolverFactory func(t *testing.T) retrieval.Backend[StructMeta]

// RunRetrievePartialProjectionSuite checks partial ResultSet semantics on projection failure.
func RunRetrievePartialProjectionSuite(
	t *testing.T,
	factory RetrievePartialProjectionFactory,
	resolverFactories ...RetrievePartialProjectionResolverFactory,
) {
	t.Helper()

	t.Run("retrieve preserves partial on projection error", func(t *testing.T) {
		backend := factory(t)
		out, err := backend.Retrieve(context.Background(), "q", retrieval.RetrieveOptions{
			TopK:   partialProjectionTopK,
			Vector: []float32{1},
		})
		if err == nil {
			t.Fatal("Retrieve() error = nil, want error")
		}
		if !errors.Is(err, ragy.ErrProtocol) {
			t.Fatalf("Retrieve() error = %v, want protocol", err)
		}
		if out.Len() != 1 || out.Documents()[0].ID != "ok" {
			t.Fatalf("Documents() = %#v, want partial ok doc", out.Documents())
		}
	})

	if len(resolverFactories) == 0 {
		return
	}

	t.Run("retrieve partial preserves custom resolver merge key", func(t *testing.T) {
		backend := resolverFactories[0](t)
		out, err := backend.Retrieve(context.Background(), "q", retrieval.RetrieveOptions{
			TopK:   partialProjectionTopK,
			Vector: []float32{1},
		})
		if err == nil {
			t.Fatal("Retrieve() error = nil, want error")
		}
		if !errors.Is(err, ragy.ErrProtocol) {
			t.Fatalf("Retrieve() error = %v, want protocol", err)
		}
		if out.Len() != 1 || out.Documents()[0].ID != "ok" {
			t.Fatalf("Documents() = %#v, want partial ok doc", out.Documents())
		}
		assertPartialProjectionResolverParity(t, out)
	})
}

func testDocumentsFindMissingReturnsNil(t *testing.T, factory DocumentsStructStoreFactory) {
	t.Helper()

	store := factory(t, []retrieval.Document[StructMeta]{{
		ID:      "doc-1",
		Content: "hello",
	}})
	docs, err := store.FindByIDs(context.Background(), []string{"missing"})
	if err != nil {
		t.Fatalf("FindByIDs(): %v", err)
	}
	if docs != nil {
		t.Fatalf("FindByIDs() = %#v, want nil", docs)
	}
}

func testDocumentsDeleteEmptyFilterRejects(t *testing.T, factory DocumentsStructStoreFactory) {
	t.Helper()

	store := factory(t, []retrieval.Document[StructMeta]{{ID: "doc-1", Content: "hello"}})
	if _, err := store.DeleteByFilter(context.Background(), filter.Condition{}); err == nil {
		t.Fatal("DeleteByFilter(empty) error = nil, want error")
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("DeleteByFilter(empty) error = %v, want invalid argument", err)
	}
}

type RetrieveOptionsBackendFactory func(t *testing.T) retrieval.Backend[StructMeta]

const retrieveOptionsInvalidTopK = 3

const retrieveOptionsInvalidMinSimilarity = 1.1

// DefaultRetrieveOptions returns valid TopK for contract Retrieve calls.
func DefaultRetrieveOptions() retrieval.RetrieveOptions {
	return retrieval.RetrieveOptions{TopK: retrieveOptionsInvalidTopK}
}

// RetrieveOptionsInvalidConfig customizes RunRetrieveOptionsInvalidSuite for lexical backends.
type RetrieveOptionsInvalidConfig struct {
	Query  string
	Vector []float32
}

// RunRetrieveOptionsInvalidSuite checks shared retrieve-option validation across backends.
func RunRetrieveOptionsInvalidSuite(
	t *testing.T,
	factory RetrieveOptionsBackendFactory,
	cfg ...RetrieveOptionsInvalidConfig,
) {
	t.Helper()

	query := ""
	vector := []float32{1}
	if len(cfg) > 0 {
		if cfg[0].Query != "" {
			query = cfg[0].Query
		}
		if cfg[0].Vector == nil && cfg[0].Query != "" {
			vector = nil
		}
	}

	t.Run("fetch limit less than top k rejects", func(t *testing.T) {
		backend := factory(t)
		assertRetrieveOptionsReject(t, backend, query, retrieval.RetrieveOptions{
			Vector:     vector,
			FetchLimit: 1,
			TopK:       retrieveOptionsInvalidTopK,
		})
	})

	t.Run("negative top k rejects", func(t *testing.T) {
		backend := factory(t)
		assertRetrieveOptionsReject(t, backend, query, retrieval.RetrieveOptions{
			Vector: vector,
			TopK:   -1,
		})
	})

	t.Run("negative fetch limit rejects", func(t *testing.T) {
		backend := factory(t)
		assertRetrieveOptionsReject(t, backend, query, retrieval.RetrieveOptions{
			Vector:     vector,
			FetchLimit: -1,
			TopK:       retrieveOptionsInvalidTopK,
		})
	})

	t.Run("min similarity out of range rejects", func(t *testing.T) {
		backend := factory(t)
		assertRetrieveOptionsReject(t, backend, query, retrieval.RetrieveOptions{
			Vector:        vector,
			TopK:          retrieveOptionsInvalidTopK,
			MinSimilarity: retrieveOptionsInvalidMinSimilarity,
		})
	})

	t.Run("zero top k and zero fetch limit rejects", func(t *testing.T) {
		backend := factory(t)
		assertRetrieveOptionsReject(t, backend, query, retrieval.RetrieveOptions{Vector: vector})
	})
}

func assertRetrieveOptionsReject(
	t *testing.T,
	backend retrieval.Backend[StructMeta],
	query string,
	opts retrieval.RetrieveOptions,
) {
	t.Helper()

	out, err := backend.Retrieve(context.Background(), query, opts)
	RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Retrieve() error = %v, want invalid argument", err)
	}
}

// GraphRetrieveOptionsBackendFactory builds a graph backend requiring Graph options.
type GraphRetrieveOptionsBackendFactory func(t *testing.T) retrieval.Backend[StructMeta]

func graphOptionsInvalidFixture() *retrieval.GraphOptions {
	return &retrieval.GraphOptions{
		Seeds:     []string{"n1"},
		Direction: graph.DirectionOutbound,
		Depth:     1,
	}
}

// RunGraphRetrieveOptionsInvalidSuite checks retrieve-option validation for graph backends.
func RunGraphRetrieveOptionsInvalidSuite(t *testing.T, factory GraphRetrieveOptionsBackendFactory) {
	t.Helper()

	t.Run("fetch limit less than top k rejects with graph options", func(t *testing.T) {
		t.Parallel()

		backend := factory(t)
		out, err := backend.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
			FetchLimit: 1,
			TopK:       retrieveOptionsInvalidTopK,
			Graph:      graphOptionsInvalidFixture(),
		})
		RequireErrorResultSet(t, out, err)
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Retrieve() error = %v, want invalid argument", err)
		}
	})

	t.Run("negative top k rejects with graph options", func(t *testing.T) {
		t.Parallel()

		backend := factory(t)
		out, err := backend.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
			TopK:  -1,
			Graph: graphOptionsInvalidFixture(),
		})
		RequireErrorResultSet(t, out, err)
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Retrieve() error = %v, want invalid argument", err)
		}
	})

	t.Run("negative fetch limit rejects with graph options", func(t *testing.T) {
		t.Parallel()

		backend := factory(t)
		out, err := backend.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
			FetchLimit: -1,
			TopK:       retrieveOptionsInvalidTopK,
			Graph:      graphOptionsInvalidFixture(),
		})
		RequireErrorResultSet(t, out, err)
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Retrieve() error = %v, want invalid argument", err)
		}
	})

	t.Run("min similarity out of range rejects with graph options", func(t *testing.T) {
		t.Parallel()

		backend := factory(t)
		out, err := backend.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
			TopK:          retrieveOptionsInvalidTopK,
			MinSimilarity: retrieveOptionsInvalidMinSimilarity,
			Graph:         graphOptionsInvalidFixture(),
		})
		RequireErrorResultSet(t, out, err)
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Retrieve() error = %v, want invalid argument", err)
		}
	})

	t.Run("missing graph options rejects", func(t *testing.T) {
		t.Parallel()

		backend := factory(t)
		out, err := backend.Retrieve(context.Background(), "", retrieval.RetrieveOptions{TopK: 1})
		RequireErrorResultSet(t, out, err)
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Retrieve() error = %v, want invalid argument", err)
		}
	})
}
