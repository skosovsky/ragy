package testutil

import (
	"context"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/documents"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/internal/contracttest"
	"github.com/skosovsky/ragy/retrieval"
)

// StructRetrievalBackend is a fake retrieval.Backend for struct metadata.
type StructRetrievalBackend struct {
	Docs           []retrieval.Document[contracttest.StructMeta]
	Err            error
	Requests       []retrieval.RetrieveOptions
	FilterSchema   filter.Schema
	VectorRequired bool
	Resolver       retrieval.IdentityResolver[contracttest.StructMeta]
}

// Retrieve implements retrieval.Backend for struct-based contract tests.
func (b *StructRetrievalBackend) Retrieve(
	_ context.Context,
	query string,
	opts retrieval.RetrieveOptions,
) (retrieval.ResultSet[contracttest.StructMeta], error) {
	b.Requests = append(b.Requests, opts)
	return runFakeRetrieve(
		b.Err,
		query,
		opts,
		b.VectorRequired,
		b.Schema(),
		b.Docs,
		retrieval.DefaultResolver(b.Resolver),
	)
}

// Schema returns the configured filter schema used by the fake backend.
func (b *StructRetrievalBackend) Schema() filter.Schema {
	return b.FilterSchema
}

// StructDocumentStore is a memory-backed documents.Store fake for struct metadata.
type StructDocumentStore struct {
	Docs         []retrieval.Document[contracttest.StructMeta]
	Err          error
	FilterSchema filter.Schema
}

// FindByIDs implements documents.Store.
func (s *StructDocumentStore) FindByIDs(
	_ context.Context,
	ids []string,
) ([]retrieval.Document[contracttest.StructMeta], error) {
	if s.Err != nil {
		return nil, s.Err
	}
	if len(ids) == 0 {
		return nil, nil
	}

	byID := make(map[string]retrieval.Document[contracttest.StructMeta], len(s.Docs))
	for _, doc := range s.Docs {
		byID[doc.ID] = cloneStructDocument(doc)
	}

	out := make([]retrieval.Document[contracttest.StructMeta], 0, len(ids))
	for _, id := range ids {
		doc, ok := byID[id]
		if !ok {
			continue
		}
		out = append(out, doc)
	}
	if len(out) == 0 {
		return nil, nil
	}

	return validateStructDocuments(out)
}

// DeleteByIDs implements documents.Store.
func (s *StructDocumentStore) DeleteByIDs(_ context.Context, ids []string) (documents.DeleteResult, error) {
	if s.Err != nil {
		return documents.DeleteResult{}, s.Err
	}
	if len(ids) == 0 {
		return documents.DeleteResult{}, nil
	}

	remove := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		remove[id] = struct{}{}
	}

	deleted := 0
	kept := make([]retrieval.Document[contracttest.StructMeta], 0, len(s.Docs))
	for _, doc := range s.Docs {
		if _, ok := remove[doc.ID]; ok {
			deleted++
			continue
		}
		kept = append(kept, cloneStructDocument(doc))
	}

	s.Docs = kept
	return documents.DeleteResult{Deleted: deleted}, nil
}

// DeleteByFilter implements documents.Store.
func (s *StructDocumentStore) DeleteByFilter(_ context.Context, cond filter.Condition) (documents.DeleteResult, error) {
	schema := s.Schema()
	codec := retrieval.NewJSONCodec[contracttest.StructMeta](schema)
	return deleteByFilter(
		s.Docs,
		cond,
		schema,
		func(doc retrieval.Document[contracttest.StructMeta], condition filter.Condition) (bool, error) {
			return retrieval.MatchDocument(codec, doc, condition)
		},
		cloneStructDocument,
		s.Err,
		func(docs []retrieval.Document[contracttest.StructMeta]) {
			s.Docs = docs
		},
	)
}

// Schema returns the configured filter schema used by the fake store.
func (s *StructDocumentStore) Schema() filter.Schema {
	return s.FilterSchema
}

func validateStructDocuments(
	in []retrieval.Document[contracttest.StructMeta],
) ([]retrieval.Document[contracttest.StructMeta], error) {
	if len(in) == 0 {
		return nil, nil
	}

	out := make([]retrieval.Document[contracttest.StructMeta], 0, len(in))
	for _, doc := range in {
		if err := retrieval.ValidateDocument(doc); err != nil {
			return out, ragy.WrapProjectionError(err, "testutil validate struct documents")
		}
		out = append(out, cloneStructDocument(doc))
	}

	return out, nil
}

func cloneStructDocument(in retrieval.Document[contracttest.StructMeta]) retrieval.Document[contracttest.StructMeta] {
	return retrieval.Document[contracttest.StructMeta]{
		ID:      in.ID,
		Content: in.Content,
		Score:   in.Score,
		Meta:    in.Meta,
	}
}

var (
	_ retrieval.Backend[contracttest.StructMeta] = (*StructRetrievalBackend)(nil)
	_ documents.Store[contracttest.StructMeta]   = (*StructDocumentStore)(nil)
)
