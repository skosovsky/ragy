package documents_test

import (
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/chunking"
	"github.com/skosovsky/ragy/documents"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

type indexedMeta struct {
	Title      string `json:"title"`
	Tenant     string `json:"tenant"`
	Visibility string `json:"visibility"`
}

func TestChunkProjectionIndexAndRetainedResolution(t *testing.T) {
	for _, mode := range []string{"new revision", "deleted", "revoked"} {
		t.Run(mode, func(t *testing.T) {
			checkRetainedChunk(t, mode)
		})
	}
}

func checkRetainedChunk(t *testing.T, mode string) {
	t.Helper()
	// Arrange: real chunk projection/BM25/retained hydrator and host revision catalog.
	fixture := newHydrationFixture(t)
	text := "  Alpha beta. Gamma repeated.  "
	fixture.host.payloads[fixture.r1] = retrieval.Document[payloadMeta]{
		ID:      fixture.r1.Artifact,
		Content: text,
		Meta:    payloadMeta{"title": "old"},
	}
	fixture.host.payloads[fixture.r2] = retrieval.Document[payloadMeta]{
		ID:      fixture.r2.Artifact,
		Content: "latest text with unrelated offsets",
		Meta:    payloadMeta{"title": "new"},
	}
	index := retainedChunkIndex(t, fixture, text)
	var err error
	// Act: retrieve a chunk and resolve its supplied exact source location.
	results, err := index.Retrieve(
		t.Context(),
		retrieval.Query[struct{}]{
			Read:    fixture.read,
			Text:    "beta",
			Options: retrieval.RetrieveOptions{TopK: 1},
		},
	)
	if err != nil || results.Len() != 1 {
		t.Fatalf("retrieval: %v", err)
	}
	retrieved := results.Documents()[0]
	exact := retrieved.SourceMapping.Fragments()[0].Location
	resolved, err := fixture.reader.ResolveText(
		t.Context(),
		documents.TextResolveRequest{Read: fixture.read, Locations: []source.Locator{exact}},
	)
	// Assert: newer source content cannot replace the revision and coordinates.
	if err != nil || len(resolved) != 1 || resolved[0].Location.Reference != fixture.r1 ||
		resolved[0].Text.Text() != retrieved.Content ||
		retrieved.Content != text[exact.Span.Start:exact.Span.End] {
		t.Fatalf("resolution: %v %#v", err, resolved)
	}
	if mode == "new revision" {
		return
	}
	before := fixture.host.loadCalls
	if mode == "deleted" {
		delete(fixture.host.descriptors, fixture.r1)
	} else {
		*fixture.revoked = true
	}
	resolved, err = fixture.reader.ResolveText(
		t.Context(),
		documents.TextResolveRequest{Read: fixture.read, Locations: []source.Locator{exact}},
	)
	if !errors.Is(err, ragy.ErrUnavailable) || len(resolved) != 0 || fixture.host.loadCalls != before {
		t.Fatalf("unavailable retained quote: %v %#v", err, resolved)
	}
}

func retainedChunkIndex(t *testing.T, fixture hydrationFixture, text string) *lexical.BM25Index[indexedMeta] {
	t.Helper()
	location := source.Locator{
		Reference: fixture.r1,
		Kind:      source.TextLocation,
		Span:      source.ByteSpan{Start: 0, End: len(text)},
	}
	mapping, err := source.OriginalText(location, text)
	if err != nil {
		t.Fatal(err)
	}
	splitter, err := chunking.NewRecursive[payloadMeta](8, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	chunks, err := splitter.Split(
		t.Context(),
		retrieval.Document[payloadMeta]{
			ID:            fixture.r1.Artifact,
			Content:       text,
			Meta:          payloadMeta{"title": "old"},
			SourceMapping: mapping,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	projected, err := chunking.ProjectDocuments(
		chunks,
		chunking.ProjectionConfig[struct{}, payloadMeta, indexedMeta]{
			Source:    chunking.SourceDescriptor[struct{}]{ID: fixture.r1.Artifact},
			IndexText: chunking.OriginalIndexText[payloadMeta],
			MetadataProjector: chunking.MetadataProjectorFunc[struct{}, payloadMeta, indexedMeta](
				func(_ chunking.SourceDescriptor[struct{}], c chunking.Chunk[payloadMeta], _ chunking.ChunkIdentity) (indexedMeta, error) {
					return indexedMeta{Title: c.Meta["title"], Tenant: "a", Visibility: "public"}, nil
				},
			),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	docs := make([]retrieval.Document[indexedMeta], len(projected))
	for i, p := range projected {
		docs[i] = p.Document
	}
	fields := filter.NewSchema()
	if _, err = fields.String("tenant"); err != nil {
		t.Fatal(err)
	}
	if _, err = fields.String("visibility"); err != nil {
		t.Fatal(err)
	}
	if _, err = fields.String("title"); err != nil {
		t.Fatal(err)
	}
	schema, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	index, err := lexical.NewBM25Index(
		schema,
		lexical.Config[indexedMeta]{SearchFields: []string{"content"}},
		nil,
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err = index.Index(docs); err != nil {
		t.Fatal(err)
	}
	return index
}
