//go:build e2e

package pdf_test

import (
	"context"
	"errors"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	pdfadapter "github.com/skosovsky/ragy/adapters/pdf"
	"github.com/skosovsky/ragy/chunking"
	"github.com/skosovsky/ragy/documents"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/layout"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

type sourceMeta struct {
	Organization string `json:"organization"`
	Visibility   string `json:"visibility"`
	Artifact     string `json:"artifact"`
	Revision     string `json:"revision"`
	Coverage     string `json:"coverage"`
}

type retainedRecord struct {
	meta sourceMeta
	text string
}

type parserHost struct {
	records      map[source.Reference]retainedRecord
	payloadCalls int
}

func (h *parserHost) Describe(
	_ context.Context,
	request source.LookupRequest,
) ([]source.Descriptor[sourceMeta], error) {
	out := make([]source.Descriptor[sourceMeta], 0, len(request.References))
	for _, reference := range request.References {
		if record, exists := h.records[reference]; exists {
			out = append(out, source.Descriptor[sourceMeta]{Reference: reference, Access: record.meta})
		}
	}
	return out, nil
}

func (h *parserHost) Load(
	_ context.Context,
	request source.LookupRequest,
) ([]source.Materialized[retrieval.Document[sourceMeta]], error) {
	h.payloadCalls++
	out := make([]source.Materialized[retrieval.Document[sourceMeta]], 0, len(request.References))
	for _, reference := range request.References {
		if record, exists := h.records[reference]; exists {
			out = append(
				out,
				source.Materialized[retrieval.Document[sourceMeta]]{
					Reference: reference,
					Payload: retrieval.Document[sourceMeta]{
						ID:      reference.Artifact,
						Content: record.text,
						Meta:    record.meta,
					},
				},
			)
		}
	}
	return out, nil
}

func parserScope(t *testing.T) (filter.Schema, access.Binding) {
	t.Helper()
	fields := filter.NewSchema()
	organization, err := fields.String("organization")
	if err != nil {
		t.Fatal(err)
	}
	visibility, err := fields.String("visibility")
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"artifact", "revision", "coverage"} {
		if _, fieldErr := fields.String(name); fieldErr != nil {
			t.Fatal(fieldErr)
		}
	}
	schema, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	builder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	mandatory, err := filter.In(filter.Eq(builder, organization, "a"), visibility, "public").Build()
	if err != nil {
		t.Fatal(err)
	}
	now := time.Unix(100, 0)
	read, err := access.Scoped(access.ScopedConfig{
		Snapshot: access.Snapshot{
			Identity:    "host-policy",
			PolicyEpoch: 7,
			IssuedAt:    now,
			ExpiresAt:   now.Add(time.Minute),
		},
		Schema:      schema,
		Mandatory:   mandatory,
		Publication: access.CurrentPublication(),
		Authority: access.AuthorityFunc(
			func(context.Context, access.Snapshot) error { return nil },
		),
		Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	return schema, read
}

func TestE2EParserProjectionIndexRetrieveAndScopedResolve(t *testing.T) {
	// Arrange: parse actual PDF bytes, then project host-owned scalar metadata.
	ctx := context.Background()
	parser, err := pdfadapter.New(parserConfig(integrationPython(t)))
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := parser.Parse(ctx, layout.Input{Reference: fixtureReference(), Data: loadFixture(t)})
	if err != nil {
		t.Fatal(err)
	}
	schema, read := parserScope(t)
	codec := retrieval.NewJSONCodec[sourceMeta](schema)
	host := &parserHost{records: make(map[source.Reference]retainedRecord)}
	var indexed []retrieval.Document[sourceMeta]
	for _, page := range parsed.Pages {
		if page.Text == "" {
			continue
		}
		meta := sourceMeta{
			Organization: "a",
			Visibility:   "public",
			Artifact:     page.Reference.Artifact,
			Revision:     page.Reference.Revision,
			Coverage:     string(parsed.Coverage),
		}
		host.records[page.Reference] = retainedRecord{meta: meta, text: page.Text}
		indexed = append(indexed, projectParsedPage(ctx, t, page, meta)...)
	}
	index, err := lexical.NewBM25Index[sourceMeta](
		schema,
		lexical.Config[sourceMeta]{SearchFields: []string{"content"}},
		nil,
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if indexErr := index.Index(indexed); indexErr != nil {
		t.Fatal(indexErr)
	}
	reader, err := documents.NewHydrator(
		documents.HydrationConfig[sourceMeta, sourceMeta]{
			Target:     "lexical",
			Schema:     schema,
			Catalog:    host,
			Loader:     host,
			Attributes: codec.Encode,
			CloneMeta:  func(meta sourceMeta) (sourceMeta, error) { return meta, nil },
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	// Act: actual scoped retrieval returns the retained representation used to resolve.
	results, err := index.Retrieve(
		ctx,
		retrieval.Query[struct{}]{Read: read, Text: "beta", Options: retrieval.RetrieveOptions{TopK: 1}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if results.Len() != 1 {
		t.Fatalf("actual index retrieval returned %d documents", results.Len())
	}
	doc := results.Documents()[0]
	page := parsed.Pages[0]
	location := source.Locator{
		Reference: page.Reference,
		Kind:      source.TextLocation,
		Span:      source.ByteSpan{Start: 6, End: 10},
	}
	resolved, err := reader.ResolveText(
		ctx,
		documents.TextResolveRequest{Read: read, Locations: []source.Locator{location}},
	)
	if err != nil {
		t.Fatal(err)
	}
	artifact, err := (retrieval.DefaultArtifactRenderer[sourceMeta]{}).Render(
		ctx,
		read,
		results,
		retrieval.ArtifactRenderOptions[sourceMeta]{
			Resource:  retrieval.RuneResource(100000),
			CloneMeta: func(meta sourceMeta) (sourceMeta, error) { return meta, nil },
			Mapping:   func(retrieval.Document[sourceMeta]) (source.MappedText, error) { return resolved[0].Text, nil },
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	// Assert: original source bytes/geometry identity and partial coverage survive handoff.
	if doc.ID != location.Reference.Artifact+"_0" || doc.Meta.Artifact != location.Reference.Artifact ||
		doc.Meta.Revision != "r1" ||
		doc.Meta.Coverage != "partial" {
		t.Fatal("projection/retrieval lost source revision or promoted coverage")
	}
	if len(artifact.Snippets) != 1 || artifact.Snippets[0].Content != "beta" ||
		artifact.Snippets[0].Mapping.Fragments()[0].Location != location {
		t.Fatal("artifact lost original citation")
	}
	checkRetainedResolution(t, reader, host, read, location)
}

func checkRetainedResolution(
	t *testing.T,
	reader *documents.Hydrator[sourceMeta, sourceMeta],
	host *parserHost,
	read access.Binding,
	location source.Locator,
) {
	t.Helper()
	// Arrange: r2 is present; resolving r1 must not choose the latest record.
	latest := location.Reference
	latest.Revision = "r2"
	original := host.records[location.Reference]
	newMeta := original.meta
	newMeta.Revision = "r2"
	host.records[latest] = retainedRecord{meta: newMeta, text: "latest replacement text"}
	// Act.
	resolved, err := reader.ResolveText(
		context.Background(),
		documents.TextResolveRequest{Read: read, Locations: []source.Locator{location}},
	)
	// Assert.
	if err != nil || len(resolved) != 1 || resolved[0].Text.Text() != "beta" {
		t.Fatal("r2 substituted for retained r1")
	}
	// Arrange/Act: deleting r1 leaves r2 intact, without a fallback materialization.
	delete(host.records, location.Reference)
	before := host.payloadCalls
	missing, err := reader.ResolveText(
		context.Background(),
		documents.TextResolveRequest{Read: read, Locations: []source.Locator{location}},
	)
	if !errors.Is(err, ragy.ErrUnavailable) || len(missing) != 0 || host.payloadCalls != before {
		t.Fatal("missing r1 loaded latest or leaked a citation")
	}
	// Arrange/Act: denied permission metadata also prevents payload loading.
	original.meta.Organization = "foreign"
	host.records[location.Reference] = original
	denied, err := reader.ResolveText(
		context.Background(),
		documents.TextResolveRequest{Read: read, Locations: []source.Locator{location}},
	)
	if !errors.Is(err, ragy.ErrUnavailable) || len(denied) != 0 || host.payloadCalls != before {
		t.Fatal("denied source materialized")
	}
}

func projectParsedPage(
	ctx context.Context,
	t *testing.T,
	page layout.Page,
	meta sourceMeta,
) []retrieval.Document[sourceMeta] {
	t.Helper()
	// Normalized page text is retained under its explicit representation. Splitters
	// slice this supplied mapping rather than infer offsets from finished text.
	splitter, err := chunking.NewRecursive[sourceMeta](256, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	location := source.Locator{
		Reference: page.Reference,
		Kind:      source.TextLocation,
		Span:      source.ByteSpan{Start: 0, End: len(page.Text)},
	}
	mapping, mapErr := source.OriginalText(location, page.Text)
	if mapErr != nil {
		t.Fatal(mapErr)
	}
	chunks, err := splitter.Split(
		ctx,
		retrieval.Document[sourceMeta]{
			ID:             page.Reference.Artifact,
			Content:        page.Text,
			Meta:           meta,
			SourceMapping:  mapping,
			SourceSupports: mapping.Supports(),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 1 || chunks[0].Content != page.Text {
		t.Fatal("fixture page no longer fits the declared exact whole-page chunk profile")
	}
	documents, err := chunking.ProjectDocuments(chunks, chunking.ProjectionConfig[sourceMeta, sourceMeta, sourceMeta]{
		Source:    chunking.SourceDescriptor[sourceMeta]{ID: page.Reference.Artifact, Meta: meta},
		IndexText: chunking.OriginalIndexText[sourceMeta],
		MetadataProjector: chunking.MetadataProjectorFunc[sourceMeta, sourceMeta, sourceMeta](
			func(input chunking.SourceDescriptor[sourceMeta], _ chunking.Chunk[sourceMeta], _ chunking.ChunkIdentity) (sourceMeta, error) {
				return input.Meta, nil
			},
		),
	})
	if err != nil {
		t.Fatal(err)
	}
	result := make([]retrieval.Document[sourceMeta], len(documents))
	for i, projected := range documents {
		result[i] = projected.Document
	}
	return result
}
