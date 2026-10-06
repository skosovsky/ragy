package pdf_test

import (
	"bytes"
	"context"
	"os"
	"testing"

	pdfadapter "github.com/skosovsky/ragy/adapters/pdf"
	"github.com/skosovsky/ragy/layout"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

type layoutPayloadHost struct {
	*parserHost

	payloads map[source.Reference]layout.Retained
}

func (h *layoutPayloadHost) Load(
	_ context.Context,
	request source.LookupRequest,
) ([]source.Materialized[layout.Retained], error) {
	h.payloadCalls++
	var out []source.Materialized[layout.Retained]
	for _, reference := range request.References {
		if payload, exists := h.payloads[reference]; exists {
			out = append(out, source.Materialized[layout.Retained]{Reference: reference, Payload: payload})
		}
	}
	return out, nil
}

func TestActualParserRetainedPageCellImageDocumentResolution(t *testing.T) {
	// Arrange: source host retains original fixture bytes and actual parsed representations.
	ctx := context.Background()
	pdfBytes := loadFixture(t)
	parser, err := pdfadapter.New(parserConfig(integrationPython(t)))
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := parser.Parse(ctx, layout.Input{Reference: fixtureReference(), Data: pdfBytes})
	if err != nil {
		t.Fatal(err)
	}
	pixels, err := os.ReadFile("testdata/diagram.png")
	if err != nil {
		t.Fatal(err)
	}
	page := parsed.Pages[0]
	pageLocation := source.Locator{Reference: page.Reference, Kind: source.PageLocation, Page: page.Geometry}
	cell, image := page.Cells[0], page.Images[0]
	documentLocation := source.Locator{Reference: parsed.Reference, Kind: source.DocumentLocation}
	derived, err := source.DerivedText("fixture diagram", []source.Locator{image.Location})
	if err != nil {
		t.Fatal(err)
	}
	host := &layoutPayloadHost{
		parserHost: &parserHost{records: make(map[source.Reference]retainedRecord)},
		payloads: map[source.Reference]layout.Retained{
			page.Reference: {
				Original:    pageLocation,
				Text:        page.Text,
				Words:       page.Words,
				Coverage:    page.Coverage,
				Diagnostics: page.Diagnostics,
			},
			cell.Location.Reference: {
				Original:   cell.Location,
				Text:       cell.Text,
				CellRegion: cell.Region,
				Coverage:   layout.Complete,
			},
			image.Location.Reference: {
				Original:    image.Location,
				Bytes:       pixels,
				MediaType:   "image/png",
				Coverage:    layout.Partial,
				Diagnostics: page.Diagnostics,
				Derived:     derived,
			},
			parsed.Reference: {
				Original:    documentLocation,
				Bytes:       pdfBytes,
				MediaType:   "application/pdf",
				Coverage:    parsed.Coverage,
				Diagnostics: page.Diagnostics,
			},
		},
	}
	for reference, retained := range host.payloads {
		host.records[reference] = retainedRecord{
			meta: sourceMeta{
				Organization: "a",
				Visibility:   "public",
				Artifact:     reference.Artifact,
				Revision:     reference.Revision,
				Coverage:     string(retained.Coverage),
			},
		}
	}
	schema, read := parserScope(t)
	codec := retrieval.NewJSONCodec[sourceMeta](schema)
	resolver, err := layout.NewResolver(
		layout.ResolverConfig[sourceMeta]{
			Target:     "layout",
			Schema:     schema,
			Catalog:    host,
			Loader:     host,
			Attributes: codec.Encode,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	textLocation := source.Locator{
		Reference: page.Reference,
		Kind:      source.TextLocation,
		Span:      source.ByteSpan{Start: 6, End: 10},
	}
	// Act.
	resolved, err := resolver.Resolve(
		ctx,
		layout.ResolveRequest{
			Read:      read,
			Locations: []source.Locator{textLocation, cell.Location, image.Location, documentLocation, cell.Location},
		},
	)
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	if len(resolved) != 4 || resolved[0].Text.Text() != "beta" || resolved[1].CellText != "Revenue" ||
		resolved[1].OriginalRegion != cell.Region {
		t.Fatal("real source text/cell geometry/dedup failed")
	}
	if !bytes.Equal(resolved[2].Bytes, pixels) || resolved[2].Derived.Text() != "fixture diagram" ||
		resolved[2].Text.Text() != "" {
		t.Fatal("derived description substituted original image bytes")
	}
	if !bytes.Equal(resolved[3].Bytes, pdfBytes) || resolved[3].Coverage != layout.Partial ||
		resolved[2].Coverage != layout.Partial {
		t.Fatal("document bytes/partial coverage lost")
	}
	resolved[2].Bytes[0] = 0
	if host.payloads[image.Location.Reference].Bytes[0] == 0 {
		t.Fatal("resolved image mutated host retention")
	}
	if host.payloadCalls != 1 {
		t.Fatal("multiple materializations for one batch")
	}
}
