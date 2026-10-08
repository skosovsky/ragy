//go:build e2e

package pdf_test

import (
	"bytes"
	"context"
	"errors"
	"os"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/layout"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

func newModalityRetention(t *testing.T, parsed layout.Document, projected []layout.Projected) *layoutPayloadHost {
	t.Helper()
	pixels, err := os.ReadFile("testdata/diagram.png")
	if err != nil {
		t.Fatal(err)
	}
	host := &layoutPayloadHost{
		parserHost: &parserHost{records: make(map[source.Reference]retainedRecord)},
		payloads:   make(map[source.Reference]layout.Retained),
	}
	for _, page := range parsed.Pages {
		host.payloads[page.Reference] = layout.Retained{
			Original: source.Locator{Reference: page.Reference, Kind: source.PageLocation, Page: page.Geometry},
			Text:     page.Text, Words: page.Words, Coverage: page.Coverage, Diagnostics: page.Diagnostics,
		}
		for _, cell := range page.Cells {
			host.payloads[cell.Location.Reference] = layout.Retained{
				Original: cell.Location, Text: cell.Text, CellRegion: cell.Region, Coverage: page.Coverage,
				Diagnostics: page.Diagnostics,
			}
		}
		for _, image := range page.Images {
			host.payloads[image.Location.Reference] = layout.Retained{
				Original: image.Location, Bytes: pixels, MediaType: "image/png", Coverage: page.Coverage,
				Diagnostics: page.Diagnostics,
			}
		}
	}
	for _, evidence := range projected {
		reference := evidence.Location.Reference
		retained := host.payloads[reference]
		if evidence.Location.Kind == source.ImageLocation {
			retained.Derived = evidence.Text
		}
		host.payloads[reference] = retained
		host.records[reference] = retainedRecord{meta: sourceMeta{
			Organization: "a", Visibility: "public", Artifact: reference.Artifact,
			Revision: reference.Revision, Coverage: string(retained.Coverage),
		}}
	}
	return host
}

func checkRetrievedSupports(
	ctx context.Context, t *testing.T, read access.Binding,
	artifact retrieval.RetrievalContextArtifact[sourceMeta], host *layoutPayloadHost,
) {
	t.Helper()
	schema, _ := parserScope(t)
	codec := retrieval.NewJSONCodec[sourceMeta](schema)
	resolver, err := layout.NewResolver(layout.ResolverConfig[sourceMeta]{
		Target: "layout", Schema: schema, Catalog: host, Loader: host, Attributes: codec.Encode,
	})
	if err != nil {
		t.Fatal(err)
	}
	var locations []source.Locator
	for _, snippet := range artifact.Snippets {
		locations = append(locations, snippet.Supports...)
	}
	if len(locations) == 0 {
		t.Fatal("retrieved artifact has no resolvable source supports")
	}
	before := host.payloadCalls
	resolved, err := resolver.Resolve(ctx, layout.ResolveRequest{Read: read, Locations: locations})
	if err != nil {
		t.Fatal(err)
	}
	if len(resolved) == 0 || host.payloadCalls != before+1 {
		t.Fatal("retrieved supports did not resolve in one admitted batch")
	}
	for _, item := range resolved {
		checkOriginalModality(t, item, host)
	}
	for _, item := range resolved {
		checkModalityRetentionFailure(t, resolver, read, item.Location, host)
	}
}

func checkOriginalModality(t *testing.T, item layout.Resolved, host *layoutPayloadHost) {
	t.Helper()
	retained := host.payloads[item.Location.Reference]
	if item.Location.Reference.Revision != "r1" || item.Original.Reference != item.Location.Reference ||
		item.Coverage != layout.Partial {
		t.Fatal("resolved evidence changed source revision or coverage")
	}
	switch item.Location.Kind {
	case source.TextLocation:
		if item.Text.Text() != retained.Text || item.Text.Fragments()[0].Location != item.Location {
			t.Fatal("retrieved page evidence resolved different original bytes")
		}
	case source.CellLocation:
		if item.CellText != retained.Text || item.OriginalRegion != retained.CellRegion {
			t.Fatal("retrieved merged cell lost original text or extent")
		}
	case source.ImageLocation:
		if !bytes.Equal(item.Bytes, retained.Bytes) || item.Text.Text() != "" ||
			item.Derived.Text() != "fixture diagram" ||
			item.Derived.Fragments()[0].Origin != source.DerivedContent {
			t.Fatal("retrieved description replaced original image")
		}
	case source.DocumentLocation, source.PageLocation, source.RegionLocation:
		t.Fatal("unexpected modality in indexed fixture")
	}
}

func checkModalityRetentionFailure(
	t *testing.T, resolver *layout.Resolver[sourceMeta], read access.Binding,
	location source.Locator, host *layoutPayloadHost,
) {
	t.Helper()
	// Arrange: retain both revisions, then make exactly the retrieved r1 unavailable.
	original := host.records[location.Reference]
	latest := location.Reference
	latest.Revision = "r2"
	latestMeta := original.meta
	latestMeta.Revision = "r2"
	host.records[latest] = retainedRecord{meta: latestMeta}
	latestPayload := host.payloads[location.Reference]
	latestPayload.Original.Reference = latest
	host.payloads[latest] = latestPayload
	delete(host.records, location.Reference)
	before := host.payloadCalls
	request := layout.ResolveRequest{Read: read, Locations: []source.Locator{location}}
	// Act/Assert: missing and denied r1 must not load a payload or resolve latest.
	assertUnavailableModality(t, resolver, request, host, before)
	denied := original
	denied.meta.Organization = "foreign"
	host.records[location.Reference] = denied
	assertUnavailableModality(t, resolver, request, host, before)
	host.records[location.Reference] = original
}

func assertUnavailableModality(
	t *testing.T, resolver *layout.Resolver[sourceMeta], request layout.ResolveRequest,
	host *layoutPayloadHost, before int,
) {
	t.Helper()
	resolved, err := resolver.Resolve(context.Background(), request)
	if !errors.Is(err, ragy.ErrUnavailable) || len(resolved) != 0 || host.payloadCalls != before {
		t.Fatal("unavailable retrieved source loaded payload, substituted latest or exposed partial evidence")
	}
}
