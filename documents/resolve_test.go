package documents_test

import (
	"context"
	"testing"

	"github.com/skosovsky/ragy/documents"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

func TestTextResolutionPinsRetainedRevisionAndDeduplicatesLocations(t *testing.T) {
	// Arrange.
	fixture := newHydrationFixture(t)
	fixture.host.payloads[fixture.r1] = retrieval.Document[payloadMeta]{
		ID:      fixture.r1.Artifact,
		Content: "Alpha beta. Gamma.",
		Meta:    payloadMeta{"title": "old"},
	}
	fixture.host.payloads[fixture.r2] = retrieval.Document[payloadMeta]{
		ID:      fixture.r2.Artifact,
		Content: "new revision only",
		Meta:    payloadMeta{"title": "new"},
	}
	beta := source.Locator{Reference: fixture.r1, Kind: source.TextLocation, Span: source.ByteSpan{Start: 6, End: 10}}
	gamma := beta
	gamma.Span = source.ByteSpan{Start: 12, End: 17}
	// Act.
	resolved, err := fixture.reader.ResolveText(
		context.Background(),
		documents.TextResolveRequest{Read: fixture.read, Locations: []source.Locator{beta, gamma, beta}},
	)
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	if len(resolved) != 2 || resolved[0].Text.Text() != "beta" || resolved[1].Text.Text() != "Gamma" {
		t.Fatalf("unexpected citations: %+v", resolved)
	}
	if fixture.host.loadCalls != 1 || len(fixture.host.loaded) != 1 || fixture.host.loaded[0] != fixture.r1 {
		t.Fatal("unnecessary/latest revision materialization")
	}
	if resolved[0].Location != beta || resolved[0].Text.Fragments()[0].Location != beta {
		t.Fatal("retained location changed")
	}
}

func TestTextResolutionUnavailableAndDeniedNeverLoadLatest(t *testing.T) {
	for _, mode := range []string{"deleted", "denied", "revoked", "latest substitution"} {
		t.Run(mode, func(t *testing.T) {
			// Arrange.
			fixture := newHydrationFixture(t)
			switch mode {
			case "deleted":
				delete(fixture.host.descriptors, fixture.r1)
			case "denied":
				fixture.host.descriptors[fixture.r1] = permissionMeta{tenant: "foreign", visibility: "public"}
			case "revoked":
				fixture.host.afterLoad = func() { *fixture.revoked = true }
			case "latest substitution":
				fixture.host.wrongPayload = true
			}
			locator := source.Locator{
				Reference: fixture.r1,
				Kind:      source.TextLocation,
				Span:      source.ByteSpan{Start: 0, End: 3},
			}
			// Act.
			resolved, err := fixture.reader.ResolveText(
				context.Background(),
				documents.TextResolveRequest{Read: fixture.read, Locations: []source.Locator{locator}},
			)
			// Assert.
			if err == nil || len(resolved) != 0 {
				t.Fatalf("citation leaked: %+v error=%v", resolved, err)
			}
			if (mode == "deleted" || mode == "denied") && fixture.host.loadCalls != 0 {
				t.Fatal("inaccessible revision payload loaded")
			}
		})
	}
}

func TestTextResolutionRejectsInvalidUTF8SpanWithoutPartialDelivery(t *testing.T) {
	// Arrange.
	fixture := newHydrationFixture(t)
	fixture.host.payloads[fixture.r1] = retrieval.Document[payloadMeta]{ID: fixture.r1.Artifact, Content: "АБВ"}
	valid := source.Locator{Reference: fixture.r1, Kind: source.TextLocation, Span: source.ByteSpan{Start: 2, End: 4}}
	invalid := valid
	invalid.Span = source.ByteSpan{Start: 1, End: 4}
	// Act.
	resolved, err := fixture.reader.ResolveText(
		context.Background(),
		documents.TextResolveRequest{Read: fixture.read, Locations: []source.Locator{valid, invalid}},
	)
	// Assert.
	if err == nil || len(resolved) != 0 {
		t.Fatal("partial or invalid byte span delivered")
	}
	if fixture.host.loadCalls != 1 {
		t.Fatal("revision loaded more than once")
	}
}

func TestTextResolutionRejectsOtherRepresentationsBeforeIO(t *testing.T) {
	// Arrange.
	fixture := newHydrationFixture(t)
	locator := source.Locator{Reference: fixture.r1, Kind: source.DocumentLocation}
	// Act.
	resolved, err := fixture.reader.ResolveText(
		context.Background(),
		documents.TextResolveRequest{Read: fixture.read, Locations: []source.Locator{locator}},
	)
	// Assert.
	if err == nil || len(resolved) != 0 || fixture.host.describeCalls != 0 || fixture.host.loadCalls != 0 {
		t.Fatal("unsupported representation admitted")
	}
}
