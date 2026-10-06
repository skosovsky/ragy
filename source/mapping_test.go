package source_test

import (
	"testing"

	"github.com/skosovsky/ragy/source"
)

func originalBeta(t *testing.T) source.MappedText {
	t.Helper()
	locator := source.Locator{
		Reference: retainedReference(),
		Kind:      source.TextLocation,
		Span:      source.ByteSpan{Start: 6, End: 10},
	}
	mapped, err := source.OriginalText(locator, "Alpha beta. Gamma.")
	if err != nil {
		t.Fatal(err)
	}
	return mapped
}

func TestTruncationRecalculatesExactSourceSpan(t *testing.T) {
	// Arrange.
	beta := originalBeta(t)
	// Act.
	trimmed, err := beta.Slice(source.ByteSpan{Start: 0, End: 2})
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	fragments := trimmed.Fragments()
	if trimmed.Text() != "be" || len(fragments) != 1 ||
		fragments[0].Location.Span != (source.ByteSpan{Start: 6, End: 8}) {
		t.Fatalf("wrong truncation: %q %+v", trimmed.Text(), fragments)
	}
	if fragments[0].Precision != source.ExactPrecision || fragments[0].Origin != source.OriginalContent {
		t.Fatal("lost exact origin")
	}
	if beta.Fragments()[0].Location.Span != (source.ByteSpan{Start: 6, End: 10}) {
		t.Fatal("mutated original mapping")
	}
}

func TestPrefixEnrichmentRetainsOriginalOffsets(t *testing.T) {
	// Arrange.
	beta := originalBeta(t)
	prefix, err := source.DerivedText("context: ", beta.Supports())
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	enriched, err := source.JoinMapped("", prefix, beta)
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	fragments := enriched.Fragments()
	if enriched.Text() != "context: beta" || fragments[1].Rendered != (source.ByteSpan{Start: 9, End: 13}) ||
		fragments[1].Location.Span != (source.ByteSpan{Start: 6, End: 10}) {
		t.Fatalf("enrichment shifted source: %+v", fragments)
	}
	if fragments[0].Precision != source.UnavailablePrecision || fragments[0].Origin != source.DerivedContent {
		t.Fatal("derived prefix presented as original")
	}
	clipped, err := enriched.Slice(source.ByteSpan{Start: 7, End: 11})
	if err != nil {
		t.Fatal(err)
	}
	if clipped.Text() != ": be" || clipped.Fragments()[1].Location.Span != (source.ByteSpan{Start: 6, End: 8}) {
		t.Fatal("cross-fragment truncation is incorrect")
	}
}

func TestMultiSourceJoinOwnsAndRetainsEverySupport(t *testing.T) {
	// Arrange.
	first := originalBeta(t)
	other := source.Locator{
		Reference: retainedReference(),
		Kind:      source.TextLocation,
		Span:      source.ByteSpan{Start: 2, End: 4},
	}
	other.Reference.Source = "second"
	second, err := source.OriginalText(other, "АБВ")
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	joined, err := source.JoinMapped(" | ", first, second)
	if err != nil {
		t.Fatal(err)
	}
	exported := joined.Fragments()
	exported[0].Supports[0].Reference.Source = "mutated"
	exported[0].Location.Reference.Revision = "r2"
	// Assert.
	if joined.Text() != "beta | Б" || len(joined.Supports()) != 2 || len(joined.Fragments()) != 3 {
		t.Fatal("joined sources lost")
	}
	if joined.Fragments()[0].Supports[0].Reference.Source != "manual" ||
		joined.Fragments()[0].Location.Reference.Revision != "r1" {
		t.Fatal("export mutated retained snapshot")
	}
	if joined.Fragments()[1].Precision != source.UnavailablePrecision {
		t.Fatal("separator has false precision")
	}
	if _, sliceErr := joined.Slice(source.ByteSpan{Start: 0, End: 8}); sliceErr == nil {
		t.Fatal("UTF-8 interior admitted")
	}
}

func TestMappingRejectsAbsentMalformedAndWrongRepresentationLocations(t *testing.T) {
	var absent source.MappedText
	if err := absent.Validate(); err == nil {
		t.Fatal("absent mapping claims exact evidence")
	}
	beta := originalBeta(t)
	if _, err := source.DerivedText("description", nil); err == nil {
		t.Fatal("unsupported description admitted")
	}
	if _, err := source.DerivedText(string([]byte{0xff}), beta.Supports()); err == nil {
		t.Fatal("invalid text admitted")
	}
	locator := beta.Supports()[0]
	locator.Kind = source.DocumentLocation
	locator.Span = source.ByteSpan{}
	if _, err := source.OriginalText(locator, "beta"); err == nil {
		t.Fatal("document-level location used as byte mapping")
	}
	invalid := beta.Supports()[0]
	invalid.Span = source.ByteSpan{Start: 1, End: 4}
	if _, err := source.OriginalText(invalid, "АБВ"); err == nil {
		t.Fatal("codepoint interior admitted")
	}
}
