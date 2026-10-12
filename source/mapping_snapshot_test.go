package source_test

import (
	"encoding/json"
	"testing"

	"github.com/skosovsky/ragy/source"
)

func TestMappedSnapshotValidationRemainsOwned(t *testing.T) {
	// Arrange: all caller-visible support/fragment slices are independently owned.
	text := "éabc"
	location := source.Locator{
		Reference: retainedReference(),
		Kind:      source.TextLocation,
		Span:      source.ByteSpan{End: len(text)},
	}
	supports := []source.Locator{location}
	mapping, err := source.SupportedOriginalText(text, supports)
	if err != nil {
		t.Fatal(err)
	}
	original, err := source.OriginalText(location, text)
	if err != nil {
		t.Fatal(err)
	}
	joined, err := source.JoinMapped(" ", original, mapping)
	if err != nil {
		t.Fatal(err)
	}

	// Act: mutate every exposed slice, then try to replace the snapshot with invalid JSON.
	supports[0] = source.Locator{}
	fragments := joined.Fragments()
	for i := range fragments {
		fragments[i].Rendered.End = -1
		fragments[i].Supports[0] = source.Locator{}
	}
	var decoded source.MappedText
	data, err := json.Marshal(joined)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	copyBeforeReplacement := decoded
	replacementErr := json.Unmarshal(
		[]byte(`{"schema":"ragy.source-mapping","state":"mapped","text":"x","fragments":[]}`),
		&decoded,
	)
	part, sliceErr := decoded.Slice(source.ByteSpan{Start: 0, End: 2})
	_, badBoundaryErr := decoded.Slice(source.ByteSpan{Start: 0, End: 1})

	// Assert: cached validity cannot be changed through caller slices or invalid decode.
	if replacementErr == nil || sliceErr != nil || badBoundaryErr == nil {
		t.Fatalf("replace=%v slice=%v boundary=%v", replacementErr, sliceErr, badBoundaryErr)
	}
	if mapping.Validate() != nil || joined.Validate() != nil || decoded.Validate() != nil ||
		copyBeforeReplacement.Validate() != nil {
		t.Fatal("invalid owned snapshot")
	}
	if part.Text() != "é" || decoded.Text() != joined.Text() || copyBeforeReplacement.Text() != joined.Text() {
		t.Fatal("snapshot changed")
	}
	for _, fragment := range decoded.Fragments() {
		if fragment.Rendered.End < 0 || fragment.Supports[0] == (source.Locator{}) {
			t.Fatal("aliased fragment")
		}
	}
}
