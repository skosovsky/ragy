package source_test

import (
	"encoding/json"
	"testing"

	"github.com/skosovsky/ragy/source"
)

func TestOriginalTextsMatchesSingleRangeContract(t *testing.T) {
	// Arrange: every interval over multibyte/repeated content, including invalid bounds.
	text := " аβа " + string([]byte{0xf0, 0x9f, 0x98, 0x80})
	ref := originalBeta(t).Supports()[0].Reference
	for start := -1; start <= len(text); start++ {
		for end := start; end <= len(text)+1; end++ {
			location := source.Locator{
				Reference: ref, Kind: source.TextLocation, Span: source.ByteSpan{Start: start, End: end},
			}
			// Act.
			single, singleErr := source.OriginalText(location, text)
			batch, batchErr := source.OriginalTexts([]source.Locator{location}, text)
			// Assert: same validity; admitted mappings preserve exact original bytes/order.
			if (singleErr == nil) != (batchErr == nil) {
				t.Fatal(start, end, singleErr, batchErr)
			}
			if singleErr != nil {
				if batch != nil {
					t.Fatal("invalid batch exposed payload")
				}
				continue
			}
			first, err := json.Marshal(single)
			if err != nil {
				t.Fatal(err)
			}
			second, err := json.Marshal(batch[0])
			if err != nil || string(first) != string(second) || batch[0].Validate() != nil {
				t.Fatal(start, end, err)
			}
		}
	}
}

func TestOriginalTextsWholeBatchOwnershipAndFailure(t *testing.T) {
	// Arrange: repeated source text with distinct spans, reverse encounter order preserved.
	ref := originalBeta(t).Supports()[0].Reference
	locations := []source.Locator{
		{Reference: ref, Kind: source.TextLocation, Span: source.ByteSpan{Start: 2, End: 3}},
		{Reference: ref, Kind: source.TextLocation, Span: source.ByteSpan{Start: 0, End: 1}},
	}
	// Act.
	batch, err := source.OriginalTexts(locations, "a a")
	// Assert.
	if err != nil || len(batch) != 2 || batch[0].Fragments()[0].Location != locations[0] {
		t.Fatal(batch, err)
	}
	locations[0].Reference.Revision = "mutated"
	exported := batch[0].Fragments()
	exported[0].Supports[0].Reference.Revision = "mutated"
	if batch[0].Supports()[0].Reference.Revision != ref.Revision {
		t.Fatal("batch aliases")
	}
	locations[1].Span.End = 100
	if output, batchErr := source.OriginalTexts(locations, "a a"); batchErr == nil || output != nil {
		t.Fatal("late error exposed payload", batchErr)
	}
	if output, batchErr := source.OriginalTexts(locations, string([]byte{0xff})); batchErr == nil || output != nil {
		t.Fatal("malformed snapshot admitted")
	}
	if output, batchErr := source.OriginalTexts(nil, ""); batchErr != nil || output != nil {
		t.Fatal(output, batchErr)
	}
}
