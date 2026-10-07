package source_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/skosovsky/ragy/source"
)

// Mirrors long-page word mapping and joined-fragment validation, without I/O/OCR.
func BenchmarkLongPageWordMappings(b *testing.B) {
	for _, words := range []int{512, 2048} {
		b.Run(fmt.Sprintf("words=%d", words), func(b *testing.B) { benchmarkWordMappings(b, words) })
	}
}
func benchmarkWordMappings(b *testing.B, words int) {
	word := "длинноеслово "
	text := strings.Repeat(word, words)
	reference := source.Reference{
		Namespace: "n", Source: "page", Revision: "r1", Transformation: "original",
		AccessFingerprint: "acl", Artifact: "p1", Representation: "utf8",
	}
	locations := make([]source.Locator, words)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		for i := range words {
			locations[i] = source.Locator{Kind: source.TextLocation, Reference: reference,
				Span: source.ByteSpan{Start: i * len(word), End: (i+1)*len(word) - 1}}
		}
		parts, err := source.OriginalTexts(locations, text)
		if err != nil {
			b.Fatal(err)
		}

		joined, err := source.JoinMapped(" ", parts...)
		if err != nil || joined.Validate() != nil || joined.Text() != strings.TrimSpace(text) {
			b.Fatal("word mapping changed", err)
		}
	}
}
