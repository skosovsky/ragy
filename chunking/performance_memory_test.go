package chunking

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"testing"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

//nolint:gochecknoglobals // Retain benchmark output so the measured public result escapes as in the audit.
var performanceChunkSink []Chunk[struct{}]

func performanceMappedText(tb testing.TB, text string) source.MappedText {
	tb.Helper()
	mapping, err := source.OriginalText(source.Locator{
		Reference: source.Reference{
			Namespace:         "n",
			Source:            "s",
			Revision:          "r",
			Transformation:    "t",
			AccessFingerprint: "a",
			Artifact:          "f",
			Representation:    "text",
		},
		Kind: source.TextLocation,
		Span: source.ByteSpan{End: len(text)},
	}, text)
	if err != nil {
		tb.Fatal(err)
	}
	return mapping
}

// BenchmarkAuditRecursive preserves the exact audit fixture and name for comparison.
func BenchmarkAuditRecursive(b *testing.B) {
	for _, size := range []int{16384, 65536, 262144, 1048576} {
		b.Run(strconv.Itoa(size), func(b *testing.B) {
			benchmarkRecursive(b, strings.Repeat("a", size), false, 0)
		})
	}
}

func BenchmarkRecursiveValidationMatrix(b *testing.B) {
	for _, mapped := range []bool{false, true} {
		for _, unit := range []string{"a", "é"} {
			for _, overlap := range []int{0, 32} {
				for _, size := range []int{128, 16384, 65536, 262144, 1048576} {
					encoding := "ASCII"
					if len(unit) > 1 {
						encoding = "Multibyte"
					}
					name := fmt.Sprintf("Mapped=%t/%s/Overlap=%d/Bytes=%d", mapped, encoding, overlap, size)
					b.Run(name, func(b *testing.B) {
						benchmarkRecursive(b, strings.Repeat(unit, size/len(unit)), mapped, overlap)
					})
				}
			}
		}
	}
}

func benchmarkRecursive(b *testing.B, text string, mapped bool, overlap int) {
	doc := retrieval.Document[struct{}]{ID: "d", Content: text}
	if mapped {
		doc.SourceMapping = performanceMappedText(b, text)
	}
	splitter, err := NewRecursive[struct{}](256, overlap, nil)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.SetBytes(int64(len(text)))
	b.ResetTimer()
	for range b.N {
		performanceChunkSink, err = splitter.Split(context.Background(), doc)
		if err != nil {
			b.Fatal(err)
		}
	}
}

//nolint:gocognit // Keep the AAA contract matrix together to compare byte spans and mapping ownership across variants.
func TestRecursiveValidatedSnapshotMatrix(t *testing.T) {
	for _, mapped := range []bool{false, true} {
		for _, overlap := range []int{0, 2} {
			for _, unit := range []string{"a", "é", "界", "🙂"} {
				t.Run(fmt.Sprintf("Mapped=%t/Overlap=%d/%s", mapped, overlap, unit), func(t *testing.T) {
					// Arrange: leading/trailing whitespace exercises byte trimming.
					text := " \t" + strings.Repeat(unit, 40) + "\n "
					doc := retrieval.Document[struct{}]{ID: "d", Content: text}
					if mapped {
						doc.SourceMapping = performanceMappedText(t, text)
					}
					splitter, err := NewRecursive[struct{}](8, overlap, []string{"|"})
					if err != nil {
						t.Fatal(err)
					}

					// Act.
					chunks, err := splitter.Split(t.Context(), doc)

					// Assert: exact spans, overlap and source offsets remain byte accurate.
					if err != nil {
						t.Fatal(err)
					}
					covered := 2
					for i, chunk := range chunks {
						if err := chunk.InputSpan.ValidateText(text); err != nil {
							t.Fatal(err)
						}
						if chunk.Content != text[chunk.InputSpan.Start:chunk.InputSpan.End] ||
							!utf8.ValidString(chunk.Content) {
							t.Fatal("content/span mismatch")
						}
						expectedStart := 2 + i*(8-overlap)*len(unit)
						if chunk.InputSpan.Start != expectedStart || chunk.InputSpan.End > 2+40*len(unit) ||
							chunk.InputSpan.Start > covered {
							t.Fatalf(
								"span: %+v, expected start %d, covered %d",
								chunk.InputSpan,
								expectedStart,
								covered,
							)
						}
						covered = chunk.InputSpan.End
						if mapped {
							fragments := chunk.SourceMapping.Fragments()
							if len(fragments) != 1 || fragments[0].Location.Span != chunk.InputSpan ||
								chunk.SourceMapping.Text() != chunk.Content {
								t.Fatal("mapping changed")
							}
							fragments[0].Supports[0] = source.Locator{}
							if chunk.SourceMapping.Validate() != nil ||
								chunk.SourceMapping.Fragments()[0].Supports[0] == (source.Locator{}) {
								t.Fatal("mapping aliases owned output")
							}
						}
					}
					if covered != 2+40*len(unit) {
						t.Fatal("incomplete coverage")
					}
				})
			}
		}
	}
}

func TestChunkingSnapshotAdmissionAndBoundaries(t *testing.T) {
	t.Run("malformed input", func(t *testing.T) {
		// Arrange.
		splitter, _ := NewRecursive[struct{}](2, 0, nil)
		doc := retrieval.Document[struct{}]{ID: "d", Content: "a\xffb"}
		// Act.
		chunks, err := splitter.Split(t.Context(), doc)
		// Assert.
		if !errors.Is(err, ragy.ErrInvalidArgument) || chunks != nil {
			t.Fatalf("chunks=%v err=%v", chunks, err)
		}
	})
	for _, span := range []source.ByteSpan{{Start: -1, End: 1}, {Start: 0, End: 5}, {Start: 1, End: 2}, {Start: 0, End: 1}, {Start: 2, End: 2}} {
		t.Run(fmt.Sprint(span), func(t *testing.T) {
			// Arrange: a validated snapshot with adversarial segmenter spans.
			doc, err := validateSource(t.Context(), retrieval.Document[struct{}]{ID: "d", Content: "éa"})
			if err != nil {
				t.Fatal(err)
			}
			// Act.
			chunks, err := buildChunks(t.Context(), doc, []source.ByteSpan{span})
			// Assert.
			if !errors.Is(err, ragy.ErrInvalidArgument) || chunks != nil {
				t.Fatalf("chunks=%v err=%v", chunks, err)
			}
		})
	}
	t.Run("cancelled", func(t *testing.T) {
		// Arrange.
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		splitter, _ := NewRecursive[struct{}](2, 0, nil)
		// Act.
		chunks, err := splitter.Split(ctx, retrieval.Document[struct{}]{ID: "d", Content: "éa"})
		// Assert.
		if !errors.Is(err, context.Canceled) || chunks != nil {
			t.Fatalf("chunks=%v err=%v", chunks, err)
		}
	})
}
