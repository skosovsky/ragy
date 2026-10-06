package chunking_test

import (
	"errors"
	"strings"
	"testing"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/chunking"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
	"github.com/skosovsky/ragy/testutil"
)

func FuzzRecursiveCoordinates(f *testing.F) {
	for _, text := range []string{"alpha beta gamma delta\n\nTAIL", "repeat repeat repeat", " 你好。αα \n end ", "# A\ntext\n# B\nlast"} {
		f.Add(text, uint8(8), uint8(2))
	}
	f.Fuzz(func(t *testing.T, text string, sizeByte, overlapByte uint8) {
		checkRecursiveCoordinateProperty(t, text, sizeByte, overlapByte)
	})
}
func TestInvalidSegmenterRangesRejectBeforeEmbedding(t *testing.T) {
	// Arrange: transformed/reordered segments cannot be treated as original ranges.
	segmenter := &fixedSegmenter{Spans: []source.ByteSpan{{Start: 4, End: 7}, {Start: 0, End: 3}}}
	embedder := &testutil.DenseEmbedder{}
	splitter, err := chunking.NewSemantic[struct{}](embedder, segmenter, 0.5, 1)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	chunks, err := splitter.Split(t.Context(), retrieval.Document[struct{}]{ID: "segments", Content: "one two"})
	// Assert.
	if !errors.Is(err, ragy.ErrProtocol) || chunks != nil || len(embedder.Requests) != 0 {
		t.Fatal("invalid segmenter invoked embedding", err)
	}
}

func checkRecursiveCoordinateProperty(t *testing.T, text string, sizeByte, overlapByte uint8) {
	t.Helper()
	// Arrange: generate valid bounded rune sizes and original source mapping.
	size := int(sizeByte)%64 + 1
	overlap := int(overlapByte) % size
	splitter, err := chunking.NewRecursive[struct{}](size, overlap, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !utf8.ValidString(text) || strings.TrimSpace(text) == "" {
		// Act/Assert: invalid source has no fragments or invented ranges.
		chunks, splitErr := splitter.Split(t.Context(), retrieval.Document[struct{}]{ID: "invalid", Content: text})
		if splitErr == nil || chunks != nil {
			t.Fatal("invalid source accepted")
		}
		return
	}
	doc := originalDocument(t, text)
	// Act.
	chunks, err := splitter.Split(t.Context(), doc)
	// Assert: no substring recovery, foreign coordinates or reordering.
	if err != nil {
		t.Fatal(err)
	}
	previous := 0
	for i, chunk := range chunks {
		if chunk.InputSpan.ValidateText(text) != nil || chunk.InputSpan.Start < previous || chunk.Index != i ||
			chunk.Total != len(chunks) ||
			chunk.Content != text[chunk.InputSpan.Start:chunk.InputSpan.End] ||
			chunk.SourceMapping.Text() != chunk.Content ||
			utf8.RuneCountInString(chunk.Content) > size {
			t.Fatalf("coordinate invariant: %#v", chunk)
		}
		previous = chunk.InputSpan.Start
		for _, fragment := range chunk.SourceMapping.Fragments() {
			if fragment.Precision != source.ExactPrecision || fragment.Location.Reference.Revision != "r1" {
				t.Fatal("lost source identity")
			}
		}
	}
}
