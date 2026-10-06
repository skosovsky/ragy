package chunking_test

import (
	"context"
	"errors"
	"math"
	"strings"
	"sync/atomic"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/chunking"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
	"github.com/skosovsky/ragy/testutil"
)

func originalDocument(t *testing.T, text string) retrieval.Document[struct{}] {
	t.Helper()
	location := source.Locator{
		Reference: source.Reference{
			Namespace:         "n",
			Source:            "policy",
			Revision:          "r1",
			Transformation:    "original",
			AccessFingerprint: "acl",
			Artifact:          "text",
			Representation:    "utf8",
		},
		Kind: source.TextLocation,
		Span: source.ByteSpan{Start: 0, End: len(text)},
	}
	mapping, err := source.OriginalText(location, text)
	if err != nil {
		t.Fatal(err)
	}
	return retrieval.Document[struct{}]{ID: "policy", Content: text, SourceMapping: mapping}
}
func TestRecursivePreservesSourceOrder(t *testing.T) {
	// Arrange: earlier long text must finish splitting before the late short section.
	doc := originalDocument(t, "alpha beta gamma delta\n\nTAIL")
	splitter, err := chunking.NewRecursive[struct{}](8, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	chunks, err := splitter.Split(t.Context(), doc)
	// Assert.
	if err != nil || len(chunks) < 2 || chunks[0].Content != "alpha" || chunks[len(chunks)-1].Content != "TAIL" {
		t.Fatalf("order: %v %#v", err, chunks)
	}
	previous := 0
	for i, chunk := range chunks {
		if chunk.InputSpan.Start < previous || chunk.Index != i || chunk.Total != len(chunks) {
			t.Fatalf("position: %#v", chunk)
		}
		previous = chunk.InputSpan.Start
	}
}
func TestRangeMappingOnRepeatedUnicodeAndOverlap(t *testing.T) {
	for _, text := range []string{"  αααααααααααα  ", "repeat repeat repeat\n\nrepeat", "\t你好。你好。世界\n TAIL  ", "first\r\nsecond\r\nthird"} {
		t.Run(text, func(t *testing.T) {
			checkRangeMapping(t, text)
		})
	}
}
func TestProjectionSeparatesContextFromOriginalQuote(t *testing.T) {
	// Arrange.
	doc := originalDocument(t, "  original evidence  ")
	splitter, err := chunking.NewRecursive[struct{}](64, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	chunks, err := splitter.Split(t.Context(), doc)
	if err != nil {
		t.Fatal(err)
	}
	chunks[0].Context = "synthetic explanation"
	// Act.
	projected, err := chunking.ProjectDocuments(
		chunks,
		chunking.ProjectionConfig[struct{}, struct{}, struct{}]{
			Source:    chunking.SourceDescriptor[struct{}]{ID: doc.ID},
			IndexText: func(c chunking.Chunk[struct{}]) (string, error) { return c.Context + "\n" + c.Content, nil },
			MetadataProjector: chunking.MetadataProjectorFunc[struct{}, struct{}, struct{}](
				func(chunking.SourceDescriptor[struct{}], chunking.Chunk[struct{}], chunking.ChunkIdentity) (struct{}, error) {
					return struct{}{}, nil
				},
			),
		},
	)
	// Assert: retrieval quote remains exact; synthetic index bytes have no exact claim.
	if err != nil || len(projected) != 1 {
		t.Fatal(err)
	}
	item := projected[0]
	if item.Content != "original evidence" || item.IndexText != "synthetic explanation\noriginal evidence" ||
		item.SourceMapping.Fragments()[0].Precision != source.ExactPrecision ||
		item.IndexMapping.Fragments()[0].Precision != source.UnavailablePrecision ||
		item.IndexMapping.Fragments()[0].Origin != source.DerivedContent {
		t.Fatalf("projection: %#v", item)
	}
}
func TestDerivedAndUnmappedSourcesCannotAcquireExactCoordinates(t *testing.T) {
	for _, mapped := range []bool{false, true} {
		t.Run(map[bool]string{false: "unmapped", true: "derived"}[mapped], func(t *testing.T) {
			checkMappingPrecision(t, mapped)
		})
	}
}
func TestSemanticGroupingIsScaleInvariant(t *testing.T) {
	for _, scale := range []float32{1, 1000, math.MaxFloat32} {
		t.Run("scale", func(t *testing.T) {
			// Arrange.
			embedder := &testutil.DenseEmbedder{Vectors: [][]float32{{scale, 0}, {scale, 0}}}
			splitter, err := chunking.NewSemantic[struct{}](embedder, chunking.DefaultSentenceSegmenter{}, 0.99, 1)
			if err != nil {
				t.Fatal(err)
			}
			doc := originalDocument(t, "First.\n\nSecond.")
			// Act.
			chunks, err := splitter.Split(t.Context(), doc)
			// Assert: contiguous source separators survive grouping at every finite scale.
			if err != nil || len(chunks) != 1 || chunks[0].Content != doc.Content ||
				chunks[0].SourceMapping.Text() != doc.Content {
				t.Fatalf("semantic scale: %v %#v", err, chunks)
			}
		})
	}
}
func TestSemanticRejectsNonFiniteInputs(t *testing.T) {
	for _, threshold := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		// Arrange/Act.
		_, err := chunking.NewSemantic[struct{}](
			&testutil.DenseEmbedder{},
			chunking.DefaultSentenceSegmenter{},
			threshold,
			1,
		)
		// Assert.
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatal("non-finite threshold accepted", err)
		}
	}
	for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		// Arrange.
		splitter, err := chunking.NewSemantic[struct{}](
			&testutil.DenseEmbedder{Vectors: [][]float32{{value, 0}, {1, 0}}},
			chunking.DefaultSentenceSegmenter{},
			0.5,
			1,
		)
		if err != nil {
			t.Fatal(err)
		}
		// Act.
		chunks, err := splitter.Split(t.Context(), originalDocument(t, "One. Two."))
		// Assert.
		if !errors.Is(err, ragy.ErrProtocol) || chunks != nil {
			t.Fatal("non-finite vector accepted", err)
		}
	}
}

type cancellationChecks struct {
	context.Context

	cancel context.CancelFunc
	checks atomic.Int64
}

func (c *cancellationChecks) Err() error {
	if c.checks.Add(1) == 1024 {
		c.cancel()
	}
	return c.Context.Err()
}
func TestRecursiveChecksCancellationInsideCPUWork(t *testing.T) {
	// Arrange: cancel deterministically during the source rune scan.
	parent, cancel := context.WithCancel(t.Context())
	defer cancel()
	ctx := &cancellationChecks{Context: parent, cancel: cancel}
	splitter, err := chunking.NewRecursive[struct{}](8, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	chunks, err := splitter.Split(ctx, retrieval.Document[struct{}]{ID: "cpu", Content: strings.Repeat("α", 10000)})
	// Assert.
	if !errors.Is(err, context.Canceled) || chunks != nil || ctx.checks.Load() < 1024 {
		t.Fatalf("CPU cancellation: %v", err)
	}
}

func checkRangeMapping(t *testing.T, text string) {
	t.Helper()
	// Arrange.
	doc := originalDocument(t, text)
	splitter, err := chunking.NewRecursive[struct{}](5, 2, []string{"\n\n", "\n"})
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	chunks, err := splitter.Split(t.Context(), doc)
	// Assert: original byte intervals distinguish repeated text and overlap.
	if err != nil || len(chunks) == 0 {
		t.Fatal(err)
	}
	for _, chunk := range chunks {
		if chunk.Content != text[chunk.InputSpan.Start:chunk.InputSpan.End] ||
			chunk.SourceMapping.Text() != chunk.Content {
			t.Fatalf("mapping: %#v", chunk)
		}
		fragments := chunk.SourceMapping.Fragments()
		if len(fragments) != 1 || fragments[0].Precision != source.ExactPrecision ||
			fragments[0].Location.Span != chunk.InputSpan ||
			fragments[0].Location.Reference.Revision != "r1" {
			t.Fatalf("exact: %#v", fragments)
		}
	}
}

func checkMappingPrecision(t *testing.T, mapped bool) {
	t.Helper()
	// Arrange.
	doc := originalDocument(t, "repeated repeated")
	if mapped {
		derived, err := source.DerivedText(doc.Content, doc.SourceMapping.Supports())
		if err != nil {
			t.Fatal(err)
		}
		doc.SourceMapping = derived
	} else {
		doc.SourceMapping = source.MappedText{}
	}
	splitter, err := chunking.NewRecursive[struct{}](5, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	chunks, err := splitter.Split(t.Context(), doc)
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	for _, chunk := range chunks {
		for _, fragment := range chunk.SourceMapping.Fragments() {
			if fragment.Precision == source.ExactPrecision {
				t.Fatal("invented exact mapping")
			}
		}
	}
}
