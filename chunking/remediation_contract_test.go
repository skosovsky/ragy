package chunking

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

type contractBase []Chunk[struct{}]

func (c contractBase) Split(context.Context, retrieval.Document[struct{}]) ([]Chunk[struct{}], error) {
	return c, nil
}

type contractContext struct {
	calls  int
	output string
}

func (g *contractContext) Context(context.Context, retrieval.Document[struct{}], Chunk[struct{}]) (string, error) {
	g.calls++
	return g.output, nil
}

func TestContextualAdmitsCompleteBatchBeforeGenerators(t *testing.T) {
	for _, scenario := range []string{"duplicate-index", "unordered", "wrong-total", "wrong-source", "bad-content", "bad-context", "bad-span"} {
		t.Run(scenario, func(t *testing.T) {
			// Arrange: second chunk fails while first is valid.
			chunks := contractBase{
				{ID: "a", SourceID: "src", Index: 0, Total: 2, Content: "ab"},
				{ID: "b", SourceID: "src", Index: 1, Total: 2, Content: "cd"},
			}
			switch scenario {
			case "duplicate-index":
				chunks[1].Index = 0
			case "unordered":
				chunks[0].Index = 1
				chunks[1].Index = 0
			case "wrong-total":
				chunks[1].Total = 3
			case "wrong-source":
				chunks[1].SourceID = "other"
			case "bad-content":
				chunks[1].Content = string([]byte{0xff})
			case "bad-context":
				chunks[1].Context = string([]byte{0xff})
			case "bad-span":
				chunks[1].InputSpan = source.ByteSpan{Start: 0, End: 2}
			}
			generator := &contractContext{}
			splitter, err := NewContextual[struct{}](chunks, generator, 1)
			if err != nil {
				t.Fatal(err)
			}
			// Act.
			result, err := splitter.Split(t.Context(), retrieval.Document[struct{}]{ID: "src", Content: "abcd"})
			// Assert: no identity repair by sorting, no callback or partial chunks.
			if !errors.Is(err, ragy.ErrProtocol) || result != nil || generator.calls != 0 {
				t.Fatal(result, err, generator.calls)
			}
		})
	}
}

func TestContextualUnknownTotalAndGeneratedUTF8(t *testing.T) {
	// Arrange: standalone Total0 is explicitly allowed, with exact supplied span.
	base := contractBase{{ID: "a", SourceID: "src", Content: "ab", InputSpan: source.ByteSpan{Start: 0, End: 2}}}
	generator := &contractContext{output: "context"}
	splitter, err := NewContextual[struct{}](base, generator, 1)
	if err != nil {
		t.Fatal(err)
	}
	doc := retrieval.Document[struct{}]{ID: "src", Content: "ab"}
	// Act.
	chunks, err := splitter.Split(t.Context(), doc)
	// Assert.
	if err != nil || len(chunks) != 1 || chunks[0].Context != "context" || chunks[0].Total != 0 {
		t.Fatal(chunks, err)
	}
	generator.output = string([]byte{0xff})
	chunks, err = splitter.Split(t.Context(), doc)
	if !errors.Is(err, ragy.ErrProtocol) || chunks != nil {
		t.Fatal(chunks, err)
	}
}

func TestSentenceRangesPreserveNonWhitespaceCoverage(t *testing.T) {
	for _, spans := range [][]source.ByteSpan{
		{{Start: 2, End: 4}}, {{Start: 0, End: 2}}, {{Start: 0, End: 1}, {Start: 2, End: 4}},
	} {
		// Arrange/Act: prefix, tail, or interior substantive bytes omitted.
		_, err := sentenceTexts(t.Context(), "abcd", spans)
		// Assert.
		if !errors.Is(err, ragy.ErrProtocol) {
			t.Fatal(spans, err)
		}
	}
	spans := []source.ByteSpan{{Start: 1, End: 2}, {Start: 3, End: 4}}
	texts, err := sentenceTexts(t.Context(), " a b ", spans)
	if err != nil || len(texts) != 2 {
		t.Fatal(texts, err)
	}
}

func TestProjectionSuppressesAllPayloadOnLateErrors(t *testing.T) {
	for _, scenario := range []string{"identity", "metadata", "index", "document"} {
		t.Run(scenario, func(t *testing.T) {
			checkProjectionLateError(t, scenario)
		})
	}
}

func TestDefaultProjectionIdentityAuthority(t *testing.T) {
	// Arrange.
	policy := DefaultChunkIdentityPolicy[struct{}, struct{}]{}
	descriptor := SourceDescriptor[struct{}]{ID: "src", URI: "host://r1", StorageID: "source-storage"}
	// Act/Assert: disagreeing source identity is rejected; optional chunk id falls back.
	_, mismatchErr := policy.Identity(descriptor, Chunk[struct{}]{SourceID: "other", Content: "a"})
	if !errors.Is(mismatchErr, ragy.ErrInvalidArgument) {
		t.Fatal(mismatchErr)
	}
	for i := range 2 {
		id, err := policy.Identity(descriptor, Chunk[struct{}]{Index: i, Content: "a"})
		if err != nil || id.StorageID != "source-storage" || id.SourceID != "src" ||
			id.DocumentID == "" || id.MergeKey == id.DocumentID {
			t.Fatal(id, err)
		}
	}
}

func checkProjectionLateError(t *testing.T, scenario string) {
	t.Helper()
	// Arrange: first document would otherwise be delivered.
	failure := errors.New("late projection failure")
	chunks := []Chunk[struct{}]{
		{ID: "a", SourceID: "src", Content: "a"},
		{ID: "b", SourceID: "src", Index: 1, Content: "b"},
	}
	cfg := ProjectionConfig[struct{}, struct{}, struct{}]{
		Source: SourceDescriptor[struct{}]{ID: "src"},
		IndexText: func(c Chunk[struct{}]) (string, error) {
			if c.Index == 1 && scenario == "index" {
				return "", failure
			}
			return c.Content, nil
		},
		IdentityPolicy: ChunkIdentityPolicyFunc[struct{}, struct{}](func(
			s SourceDescriptor[struct{}], c Chunk[struct{}],
		) (ChunkIdentity, error) {
			if c.Index == 1 && scenario == "identity" {
				return ChunkIdentity{}, failure
			}
			id, err := (DefaultChunkIdentityPolicy[struct{}, struct{}]{}).Identity(s, c)
			if c.Index == 1 && scenario == "document" {
				id.DocumentID = ""
			}
			return id, err
		}),
		MetadataProjector: MetadataProjectorFunc[struct{}, struct{}, struct{}](func(
			_ SourceDescriptor[struct{}], c Chunk[struct{}], _ ChunkIdentity,
		) (struct{}, error) {
			if c.Index == 1 && scenario == "metadata" {
				return struct{}{}, failure
			}
			return struct{}{}, nil
		}),
	}
	// Act.
	result, err := ProjectDocuments(chunks, cfg)
	// Assert.
	if err == nil || result != nil {
		t.Fatal(result, err)
	}
	if scenario != "document" && !errors.Is(err, failure) {
		t.Fatal(err)
	}
}

func TestDeclaredSplitterGrammarAndBoundaryOmissions(t *testing.T) {
	// Arrange/Act: punctuation separators and repeated boundary tokens are explicit omissions.
	splitter, err := NewRecursive[struct{}](2, 0, []string{"|"})
	if err != nil {
		t.Fatal(err)
	}
	chunks, err := splitter.Split(t.Context(), retrieval.Document[struct{}]{ID: "src", Content: "ab||cd"})
	// Assert: no substring coordinate recovery or delimiter fabrication.
	if err != nil || len(chunks) != 2 || chunks[0].Content != "ab" ||
		chunks[1].Content != "cd" || chunks[1].InputSpan.Start != 4 {
		t.Fatal(chunks, err)
	}
	fallback, err := NewRecursive[struct{}](8, 3, nil)
	if err != nil {
		t.Fatal(err)
	}
	chunks, err = fallback.Split(t.Context(), retrieval.Document[struct{}]{ID: "src", Content: "abcdefghij"})
	if err != nil || len(chunks) != 2 || chunks[0].InputSpan.End != 8 || chunks[1].InputSpan.Start != 5 {
		t.Fatal(chunks, err)
	}
	spans, err := markdownRanges(t.Context(), "# H\n```\n#code\n```\nSetext\n===\n")
	if err != nil || len(spans) != 2 || spans[1].Start != 8 {
		t.Fatal(spans, err)
	}
	sentences, err := (DefaultSentenceSegmenter{}).Split(t.Context(), "v1.2 Mr.X!")
	if err != nil || len(sentences) != 3 {
		t.Fatal(sentences, err)
	}
}
