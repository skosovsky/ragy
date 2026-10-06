# Chunking and index projection

A splitter takes a typed `retrieval.Document` and returns original fragments. Sizes and overlap count Unicode runes; `InputSpan` addresses UTF-8 bytes in that input's `Content`. Separators and trimming adjust these ranges while splitting. Repeated text never requires a substring search to recover its coordinates.

Provide `Document.SourceMapping` from the retained representation (for example `source.OriginalText` or `layout.Project`). Every chunk slices that mapping. Derived image descriptions and logical cell text retain support-only precision; absent mapping does not become an exact quote. Source references include the original revision, transformation and access fingerprint.

`SentenceSegmenter.Split(ctx, text)` returns ordered byte ranges. A custom segmenter chooses boundaries, not replacement text. Semantic groups preserve the original bytes between their first and last sentences. Finite nonzero embeddings need not have unit norm.

Projection separates quote content from indexing:

```go
projected, err := chunking.ProjectDocuments(chunks,
    chunking.ProjectionConfig[SourceMeta, ChunkMeta, DocumentMeta]{
        Source: sourceDescriptor,
        MetadataProjector: metadataProjector,
        IndexText: func(c chunking.Chunk[ChunkMeta]) (string, error) {
            return c.Context + "\n" + c.Content, nil
        },
    })
```

The host passes `projected[i].IndexText` to its encoder or dedicated lexical field and stores `projected[i].Document` as the original retrieval payload. `IndexMapping` is derived/support-only when the selected text differs from original content. `OriginalIndexText[T]` explicitly chooses unchanged content. The library does not choose a contextual indexing strategy.

Context generators must honor cancellation and treat source/chunk metadata as immutable. On failure, Split cancels siblings, stops new work and waits for all started cooperative callbacks. A host function ignoring context can delay the return; no forced termination is promised.

Executable examples and checks: [source-to-resolution integration](../documents/chunking_integration_test.go), [actual PDF integration](../adapters/pdf/pipeline_test.go), [coordinate properties](ranges_fuzz_test.go), and [callback barriers](concurrency_test.go).
