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

## Split grammar, coverage and admission

Recursive overlap is best-effort and only applies to fixed-size fallback after
separators fail. Separator-group boundaries have no added overlap. Whitespace
trimming may shorten fallback overlap. With size=8, overlap=3, “one two three four
five six” separates at words without shared text; “abcdefghij” has fallback
ranges [0,8), [5,10). Original InputSpan always describes the actual bytes.

Custom separator bytes remain inside a merged fragment but are omitted when a
separator becomes a split boundary; repeated boundary delimiters are omitted too.
For size=2 and separator “|”, “ab||cd” yields “ab”, “cd”. Select separators whose
omission is acceptable; exact coordinates alone do not prove full information
coverage. Markdown is a trimmed hash-line heuristic (`#anything`), including
hash lines inside fenced code, with no Setext parser. Default sentence segmentation
is punctuation-based, not abbreviation/decimal-aware NLP. Bring your own segmenter
when these grammars are unsuitable; no model/provider fallback is installed.

SentenceSegmenter ranges must be ordered, nonoverlapping, nonempty and on UTF-8
boundaries. Prefix/tail/inter-range omissions may contain whitespace only.
Substantive dropped bytes are ErrProtocol before embedding. Semantic groups retain
original bytes within their first/last boundaries; they do not invent source text.

ValidateChunk checks standalone UTF-8 Content/Context, identity, index/Total and
mapping/support shape. Total0 means unknown. Zero InputSpan means absent; absolute
source bytes require the original input to verify. Contextual validates the whole
base batch before any generator: ordered indices 0..N-1, matching SourceID, Total0
or Total=N, standalone shape, and exact supplied nonzero InputSpan content. Invalid
batches are ErrProtocol, never rescued by sorting. Generated context must be valid
UTF-8. Failure returns no chunks and cancels/joins cooperative started siblings.

## Projection identity and ownership

ProjectDocuments is all-or-nothing for identity, metadata, index-text or document
errors. Callback work already performed is not undone; projection writes no index.
Default identity rejects conflicting descriptor/chunk source IDs. Chunk ID is the
document ID; missing ID uses sourceID_index. Descriptor URI supplies URI#chunk=index
MergeKey; StorageID is shared source storage identity, not a unique chunk primary
key. Host must version source IDs/URIs across revisions/rechunking or supply an
explicit IdentityPolicy. A custom policy may deliberately override the default.

Source/chunk Meta and projected Meta are borrowed host values unless a callback
explicitly clones them. Contextual callbacks concurrently read immutable metadata;
no JSON clone is silently added to BYOT. SourceMapping owns its internal supports;
original quote Content stays separate from synthetic Context/IndexText.

[Current retained-source integration guide](../source/README.md) links runnable
source→layout→chunk→index→authorized Resolve profiles and their exact boundaries.
