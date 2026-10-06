# Optional PDF/layout parser

This adapter invokes an external Python interpreter with `pdfplumber` and `pypdf`.
The Go core depends on neither engine. Install those optional dependencies in the
interpreter supplied by host configuration.

```go
parser, err := pdf.New(pdf.Config{
    Python: pythonExecutable,
    Transformation: transformationFingerprint,
    Limits: pdf.Limits{
        InputBytes: 10 << 20,
        OutputBytes: 10 << 20,
        Pages: 100,
        Words: 10000,
        Cells: 1000,
        Images: 1000,
        Timeout: 5 * time.Second,
    },
})
// sourceReference identifies the authorized retained PDF bytes, uses
// Representation="pdf-binary" and the same Transformation as the configuration.
parsed, err := parser.Parse(ctx, layout.Input{
    Reference: sourceReference,
    Data: authorizedRetainedPDFBytes,
})
```

Host owns authentication, source authorization/materialization, retention and the
truthful association between bytes and revision. This parser accepts authorized
bytes for ingestion; it is not a raw URL/path lookup or a scoped retrieval backend.
The host transformation fingerprint must include the selected engine/configuration,
normalization algorithm and relevant dependency identities. Changing those without
changing the fingerprint is a host contract violation.

Output has schema identity `ragy.layout` and is validated before delivery. Words
are joined by single spaces in `normalized-page-text`, with half-open UTF-8 byte
spans. Those offsets never address PDF binary. Physical page indices and printed
labels are independent. Geometry is unrotated top-left page points; native rotated
rectangles are converted back while clockwise rotation remains explicit. Logical
merged cells occur once with row/column spans. Image regions refer to original
representations and have no invented descriptions or OCR text.

This profile supports MediaBox origin (0,0), no differing CropBox, rotations
0/90/180/270 and unrotated table grids. Unsupported geometry/grids fail explicitly.
A page limit keeps actual PageCount and partial coverage. Images without OCR yield
`ocr_unprocessed` and partial page/document coverage; empty text is not successful
OCR. OCR accuracy, image descriptions and geometry highlighting are separate host
or optional adapter responsibilities. No retry or background worker is started.
Caller context/earlier deadline bounds the external process. Input/output bytes and
per-page elements have explicit limits; error returns contain no partial document
or source/parser exception text.

Run the actual parser integration, not only deterministic unit tests:

```sh
RAGY_PDF_PYTHON=/path/to/python-with-dependencies go test -race -v ./...
```

Without that variable, real-engine tests explicitly skip. Unit tests still run;
skipped integration must not be counted as accepted capability. `testdata` contains
a deterministic two-page PDF/PNG, its generator and an independent fixture verifier.
The integration covers text/table/image/rotation, rejected geometry, malformed input,
limits/deadline and a real parse/chunk/project/BM25/scoped-resolve chain. The source
store in that chain is a host reference implementation with r1/r2, deleted and
permission-denied scenarios; it does not establish persistent storage or lifecycle.

Typed retained geometry/cell/image source resolution and explicit OCR observation
simulation are supplied by layout. The adapter module also tests actual cell/image
text projection, BM25 retrieval and artifact rendering with original/derived origin
and partial coverage. The durable integration described below verifies source
mappings/supports and typed coverage through managed indexing/publication. Parser
coverage and generic lifecycle inventory coverage retain their distinct meanings.

The adapter tests also run actual PDF/layout projection through persistent dense
indexing and Executor/filestore publication, then reopen both index and ledger.
Partial coverage and original/derived mappings survive r1/r2 publication; pinned r1
citations resolve through an explicitly retained host source. Deleted or denied r1
fails before payload loading even while r2 exists. This test establishes durable
index/manifest integration; its original layout/blob retention host is in memory.
