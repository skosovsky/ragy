package pdf_test

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"runtime"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	pdfadapter "github.com/skosovsky/ragy/adapters/pdf"
	"github.com/skosovsky/ragy/layout"
	"github.com/skosovsky/ragy/source"
)

func fixtureReference() source.Reference {
	return source.Reference{
		Namespace:         "manuals",
		Source:            "manual",
		Revision:          "r1",
		Transformation:    "pdf/layout",
		AccessFingerprint: "public-acl",
		Artifact:          "pdf",
		Representation:    "pdf-binary",
	}
}
func parserConfig(python string) pdfadapter.Config {
	return pdfadapter.Config{Python: python, Transformation: "pdf/layout", Limits: pdfadapter.Limits{
		InputBytes:  1 << 20,
		OutputBytes: 1 << 20,
		Pages:       10,
		Words:       100,
		Cells:       100,
		Images:      10,
		Timeout:     5 * time.Second,
	}}
}
func integrationPython(t *testing.T) string {
	t.Helper()
	python := os.Getenv("RAGY_PDF_PYTHON")
	if python == "" {
		t.Skip("set RAGY_PDF_PYTHON to run the optional actual-parser integration")
	}
	return python
}
func loadFixture(t *testing.T) []byte {
	t.Helper()
	data, err := os.ReadFile("testdata/manual.pdf")
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestActualPDFLayoutParser(t *testing.T) {
	// Arrange.
	parser, err := pdfadapter.New(parserConfig(integrationPython(t)))
	if err != nil {
		t.Fatal(err)
	}
	input := layout.Input{Reference: fixtureReference(), Data: loadFixture(t)}
	// Act.
	document, err := parser.Parse(context.Background(), input)
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	if document.PageCount != 2 || len(document.Pages) != 2 || document.Coverage != layout.Partial {
		t.Fatalf("coverage/page count %+v", document)
	}
	first, second := document.Pages[0], document.Pages[1]
	if first.Text != "Alpha beta. Gamma. Revenue 2023 2024" || first.Geometry.PrintedLabel != "i" ||
		second.Geometry.PrintedLabel != "1" {
		t.Fatal("normalized representation or labels changed")
	}
	if first.Geometry.PhysicalIndex != 0 || second.Geometry.PhysicalIndex != 1 || second.Geometry.Rotation != 90 ||
		second.Geometry.Width != 600 ||
		second.Geometry.Height != 800 {
		t.Fatal("physical page geometry changed")
	}
	if first.Words[1].Span != (source.ByteSpan{Start: 6, End: 11}) || first.Text[6:10] != "beta" {
		t.Fatal("UTF-8 offsets address a different representation")
	}
	if len(first.Cells) != 3 || first.Cells[0].Text != "Revenue" || first.Cells[0].Location.Cell.ColumnSpan != 2 {
		t.Fatalf("merged cell duplicated/lost %+v", first.Cells)
	}
	region := source.Rectangle{Left: 100, Top: 200, Right: 300, Bottom: 400}
	if len(first.Images) != 1 || len(second.Images) != 1 || first.Images[0].Location.Region != region ||
		second.Images[0].Location.Region != region {
		t.Fatal("native rotated rectangles not normalized to original coordinates")
	}
	if second.Coverage != layout.Partial || second.Text != "" || second.Diagnostics[0].Code != "ocr_unprocessed" {
		t.Fatal("missing OCR fabricated successful content")
	}
	for _, page := range document.Pages {
		if page.Reference.Revision != "r1" || page.Reference.Transformation != input.Reference.Transformation ||
			page.Reference.Representation != "normalized-page-text" {
			t.Fatal("source identity lost")
		}
	}
}

func TestActualPDFParserPageLimitPreservesPartialCoverage(t *testing.T) {
	// Arrange.
	config := parserConfig(integrationPython(t))
	config.Limits.Pages = 1
	parser, err := pdfadapter.New(config)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	document, err := parser.Parse(
		context.Background(),
		layout.Input{Reference: fixtureReference(), Data: loadFixture(t)},
	)
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	if document.PageCount != 2 || len(document.Pages) != 1 || document.Coverage != layout.Partial ||
		document.Diagnostics[0].Code != "page_limit" {
		t.Fatal("limited parse promoted to complete document")
	}
}

func TestActualPDFParserRejectsBadPDFAndElementLimits(t *testing.T) {
	python := integrationPython(t)
	for _, kind := range []string{"invalid PDF", "word limit", "output limit"} {
		t.Run(kind, func(t *testing.T) {
			// Arrange.
			config := parserConfig(python)
			data := loadFixture(t)
			switch kind {
			case "invalid PDF":
				data = []byte("not a PDF")
			case "word limit":
				config.Limits.Words = 1
			case "output limit":
				config.Limits.OutputBytes = 1
			}
			parser, err := pdfadapter.New(config)
			if err != nil {
				t.Fatal(err)
			}
			// Act.
			document, err := parser.Parse(context.Background(), layout.Input{Reference: fixtureReference(), Data: data})
			// Assert.
			if err == nil || len(document.Pages) != 0 || document.Reference.Source != "" {
				t.Fatal("failed parse delivered partial payload")
			}
		})
	}
}

func TestParserRejectsInvalidInputAndCancellationBeforeExternalExecution(t *testing.T) {
	// Arrange: non-existent executable proves rejected input does not dispatch.
	parser, err := pdfadapter.New(parserConfig("/nonexistent/ragy-python"))
	if err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"empty", "wrong representation", "wrong transformation", "oversize", "cancelled"} {
		t.Run(kind, func(t *testing.T) {
			input := layout.Input{Reference: fixtureReference(), Data: []byte("pdf bytes")}
			ctx := context.Background()
			switch kind {
			case "empty":
				input.Data = nil
			case "wrong representation":
				input.Reference.Representation = "normalized-page-text"
			case "wrong transformation":
				input.Reference.Transformation = "other"
			case "oversize":
				input.Data = make([]byte, 1<<20+1)
			case "cancelled":
				cancelled, cancel := context.WithCancel(ctx)
				cancel()
				ctx = cancelled
			}
			// Act.
			document, err := parser.Parse(ctx, input)
			// Assert.
			if err == nil || errors.Is(err, ragy.ErrUnavailable) || len(document.Pages) != 0 {
				t.Fatal("invalid input dispatched or returned payload")
			}
		})
	}
}

func TestParserPreservesEarlierDeadlineAndDoesNotExportEngineErrors(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test process uses POSIX sh; actual Python adapter remains portable")
	}
	// Arrange: a process with no child worker waits until caller cancellation.
	executable := t.TempDir() + "/parser"
	script := []byte("#!/bin/sh\nprintf 'private-source-sentinel' >&2\nwhile :; do :; done\n")
	if err := os.WriteFile(executable, script, 0700); err != nil {
		t.Fatal(err)
	}
	parser, err := pdfadapter.New(parserConfig(executable))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	// Act.
	document, err := parser.Parse(
		ctx,
		layout.Input{Reference: fixtureReference(), Data: []byte("already-authorized bytes")},
	)
	// Assert.
	if !errors.Is(err, context.DeadlineExceeded) || len(document.Pages) != 0 || document.Reference.Source != "" {
		t.Fatal("deadline was replaced or payload exported")
	}
	if err.Error() != context.DeadlineExceeded.Error() {
		t.Fatal("external stderr leaked through error")
	}
}

func TestActualPDFParserRejectsUnsupportedNativeGeometry(t *testing.T) {
	python := integrationPython(t)
	for _, kind := range []string{"crop", "rotation", "rotated table"} {
		t.Run(kind, func(t *testing.T) {
			// Arrange: real PDF bytes whose native geometry is outside this declared profile.
			script := `import sys, io
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, NumberObject
writer=PdfWriter()
writer.clone_document_from_reader(PdfReader("testdata/manual.pdf"))
if sys.argv[1] == "crop": writer.pages[0].cropbox.upper_right=(500,700)
elif sys.argv[1] == "rotation": writer.pages[0][NameObject("/Rotate")]=NumberObject(45)
else: writer.pages[0].rotate(90)
memory=io.BytesIO()
writer.write(memory)
sys.stdout.buffer.write(memory.getvalue())
`
			command := exec.CommandContext(context.Background(), python, "-c", script, kind)
			bytes, err := command.Output()
			if err != nil {
				t.Fatal(err)
			}
			parser, err := pdfadapter.New(parserConfig(python))
			if err != nil {
				t.Fatal(err)
			}
			// Act.
			document, err := parser.Parse(
				context.Background(),
				layout.Input{Reference: fixtureReference(), Data: bytes},
			)
			// Assert.
			if !errors.Is(err, ragy.ErrUnsupported) || len(document.Pages) != 0 {
				t.Fatal("unsupported geometry was guessed or partially exported")
			}
		})
	}
}
