//go:build integration

package pdf_test

import (
	"context"
	"errors"
	"os/exec"
	"testing"

	ragy "github.com/skosovsky/ragy"
	pdfadapter "github.com/skosovsky/ragy/adapters/pdf"
	"github.com/skosovsky/ragy/layout"
	"github.com/skosovsky/ragy/source"
)

func TestIntegrationPDFLayoutParser(t *testing.T) {
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

func TestIntegrationPDFParserPageLimitPreservesPartialCoverage(t *testing.T) {
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

func TestIntegrationPDFParserRejectsBadPDFAndElementLimits(t *testing.T) {
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

func TestIntegrationPDFParserRejectsUnsupportedNativeGeometry(t *testing.T) {
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
