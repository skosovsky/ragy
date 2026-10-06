package layout_test

import (
	"math"
	"testing"

	"github.com/skosovsky/ragy/layout"
	"github.com/skosovsky/ragy/source"
)

func validDocument() layout.Document {
	reference := source.Reference{
		Namespace:         "docs",
		Source:            "manual",
		Revision:          "r1",
		Transformation:    "layout",
		AccessFingerprint: "acl",
		Artifact:          "pdf",
		Representation:    "pdf-binary",
	}
	pageRef := reference
	pageRef.Artifact, pageRef.Representation = "p0/text", "normalized-page-text"
	return layout.Document{
		Schema:    layout.Schema,
		Reference: reference,
		PageCount: 1,
		Coverage:  layout.Complete,
		Pages: []layout.Page{
			{
				Reference: pageRef,
				Geometry:  source.PageGeometry{PhysicalIndex: 0, PrintedLabel: "i", Width: 600, Height: 800},
				Text:      "Alpha beta. Gamma.",
				Coverage:  layout.Complete,
				Words: []layout.Word{
					{
						ID:     "beta",
						Span:   source.ByteSpan{Start: 6, End: 10},
						Region: source.Rectangle{Left: 60, Top: 80, Right: 100, Bottom: 100},
					},
				},
			},
		},
	}
}

func TestLayoutRejectsInvalidGeometryRepresentationAndCoverage(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*layout.Document)
	}{
		{"schema", func(d *layout.Document) { d.Schema = "unknown" }},
		{"page count", func(d *layout.Document) { d.PageCount = 0 }},
		{"missing page promoted", func(d *layout.Document) { d.PageCount = 2 }},
		{"out of bounds page", func(d *layout.Document) { d.Pages[0].Geometry.PhysicalIndex = 1 }},
		{"wrong revision", func(d *layout.Document) { d.Pages[0].Reference.Revision = "r2" }},
		{"wrong access", func(d *layout.Document) { d.Pages[0].Reference.AccessFingerprint = "other" }},
		{"nonfinite geometry", func(d *layout.Document) { d.Pages[0].Geometry.Width = math.NaN() }},
		{"unsupported rotation", func(d *layout.Document) { d.Pages[0].Geometry.Rotation = 45 }},
		{"unknown coverage", func(d *layout.Document) { d.Coverage = "guessed" }},
		{"partial undiagnosed", func(d *layout.Document) { d.Pages[0].Coverage = layout.Partial }},
		{"OCR promoted", func(d *layout.Document) {
			d.Pages[0].Diagnostics = []layout.Diagnostic{{Code: "ocr_region_unreadable"}}
		}},
		{"limit promoted", func(d *layout.Document) { d.Diagnostics = []layout.Diagnostic{{Code: "page_limit"}} }},
		{"invalid span", func(d *layout.Document) { d.Pages[0].Words[0].Span.End = 100 }},
		{"inside codepoint", func(d *layout.Document) {
			d.Pages[0].Text = "АБВ"
			d.Pages[0].Words[0].Span = source.ByteSpan{Start: 1, End: 4}
		}},
		{"invalid bbox", func(d *layout.Document) { d.Pages[0].Words[0].Region.Bottom = 900 }},
		{
			"duplicate word",
			func(d *layout.Document) { d.Pages[0].Words = append(d.Pages[0].Words, d.Pages[0].Words[0]) },
		},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange.
			document := validDocument()
			tt.mutate(&document)
			// Act.
			err := document.Validate()
			// Assert.
			if err == nil {
				t.Fatal("invalid envelope admitted")
			}
		})
	}
	if err := validDocument().Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestLayoutKeepsPartialCoverageAndValidatesMergedCells(t *testing.T) {
	// Arrange.
	document := validDocument()
	page := &document.Pages[0]
	ref := page.Reference
	ref.Artifact, ref.Representation = "t0/c0", "original-cell-text"
	location := source.Locator{Reference: ref, Kind: source.CellLocation, Page: page.Geometry,
		Cell: source.TableCell{Table: "t0", Element: "c0", Row: 0, Column: 0, RowSpan: 1, ColumnSpan: 2},
	}
	page.Cells = []layout.Cell{
		{Location: location, Text: "Revenue", Region: source.Rectangle{Left: 60, Top: 110, Right: 260, Bottom: 150}},
	}
	page.Coverage = layout.Partial
	page.Diagnostics = []layout.Diagnostic{{Code: "ocr_region_unreadable", Element: "img0"}}
	document.Coverage = layout.Partial
	// Act.
	err := document.Validate()
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	duplicate := document
	duplicate.Pages = append([]layout.Page(nil), document.Pages...)
	duplicate.Pages[0].Cells = append(append([]layout.Cell(nil), page.Cells...), page.Cells[0])
	duplicate.Pages[0].Cells[1].Location.Reference.Artifact = "different-artifact-same-logical-cell"
	if validationErr := duplicate.Validate(); validationErr == nil {
		t.Fatal("merged cell became duplicate sources")
	}
	promoted := document
	promoted.Coverage = layout.Complete
	if validationErr := promoted.Validate(); validationErr == nil {
		t.Fatal("partial page promoted to complete document")
	}
}
