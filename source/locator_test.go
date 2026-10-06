package source_test

import (
	"math"
	"strconv"
	"testing"

	"github.com/skosovsky/ragy/source"
)

func retainedReference() source.Reference {
	return source.Reference{
		Namespace:         "docs",
		Source:            "manual",
		Revision:          "r1",
		Transformation:    "parser-normalized",
		AccessFingerprint: "acl1",
		Artifact:          "s1",
		Representation:    "page-text/0",
	}
}

func TestByteSpanUsesUTF8BytesAndCodepointBoundaries(t *testing.T) {
	tests := []struct {
		name, text string
		start, end int
		want       string
		valid      bool
	}{
		{"ascii", "Alpha beta. Gamma.", 6, 10, "beta", true},
		{"cyrillic", "АБВ", 2, 4, "Б", true},
		{"inside start", "АБВ", 1, 4, "", false},
		{"inside end", "АБВ", 2, 3, "", false},
		{"negative", "АБВ", -1, 4, "", false},
		{"outside", "АБВ", 2, 7, "", false},
		{"empty", "АБВ", 2, 2, "", false},
		{"invalid utf8", string([]byte{0xff}), 0, 1, "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange.
			span := source.ByteSpan{Start: tt.start, End: tt.end}
			// Act.
			err := span.ValidateText(tt.text)
			// Assert.
			if (err == nil) != tt.valid {
				t.Fatalf("valid=%v error=%v", tt.valid, err)
			}
			if err == nil && tt.text[span.Start:span.End] != tt.want {
				t.Fatal("wrong retained span")
			}
		})
	}
}

func TestPageRotationPreservesPhysicalIdentity(t *testing.T) {
	for _, rotation := range []int{0, 90, 180, 270} {
		t.Run(strconv.Itoa(rotation), func(t *testing.T) {
			// Arrange.
			page := source.PageGeometry{
				PhysicalIndex: 1,
				PrintedLabel:  "1",
				Width:         600,
				Height:        800,
				Rotation:      rotation,
			}
			original := source.Point{X: 60, Y: 80}
			// Act.
			display, err := page.Display(original)
			if err != nil {
				t.Fatal(err)
			}
			roundtrip, err := page.Original(display)
			// Assert.
			if err != nil || roundtrip != original {
				t.Fatalf("roundtrip=%v error=%v", roundtrip, err)
			}
			if rotation == 90 && display != (source.Point{X: 720, Y: 60}) {
				t.Fatalf("clockwise rotation=%v", display)
			}
			if page.PhysicalIndex != 1 || page.PrintedLabel != "1" {
				t.Fatal("page identity changed")
			}
		})
	}
}

func TestLocatorRejectsAmbiguousTagsAndInvalidGeometry(t *testing.T) {
	valid := source.Locator{
		Reference: retainedReference(),
		Kind:      source.TextLocation,
		Span:      source.ByteSpan{Start: 6, End: 10},
	}
	bad := valid
	bad.Page = source.PageGeometry{PhysicalIndex: 0, Width: 600, Height: 800}
	unknown := valid
	unknown.Kind = "unsupported"
	missing := valid
	missing.Reference.Revision = ""
	for _, locator := range []source.Locator{bad, unknown, missing} {
		if err := locator.Validate(); err == nil {
			t.Fatal("invalid locator admitted")
		}
	}
	page := source.PageGeometry{PhysicalIndex: 0, PrintedLabel: "i", Width: 600, Height: 800, Rotation: 0}
	for _, mutate := range []func(*source.PageGeometry){
		func(p *source.PageGeometry) { p.Width = math.NaN() },
		func(p *source.PageGeometry) { p.Height = math.Inf(1) },
		func(p *source.PageGeometry) { p.Rotation = 45 },
		func(p *source.PageGeometry) { p.PhysicalIndex = -1 },
	} {
		invalid := page
		mutate(&invalid)
		if err := invalid.Validate(); err == nil {
			t.Fatal("invalid page admitted")
		}
	}
	for _, rect := range []source.Rectangle{
		{Left: 100, Top: 200, Right: 300, Bottom: 900},
		{Left: 100, Top: 200, Right: 100, Bottom: 400},
		{Left: math.NaN(), Top: 200, Right: 300, Bottom: 400},
	} {
		if err := rect.Validate(page); err == nil {
			t.Fatal("invalid region admitted")
		}
	}
}

func TestCitationIdentityBindsRevisionRepresentationAndLogicalCell(t *testing.T) {
	// Arrange.
	cell := source.Locator{Reference: retainedReference(), Kind: source.CellLocation,
		Page: source.PageGeometry{PhysicalIndex: 0, PrintedLabel: "i", Width: 600, Height: 800},
		Cell: source.TableCell{Table: "t1", Element: "c1", Row: 0, Column: 0, RowSpan: 1, ColumnSpan: 2},
	}
	// Act.
	first, err := cell.Identity()
	if err != nil {
		t.Fatal(err)
	}
	same, err := cell.Identity()
	if err != nil {
		t.Fatal(err)
	}
	nextRevision := cell
	nextRevision.Reference.Revision = "r2"
	next, err := nextRevision.Identity()
	if err != nil {
		t.Fatal(err)
	}
	representation := cell
	representation.Reference.Representation = "derived-description"
	derived, err := representation.Identity()
	if err != nil {
		t.Fatal(err)
	}
	// Assert.
	if first != same || first == next || first == derived {
		t.Fatal("citation identity conflates retained locations")
	}
	invalid := cell
	invalid.Cell.ColumnSpan = 0
	if validationErr := invalid.Validate(); validationErr == nil {
		t.Fatal("invalid merged cell admitted")
	}
	text := source.Locator{
		Reference: retainedReference(),
		Kind:      source.TextLocation,
		Span:      source.ByteSpan{Start: 6, End: 10},
	}
	other := text
	other.Span = source.ByteSpan{Start: 12, End: 17}
	textID, err := text.Identity()
	if err != nil {
		t.Fatal(err)
	}
	otherID, err := other.Identity()
	if err != nil {
		t.Fatal(err)
	}
	if textID == otherID {
		t.Fatal("distinct spans share identity")
	}
}

func TestEqualGeometryHasEqualCitationIdentity(t *testing.T) {
	// Arrange.
	first := source.Locator{Reference: retainedReference(), Kind: source.RegionLocation,
		Page:   source.PageGeometry{PhysicalIndex: 0, Width: 600, Height: 800},
		Region: source.Rectangle{Left: 0, Top: 0, Right: 100, Bottom: 100},
	}
	same := first
	same.Region.Left = math.Copysign(0, -1)
	// Act.
	firstID, err := first.Identity()
	if err != nil {
		t.Fatal(err)
	}
	sameID, err := same.Identity()
	if err != nil {
		t.Fatal(err)
	}
	// Assert.
	if first != same || firstID != sameID {
		t.Fatal("equal geometry has distinct citations")
	}
}
