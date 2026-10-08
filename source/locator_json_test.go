package source_test

import (
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/source"
)

func TestLocatorEnvelopeRoundTripEveryRepresentation(t *testing.T) {
	ref := retainedReference()
	page := source.PageGeometry{PhysicalIndex: 1, PrintedLabel: "1", Width: 600, Height: 800, Rotation: 90}
	region := source.Rectangle{Left: 100, Top: 200, Right: 300, Bottom: 400}
	locations := []source.Locator{
		{Reference: ref, Kind: source.DocumentLocation},
		{Reference: ref, Kind: source.TextLocation, Span: source.ByteSpan{Start: 6, End: 10}},
		{Reference: ref, Kind: source.PageLocation, Page: page},
		{Reference: ref, Kind: source.RegionLocation, Page: page, Region: region},
		{
			Reference: ref,
			Kind:      source.CellLocation,
			Page:      page,
			Cell:      source.TableCell{Table: "t1", Element: "c1", RowSpan: 1, ColumnSpan: 2},
		},
		{Reference: ref, Kind: source.ImageLocation, Page: page, Region: region},
	}
	for _, location := range locations {
		t.Run(string(location.Kind), func(t *testing.T) {
			// Arrange.
			identity, err := location.Identity()
			if err != nil {
				t.Fatal(err)
			}
			// Act.
			encoded, err := source.EncodeLocator(location)
			if err != nil {
				t.Fatal(err)
			}
			decoded, err := source.DecodeLocator(encoded)
			// Assert: persistence retains exact geometry/reference and canonical citation identity.
			if err != nil || decoded != location {
				t.Fatal(decoded, err)
			}
			restored, err := decoded.Identity()
			if err != nil || restored != identity {
				t.Fatal(restored, err)
			}
		})
	}
}
func TestLocatorEnvelopeRejectsAmbiguousWireAndIncompatibleSchema(t *testing.T) {
	// Arrange: the same positive fixture is independently JSON Schema validated.
	encoded, err := os.ReadFile("testdata/locator.json")
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := source.DecodeLocator(encoded)
	if err != nil || decoded.Kind != source.TextLocation {
		t.Fatal(decoded, err)
	}
	text := string(encoded)
	cases := map[string]string{
		"unknown":             strings.Replace(text, `"schema":`, `"unknown":0,"schema":`, 1),
		"duplicate-schema":    strings.Replace(text, `"schema":`, `"schema":"ragy.source-locator","schema":`, 1),
		"duplicate-reference": strings.Replace(text, `"revision":`, `"revision":"r2","revision":`, 1),
		"case-alias":          strings.Replace(text, `"page":`, `"Page":`, 1),
		"null-inactive":       strings.Replace(text, `"printed_label": ""`, `"printed_label":null`, 1),
		"missing-inactive":    strings.Replace(text, `"physical_index": 0,`, "", 1),
		"future-schema":       strings.Replace(text, source.LocatorSchema, "unknown-schema", 1),
		"trailing":            text + `{}`,
		"array":               `[]`,
		"invalid-utf8":        text + string([]byte{0xff}),
	}
	for name, value := range cases {
		t.Run(name, func(t *testing.T) {
			// Act.
			value, decodeErr := source.DecodeLocator([]byte(value))
			// Assert: invalid input never yields a usable locator.
			if decodeErr == nil || value != (source.Locator{}) {
				t.Fatal(value, decodeErr)
			}
			if name == "future-schema" && !errors.Is(decodeErr, ragy.ErrUnsupported) {
				t.Fatal(decodeErr)
			}
		})
	}
	// Encode also refuses invalid union geometry.
	decoded.Page.Width = 600
	if _, err = source.EncodeLocator(decoded); err == nil {
		t.Fatal("invalid union encoded")
	}
}

func TestTypedHostLocatorExtensionKeepsCanonicalIdentity(t *testing.T) {
	type highlight struct {
		Color string `json:"color"`
		Layer int    `json:"layer"`
	}
	// Arrange: a typed host extension, separate from the canonical retained location.
	location := source.Locator{
		Reference: retainedReference(),
		Kind:      source.TextLocation,
		Span:      source.ByteSpan{Start: 6, End: 10},
	}
	original, err := location.Identity()
	if err != nil {
		t.Fatal(err)
	}
	host := source.ExtendedLocator[highlight]{Location: location, Extension: highlight{Color: "yellow", Layer: 2}}
	// Act: host owns extension persistence, validation and copy semantics.
	encoded, err := json.Marshal(host)
	if err != nil {
		t.Fatal(err)
	}
	var restored source.ExtendedLocator[highlight]
	if err = json.Unmarshal(encoded, &restored); err != nil {
		t.Fatal(err)
	}
	restored.Extension.Color = "blue"
	identity, err := restored.Location.Identity()
	// Assert: host changes cannot alter canonical revision/location identity.
	if err != nil || identity != original || host.Extension.Color != "yellow" {
		t.Fatal(restored, err)
	}
}
