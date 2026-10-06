package layout_test

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/layout"
	"github.com/skosovsky/ragy/source"
)

func twoPageDocument(first, second string, duplicate bool) layout.Document {
	document := validDocument()
	document.PageCount = 2
	page := document.Pages[0]
	page.Text, page.Words = first, nil
	document.Pages[0] = page
	page.Geometry.PhysicalIndex, page.Geometry.PrintedLabel, page.Text = 1, "ii", second
	if !duplicate {
		page.Reference.Artifact = "p1/text"
	}
	document.Pages = append(document.Pages, page)
	return document
}

func TestDuplicatePageTextReferencesRejectBeforeProjectionCallback(t *testing.T) {
	for _, texts := range [][2]string{{"AAA", "BBB"}, {"AAA", "AAA"}, {"A", "longer text"}, {"", ""}, {"", "nonempty"}} {
		for _, partial := range []bool{false, true} {
			t.Run(fmt.Sprintf("%q/%q/partial=%t", texts[0], texts[1], partial), func(t *testing.T) {
				// Arrange: ordered physical pages, but the exact retained text address is reused.
				document := twoPageDocument(texts[0], texts[1], true)
				image := source.Locator{
					Reference: document.Reference,
					Kind:      source.ImageLocation,
					Page:      document.Pages[0].Geometry,
					Region:    source.Rectangle{Left: 10, Top: 10, Right: 30, Bottom: 30},
				}
				document.Pages[0].Images = []layout.Image{{Location: image}}
				if partial {
					document.Coverage, document.Pages[1].Coverage = layout.Partial, layout.Partial
					document.Pages[1].Diagnostics = []layout.Diagnostic{{Code: "unsupported_grid"}}
				}
				calls := 0
				options := layout.ProjectionOptions{
					Read: access.Unrestricted(),
					ImageText: func(context.Context, access.Binding, layout.Image) (source.MappedText, error) {
						calls++
						return source.DerivedText("description", []source.Locator{image})
					},
				}
				// Act.
				validationErr := document.Validate()
				projected, err := layout.Project(context.Background(), document, options)
				// Assert: shape rejection applies even if no page text would be projected.
				if !errors.Is(validationErr, ragy.ErrInvalidArgument) || !errors.Is(err, ragy.ErrInvalidArgument) ||
					projected != nil ||
					calls != 0 {
					t.Fatal(validationErr, projected, err, calls)
				}
			})
		}
	}
}

func TestDistinctPageReferencesResolveIndependentExactCitations(t *testing.T) {
	// Arrange: retained pages with equal lengths but different immutable text.
	document := twoPageDocument("AAA", "BBB", false)
	fixture := newResolveFixture(t)
	for _, page := range document.Pages {
		original := source.Locator{Reference: page.Reference, Kind: source.PageLocation, Page: page.Geometry}
		fixture.host.records[page.Reference] = layout.Retained{
			Original: original,
			Text:     page.Text,
			Coverage: layout.Complete,
		}
		fixture.host.permissions[page.Reference] = retainedPermission{organization: "a"}
	}
	// Act: projection citations become admitted exact loader requests.
	projected, err := layout.Project(context.Background(), document, layout.ProjectionOptions{Read: fixture.read})
	if err != nil || len(projected) != 2 {
		t.Fatal(projected, err)
	}
	resolved, err := fixture.reader.Resolve(context.Background(), layout.ResolveRequest{
		Read: fixture.read, Locations: []source.Locator{projected[0].Location, projected[1].Location},
	})
	// Assert: no dedup collision, swap, physical-page hash workaround or latest lookup.
	if err != nil || len(resolved) != 2 || projected[0].ID == projected[1].ID || fixture.host.loads != 1 {
		t.Fatal(resolved, err)
	}
	for i, citation := range resolved {
		if citation.Text.Text() != document.Pages[i].Text || citation.Location != projected[i].Location ||
			citation.Text.Fragments()[0].Location != projected[i].Location {
			t.Fatal(citation)
		}
	}
}

func sharedSelectorsDocument() layout.Document {
	document := twoPageDocument("AAA", "BBB", false)
	for index := range document.Pages {
		page := &document.Pages[index]
		for selector := range 2 {
			region := source.Rectangle{
				Left:   float64(10 + selector*30),
				Top:    10,
				Right:  float64(30 + selector*30),
				Bottom: 30,
			}
			cell := source.Locator{
				Reference: document.Reference,
				Kind:      source.CellLocation,
				Page:      page.Geometry,
				Cell: source.TableCell{
					Table:      "table",
					Element:    strconv.Itoa(selector),
					Row:        0,
					Column:     selector,
					RowSpan:    1,
					ColumnSpan: 1,
				},
			}
			page.Cells = append(
				page.Cells,
				layout.Cell{Location: cell, Region: region, Text: fmt.Sprintf("cell-%d-%d", index, selector)},
			)
			image := source.Locator{
				Reference: document.Reference,
				Kind:      source.ImageLocation,
				Page:      page.Geometry,
				Region:    region,
			}
			page.Images = append(page.Images, layout.Image{Location: image})
		}
	}
	return document
}

func TestWholeArtifactCellImageSelectorsRemainDistinct(t *testing.T) {
	// Arrange: same original artifact, complete physical-page + cell/region selectors.
	document := sharedSelectorsDocument()
	calls := 0
	options := layout.ProjectionOptions{
		Read: access.Unrestricted(),
		ImageText: func(_ context.Context, _ access.Binding, image layout.Image) (source.MappedText, error) {
			calls++
			return source.DerivedText("image observation", []source.Locator{image.Location})
		},
	}
	// Act.
	validationErr := document.Validate()
	projected, err := layout.Project(context.Background(), document, options)
	// Assert: selector identities remain distinct within and across pages.
	if validationErr != nil || err != nil || len(projected) != 10 || calls != 4 {
		t.Fatal(validationErr, projected, err, calls)
	}
	seen := make(map[string]bool)
	for _, item := range projected {
		if seen[item.ID] || item.Text.Supports()[0] != item.Location {
			t.Fatal("selector collision or lost source", item)
		}
		seen[item.ID] = true
	}
}

func TestDuplicateOriginalSelectorsStillRejected(t *testing.T) {
	for _, kind := range []string{"cell", "image"} {
		t.Run(kind, func(t *testing.T) {
			// Arrange: duplicate the complete selector within a page of a lawful sharing document.
			document := sharedSelectorsDocument()
			if kind == "cell" {
				document.Pages[0].Cells = append(document.Pages[0].Cells, document.Pages[0].Cells[0])
			} else {
				document.Pages[0].Images = append(document.Pages[0].Images, document.Pages[0].Images[0])
			}
			calls := 0
			// Act.
			projected, err := layout.Project(
				context.Background(),
				document,
				layout.ProjectionOptions{
					Read: access.Unrestricted(),
					ImageText: func(context.Context, access.Binding, layout.Image) (source.MappedText, error) {
						calls++
						return source.MappedText{}, nil
					},
				},
			)
			// Assert: sharing does not legalize duplicates or invoke derived consumers.
			if !errors.Is(document.Validate(), ragy.ErrInvalidArgument) || !errors.Is(err, ragy.ErrInvalidArgument) ||
				projected != nil ||
				calls != 0 {
				t.Fatal(projected, err, calls)
			}
		})
	}
}
