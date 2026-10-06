package layout_test

import (
	"context"
	"errors"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/layout"
	"github.com/skosovsky/ragy/source"
)

type retainedPermission struct{ organization string }
type layoutHost struct {
	records         map[source.Reference]layout.Retained
	permissions     map[source.Reference]retainedPermission
	loads           int
	revoked         bool
	revokeAfterLoad bool
}

func (h *layoutHost) Describe(
	_ context.Context,
	request source.LookupRequest,
) ([]source.Descriptor[retainedPermission], error) {
	var out []source.Descriptor[retainedPermission]
	for _, reference := range request.References {
		if permission, exists := h.permissions[reference]; exists {
			out = append(out, source.Descriptor[retainedPermission]{Reference: reference, Access: permission})
		}
	}
	return out, nil
}

func (h *layoutHost) Load(
	_ context.Context,
	request source.LookupRequest,
) ([]source.Materialized[layout.Retained], error) {
	h.loads++
	var out []source.Materialized[layout.Retained]
	for _, reference := range request.References {
		if retained, exists := h.records[reference]; exists {
			out = append(out, source.Materialized[layout.Retained]{Reference: reference, Payload: retained})
		}
	}
	if h.revokeAfterLoad {
		h.revoked = true
	}
	return out, nil
}

type resolveFixture struct {
	reader                      *layout.Resolver[retainedPermission]
	host                        *layoutHost
	read                        access.Binding
	page, cell, image, document source.Locator
}

func newResolveFixture(t *testing.T) resolveFixture {
	t.Helper()
	root := validDocument()
	page := source.Locator{Reference: root.Pages[0].Reference, Kind: source.PageLocation, Page: root.Pages[0].Geometry}
	cell := page
	cell.Kind = source.CellLocation
	cell.Reference.Artifact, cell.Reference.Representation = "t0/c0", "original-cell-text"
	cell.Cell = source.TableCell{Table: "t0", Element: "c0", Row: 0, Column: 0, RowSpan: 1, ColumnSpan: 2}
	image := page
	image.Kind = source.ImageLocation
	image.Reference.Artifact, image.Reference.Representation = "img0", "original-image"
	image.Region = source.Rectangle{Left: 100, Top: 200, Right: 300, Bottom: 400}
	document := source.Locator{Reference: root.Reference, Kind: source.DocumentLocation}
	derived, err := source.DerivedText("fixture diagram", []source.Locator{image})
	if err != nil {
		t.Fatal(err)
	}
	host := &layoutHost{records: map[source.Reference]layout.Retained{
		page.Reference: {
			Original: page,
			Text:     root.Pages[0].Text,
			Words:    root.Pages[0].Words,
			Coverage: layout.Complete,
		},
		cell.Reference: {
			Original:   cell,
			Text:       "Revenue",
			CellRegion: source.Rectangle{Left: 60, Top: 110, Right: 260, Bottom: 150},
			Coverage:   layout.Complete,
		},
		image.Reference: {
			Original:    image,
			Bytes:       []byte{1, 2, 3},
			MediaType:   "image/png",
			Coverage:    layout.Partial,
			Diagnostics: []layout.Diagnostic{{Code: "ocr_region_unreadable", Element: "img0"}},
			Derived:     derived,
		},
		document.Reference: {
			Original:  document,
			Bytes:     []byte("original PDF"),
			MediaType: "application/pdf",
			Coverage:  layout.Complete,
		},
	}, permissions: make(map[source.Reference]retainedPermission)}
	for reference := range host.records {
		host.permissions[reference] = retainedPermission{organization: "a"}
	}
	fields := filter.NewSchema()
	organization, err := fields.String("organization")
	if err != nil {
		t.Fatal(err)
	}
	schema, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	builder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	mandatory, err := filter.Eq(builder, organization, "a").Build()
	if err != nil {
		t.Fatal(err)
	}
	now := time.Unix(100, 0)
	read, err := access.Scoped(
		access.ScopedConfig{
			Snapshot: access.Snapshot{
				Identity:    "policy",
				PolicyEpoch: 7,
				IssuedAt:    now,
				ExpiresAt:   now.Add(time.Minute),
			},
			Mandatory:   mandatory,
			Schema:      schema,
			Publication: access.CurrentPublication(),
			Now:         func() time.Time { return now },
			Authority: access.AuthorityFunc(func(context.Context, access.Snapshot) error {
				if host.revoked {
					return ragy.ErrUnavailable
				}
				return nil
			}),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	reader, err := layout.NewResolver(
		layout.ResolverConfig[retainedPermission]{
			Target:  "layout",
			Schema:  schema,
			Catalog: host,
			Loader:  host,
			Attributes: func(permission retainedPermission) (filter.RawAttributes, error) {
				return filter.RawAttributes{"organization": permission.organization}, nil
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return resolveFixture{
		reader:   reader,
		host:     host,
		read:     read,
		page:     page,
		cell:     cell,
		image:    image,
		document: document,
	}
}

func TestLayoutResolverReturnsOriginalTextCellImageAndDocument(t *testing.T) {
	// Arrange.
	fixture := newResolveFixture(t)
	text := source.Locator{
		Reference: fixture.page.Reference,
		Kind:      source.TextLocation,
		Span:      source.ByteSpan{Start: 6, End: 10},
	}
	region := fixture.page
	region.Kind = source.RegionLocation
	region.Region = source.Rectangle{Left: 60, Top: 80, Right: 100, Bottom: 100}
	request := layout.ResolveRequest{
		Read:      fixture.read,
		Locations: []source.Locator{text, fixture.cell, fixture.image, fixture.document, region, fixture.cell},
	}
	// Act.
	resolved, err := fixture.reader.Resolve(context.Background(), request)
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	if len(resolved) != 5 || resolved[0].Text.Text() != "beta" || resolved[1].CellText != "Revenue" {
		t.Fatal("text/cell or dedup failed")
	}
	if resolved[2].Text.Text() != "" || resolved[2].Derived.Text() != "fixture diagram" ||
		resolved[2].Coverage != layout.Partial {
		t.Fatal("description replaced original pixels or partial coverage")
	}
	if string(resolved[3].Bytes) != "original PDF" || resolved[4].Text.Text() != "beta" {
		t.Fatal("document or region projection failed")
	}
	if resolved[4].Text.Fragments()[0].Location.Span != text.Span {
		t.Fatal("region fabricated source offsets")
	}
	resolved[2].Bytes[0] = 99
	resolved[2].Diagnostics[0].Code = "changed"
	if fixture.host.records[fixture.image.Reference].Bytes[0] != 1 ||
		fixture.host.records[fixture.image.Reference].Diagnostics[0].Code != "ocr_region_unreadable" {
		t.Fatal("resolved output mutated retained source")
	}
	if fixture.host.loads != 1 {
		t.Fatal("batch materialization repeated")
	}
}

func TestLayoutResolverRejectsDeletedDeniedGeometryAndRevocation(t *testing.T) {
	for _, kind := range []string{"deleted", "denied", "revoked", "wrong page", "outside image", "wrong cell"} {
		t.Run(kind, func(t *testing.T) {
			// Arrange.
			fixture := newResolveFixture(t)
			location := fixture.image
			latest := fixture.image.Reference
			latest.Revision = "r2"
			old := fixture.host.records[fixture.image.Reference]
			newRecord := old
			newRecord.Original.Reference = latest
			fixture.host.records[latest] = newRecord
			fixture.host.permissions[latest] = retainedPermission{organization: "a"}
			switch kind {
			case "deleted":
				delete(fixture.host.permissions, location.Reference)
			case "denied":
				fixture.host.permissions[location.Reference] = retainedPermission{organization: "foreign"}
			case "revoked":
				fixture.host.revokeAfterLoad = true
			case "wrong page":
				location.Page.PhysicalIndex = 1
			case "outside image":
				location.Region.Right = 400
			case "wrong cell":
				location = fixture.cell
				location.Cell.Column = 1
			}
			// Act.
			resolved, err := fixture.reader.Resolve(
				context.Background(),
				layout.ResolveRequest{Read: fixture.read, Locations: []source.Locator{location}},
			)
			// Assert.
			if err == nil || len(resolved) != 0 {
				t.Fatal("invalid citation delivered")
			}
			if (kind == "deleted" || kind == "denied") &&
				(!errors.Is(err, ragy.ErrUnavailable) || fixture.host.loads != 0) {
				t.Fatal("inaccessible r1 loaded r2 or original payload")
			}
		})
	}
}

func TestRetainedRejectsDerivedOriginalConfusion(t *testing.T) {
	fixture := newResolveFixture(t)
	retained := fixture.host.records[fixture.image.Reference]
	exact := source.Locator{
		Reference: fixture.page.Reference,
		Kind:      source.TextLocation,
		Span:      source.ByteSpan{Start: 6, End: 10},
	}
	mapped, err := source.OriginalText(exact, "Alpha beta. Gamma.")
	if err != nil {
		t.Fatal(err)
	}
	retained.Derived = mapped
	if validationErr := retained.Validate(fixture.image.Reference); validationErr == nil {
		t.Fatal("original text substituted for derived image description")
	}
}
