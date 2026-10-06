package evidence_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/evidence"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

func locationFixtures() []source.Locator {
	ref := reference()
	page := source.PageGeometry{PhysicalIndex: 0, PrintedLabel: "i", Width: 600, Height: 800, Rotation: 90}
	return []source.Locator{
		{Reference: ref, Kind: source.DocumentLocation},
		{Reference: ref, Kind: source.TextLocation, Span: source.ByteSpan{Start: 6, End: 10}},
		{Reference: ref, Kind: source.PageLocation, Page: page},
		{
			Reference: ref,
			Kind:      source.RegionLocation,
			Page:      page,
			Region:    source.Rectangle{Left: 60, Top: 80, Right: 100, Bottom: 100},
		},
		{
			Reference: ref,
			Kind:      source.CellLocation,
			Page:      page,
			Cell:      source.TableCell{Table: "t1", Element: "c1", Row: 0, Column: 0, RowSpan: 1, ColumnSpan: 2},
		},
		{
			Reference: ref,
			Kind:      source.ImageLocation,
			Page:      page,
			Region:    source.Rectangle{Left: 100, Top: 200, Right: 300, Bottom: 400},
		},
	}
}

func TestLocationExportAllKindsPrivacyAndOwnership(t *testing.T) {
	// Arrange.
	read, input, policy := fixture(t)
	locations := locationFixtures()
	input.Stages[0].Hits[0].Locations = locations
	policy.AllowLocation = func(source.Locator) bool { return true }
	// Act.
	record, err := evidence.Capture(context.Background(), read, input, policy)
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := record.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	hit := snapshot.Stages[0].Hits[0]
	if hit.LocationsState != evidence.Observed || len(hit.Locations) != 6 || hit.Locations[1].Span.End != 10 ||
		hit.Locations[4].Cell.ColumnSpan != 2 {
		t.Fatal("canonical coordinates lost")
	}
	data, err := record.MarshalJSON()
	if err != nil || strings.Contains(string(data), "access_fingerprint") ||
		strings.Contains(string(data), "private raw") {
		t.Fatal("auth/text leaked through locator", err)
	}
	decoded, err := evidence.Decode(data)
	if err != nil {
		t.Fatal(err)
	}
	roundtrip, err := decoded.MarshalJSON()
	if err != nil || string(roundtrip) != string(data) {
		t.Fatal("locator roundtrip failed", err)
	}
	locations[1].Span.End = 99
	hit.Locations[4].Cell.Table = "changed"
	again, err := record.MarshalJSON()
	if err != nil || string(again) != string(data) {
		t.Fatal("location snapshot aliases caller", err)
	}
}

func TestLocationExportRequiresExplicitPolicyAndCompleteSourceIdentity(t *testing.T) {
	for _, denial := range []string{"default", "numbers", "source", "location"} {
		t.Run(denial, func(t *testing.T) {
			// Arrange.
			read, input, policy := fixture(t)
			input.Stages[0].Hits[0].Locations = locationFixtures()
			policy.AllowLocation = func(source.Locator) bool { return true }
			switch denial {
			case "default":
				policy.AllowLocation = nil
			case "numbers":
				policy.AllowNumbers = false
			case "source":
				policy.AllowIdentifier = func(kind evidence.IdentifierKind, _ string) bool { return kind != evidence.SourceIdentifier }
			case "location":
				policy.AllowLocation = func(source.Locator) bool { return false }
			}
			// Act.
			record, err := evidence.Capture(context.Background(), read, input, policy)
			// Assert.
			if err != nil {
				t.Fatal(err)
			}
			snapshot, err := record.Snapshot()
			if err != nil || snapshot.Stages[0].Hits[0].LocationsState != evidence.Omitted ||
				len(snapshot.Stages[0].Hits[0].Locations) != 0 {
				t.Fatal("location export bypassed privacy", err)
			}
		})
	}
}

func TestForeignLocationRejectedBeforeExportCallback(t *testing.T) {
	// Arrange.
	read, input, policy := fixture(t)
	locations := locationFixtures()
	locations[0].Reference.Source = "foreign"
	input.Stages[0].Hits[0].Locations = locations[:1]
	calls := 0
	policy.AllowLocation = func(source.Locator) bool { calls++; return true }
	// Act.
	_, err := evidence.Capture(context.Background(), read, input, policy)
	// Assert.
	if !errors.Is(err, ragy.ErrProtocol) || calls != 0 {
		t.Fatal("unadmitted source reached locator policy", err)
	}
}

func TestLocationPolicyRevocationSuppressesRecord(t *testing.T) {
	// Arrange.
	revoked := false
	read, schema := scoped(t, &revoked)
	_, input, policy := fixture(t)
	input.Schema = schema
	input.Codec = nil
	input.Stages[0].Hits[0].Locations = locationFixtures()
	policy.AllowLocation = func(source.Locator) bool { revoked = true; return true }
	// Act: scoped capture needs the actual mandatory codec/source admission.
	input.Codec = retrieval.NewJSONCodec[metadata](schema)
	input.SourceAdmission = func(context.Context, access.Binding, source.Reference) error { return nil }
	_, err := evidence.Capture(context.Background(), read, input, policy)
	// Assert.
	if !access.IsProtectionFailure(err) {
		t.Fatal("policy revocation exported location", err)
	}
}

func TestLocatorWireRejectsInconsistentSourceAndGeometry(t *testing.T) {
	// Arrange.
	read, input, policy := fixture(t)
	input.Stages[0].Hits[0].Locations = locationFixtures()
	policy.AllowLocation = func(source.Locator) bool { return true }
	record, err := evidence.Capture(context.Background(), read, input, policy)
	if err != nil {
		t.Fatal(err)
	}
	for _, failure := range []string{"span", "association"} {
		t.Run(failure, func(t *testing.T) {
			changed, decodeErr := record.Snapshot()
			if decodeErr != nil {
				t.Fatal(decodeErr)
			}
			if failure == "span" {
				changed.Stages[0].Hits[0].Locations[1].Span.End = 0
			} else {
				changed.Stages[0].Hits[0].Locations[0].Source.ID.Value = new(string)
				*changed.Stages[0].Hits[0].Locations[0].Source.ID.Value = "foreign"
			}
			data, marshalErr := json.Marshal(changed)
			if marshalErr != nil {
				t.Fatal(marshalErr)
			}
			// Act.
			_, decodeErr = evidence.Decode(data)
			// Assert.
			if !errors.Is(decodeErr, ragy.ErrProtocol) {
				t.Fatal("invalid locator decoded", decodeErr)
			}
		})
	}
}
