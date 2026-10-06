package layout

import (
	"bytes"
	"context"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/source"
)

// Retained is an original source artifact supplied by host retention/storage.
// Derived descriptions remain separate from original text/media and support-only.
type Retained struct {
	Original    source.Locator
	Text        string
	Bytes       []byte
	MediaType   string
	CellRegion  source.Rectangle
	Words       []Word
	Coverage    Coverage
	Diagnostics []Diagnostic
	Derived     source.MappedText
}

// Validate checks source identity, original representation and coverage truthfulness.
func (r Retained) Validate(reference source.Reference) error {
	if r.Original.Reference != reference {
		return ragy.ErrProtocol
	}
	if err := r.Original.Validate(); err != nil {
		return err
	}
	if !validCoverage(r.Coverage) || !utf8.ValidString(r.Text) {
		return invalidLayout()
	}
	if err := validateDiagnostics(r.Diagnostics); err != nil {
		return err
	}
	if r.Coverage != Complete && len(r.Diagnostics) == 0 {
		return invalidLayout()
	}
	if r.Coverage == Complete && incompleteDiagnostics(r.Diagnostics) {
		return invalidLayout()
	}
	if err := r.validateOriginal(); err != nil {
		return err
	}
	return r.validateDerived()
}

func (r Retained) validateOriginal() error {
	var zeroRegion source.Rectangle
	switch r.Original.Kind {
	case source.DocumentLocation:
		if len(r.Bytes) == 0 || r.MediaType == "" || r.Text != "" || len(r.Words) != 0 || r.CellRegion != zeroRegion {
			return invalidLayout()
		}
	case source.PageLocation:
		if len(r.Bytes) != 0 || r.MediaType != "" || r.CellRegion != zeroRegion {
			return invalidLayout()
		}
		page := Page{
			Reference:   r.Original.Reference,
			Geometry:    r.Original.Page,
			Text:        r.Text,
			Words:       r.Words,
			Cells:       nil,
			Images:      nil,
			Coverage:    r.Coverage,
			Diagnostics: r.Diagnostics,
		}
		return page.Validate()
	case source.CellLocation:
		if len(r.Bytes) != 0 || r.MediaType != "" || len(r.Words) != 0 {
			return invalidLayout()
		}
		return r.CellRegion.Validate(r.Original.Page)
	case source.ImageLocation:
		if len(r.Bytes) == 0 || r.MediaType == "" || r.Text != "" || len(r.Words) != 0 || r.CellRegion != zeroRegion {
			return invalidLayout()
		}
	case source.TextLocation, source.RegionLocation:
		return invalidLayout()
	default:
		return invalidLayout()
	}
	return nil
}

func (r Retained) validateDerived() error {
	if r.Derived.Text() == "" {
		return nil
	}
	if err := r.Derived.Validate(); err != nil {
		return err
	}
	for _, fragment := range r.Derived.Fragments() {
		if fragment.Origin != source.DerivedContent || fragment.Precision != source.UnavailablePrecision {
			return invalidLayout()
		}
		for _, support := range fragment.Supports {
			if support != r.Original {
				return invalidLayout()
			}
		}
	}
	return nil
}

// CloneRetained returns independent original bytes, word and diagnostic slices.
func CloneRetained(retained Retained) (Retained, error) {
	retained.Bytes = bytes.Clone(retained.Bytes)
	retained.Words = append([]Word(nil), retained.Words...)
	retained.Diagnostics = append([]Diagnostic(nil), retained.Diagnostics...)
	return retained, nil
}

// ResolverConfig supplies host thin metadata and exact typed retained artifacts.
type ResolverConfig[TAccess any] struct {
	Target     string
	Schema     filter.Schema
	Catalog    source.Catalog[TAccess]
	Loader     source.Loader[Retained]
	Attributes func(TAccess) (filter.RawAttributes, error)
}

// Resolver projects source-admitted retained artifacts into canonical citations.
type Resolver[TAccess any] struct {
	reader *source.Reader[TAccess, Retained]
}

func NewResolver[TAccess any](config ResolverConfig[TAccess]) (*Resolver[TAccess], error) {
	reader, err := source.NewReader(source.ReadConfig[TAccess, Retained]{
		Target:          config.Target,
		Schema:          config.Schema,
		Catalog:         config.Catalog,
		Loader:          config.Loader,
		Attributes:      config.Attributes,
		ValidatePayload: func(reference source.Reference, retained Retained) error { return retained.Validate(reference) },
		ClonePayload:    CloneRetained,
	})
	if err != nil {
		return nil, err
	}
	return &Resolver[TAccess]{reader: reader}, nil
}

// ResolveRequest carries the original trusted binding and exact location batch.
type ResolveRequest struct {
	Read      access.Binding
	Locations []source.Locator
	Filters   filter.Condition
}

// Resolved preserves original/derived distinction, retained extent and coverage.
// Bytes are the original media, not cropped/generated pixels. Text maps only the
// requested text/word evidence; CellText is original logical cell text.
type Resolved struct {
	Location       source.Locator
	Original       source.Locator
	OriginalRegion source.Rectangle
	Text           source.MappedText
	CellText       string
	Bytes          []byte
	MediaType      string
	Coverage       Coverage
	Diagnostics    []Diagnostic
	Derived        source.MappedText
}

// Resolve deduplicates locations/reference loads and returns no partial batch on
// access, retained identity/geometry, UTF-8 or freshness failure.
func (r *Resolver[TAccess]) Resolve(ctx context.Context, request ResolveRequest) ([]Resolved, error) {
	if r == nil {
		return nil, access.NonSkippable(ragy.ErrInvalidArgument)
	}
	captured := request
	captured.Locations = append([]source.Locator(nil), request.Locations...)
	if err := captured.Read.Check(ctx); err != nil {
		return nil, err
	}
	resolved, err := r.resolve(ctx, captured)
	if gateErr := captured.Read.Check(ctx); gateErr != nil {
		return nil, gateErr
	}
	if err != nil {
		return nil, access.NonSkippable(err)
	}
	return resolved, nil
}

func (r *Resolver[TAccess]) resolve(ctx context.Context, request ResolveRequest) ([]Resolved, error) {
	locations, references, err := layoutReferences(request.Locations)
	if err != nil {
		return nil, err
	}
	records, err := r.reader.Lookup(
		ctx,
		source.LookupRequest{Read: request.Read, References: references, Filters: request.Filters},
	)
	if err != nil {
		return nil, err
	}
	indexed := make(map[source.Reference]Retained, len(records))
	for _, record := range records {
		indexed[record.Reference] = record.Payload
	}
	out := make([]Resolved, 0, len(locations))
	for _, location := range locations {
		if gateErr := request.Read.Check(ctx); gateErr != nil {
			return nil, gateErr
		}
		result, projectionErr := resolveRetained(location, indexed[location.Reference])
		if projectionErr != nil {
			return nil, projectionErr
		}
		out = append(out, result)
	}
	return out, nil
}

func layoutReferences(input []source.Locator) ([]source.Locator, []source.Reference, error) {
	locations := make([]source.Locator, 0, len(input))
	references := make([]source.Reference, 0, len(input))
	seenLocations := make(map[source.Locator]struct{}, len(input))
	seenReferences := make(map[source.Reference]struct{}, len(input))
	for _, location := range input {
		if err := location.Validate(); err != nil {
			return nil, nil, err
		}
		if _, exists := seenLocations[location]; exists {
			continue
		}
		seenLocations[location] = struct{}{}
		locations = append(locations, location)
		if _, exists := seenReferences[location.Reference]; !exists {
			references = append(references, location.Reference)
			seenReferences[location.Reference] = struct{}{}
		}
	}
	return locations, references, nil
}

func resolveRetained(location source.Locator, retained Retained) (Resolved, error) {
	var absent source.MappedText
	result := Resolved{
		Location:       location,
		Original:       retained.Original,
		OriginalRegion: retainedRegion(retained),
		Text:           absent,
		CellText:       "",
		Bytes:          nil,
		MediaType:      retained.MediaType,
		Coverage:       retained.Coverage,
		Diagnostics:    append([]Diagnostic(nil), retained.Diagnostics...),
		Derived:        retained.Derived,
	}
	switch location.Kind {
	case source.DocumentLocation:
		if location != retained.Original {
			return result, invalidLayout()
		}
		result.Bytes = bytes.Clone(retained.Bytes)
	case source.TextLocation, source.PageLocation, source.RegionLocation:
		mapped, err := resolvePageText(location, retained)
		if err != nil {
			return result, err
		}
		result.Text = mapped
	case source.CellLocation:
		if location != retained.Original {
			return result, invalidLayout()
		}
		result.CellText = retained.Text
	case source.ImageLocation:
		if retained.Original.Kind != source.ImageLocation || location.Page != retained.Original.Page ||
			!contains(retained.Original.Region, location.Region) {
			return result, invalidLayout()
		}
		result.Bytes = bytes.Clone(retained.Bytes)
	default:
		return result, invalidLayout()
	}
	return result, nil
}

func pageText(retained Retained, span source.ByteSpan) (source.MappedText, error) {
	var location source.Locator
	location.Reference, location.Kind, location.Span = retained.Original.Reference, source.TextLocation, span
	return source.OriginalText(location, retained.Text)
}
func regionText(retained Retained, region source.Rectangle) (source.MappedText, error) {
	var parts []source.MappedText
	for _, word := range retained.Words {
		if !intersects(region, word.Region) {
			continue
		}
		mapped, err := pageText(retained, word.Span)
		if err != nil {
			return source.MappedText{}, err
		}
		parts = append(parts, mapped)
	}
	if len(parts) == 0 {
		return source.MappedText{}, nil
	}
	return source.JoinMapped(" ", parts...)
}
func contains(outer, inner source.Rectangle) bool {
	return inner.Left >= outer.Left && inner.Top >= outer.Top && inner.Right <= outer.Right &&
		inner.Bottom <= outer.Bottom
}
func intersects(first, second source.Rectangle) bool {
	return first.Left < second.Right && first.Right > second.Left && first.Top < second.Bottom &&
		first.Bottom > second.Top
}

func resolvePageText(location source.Locator, retained Retained) (source.MappedText, error) {
	if retained.Original.Kind != source.PageLocation {
		return source.MappedText{}, invalidLayout()
	}
	switch location.Kind {
	case source.TextLocation:
		return source.OriginalText(location, retained.Text)
	case source.PageLocation:
		if location != retained.Original {
			return source.MappedText{}, invalidLayout()
		}
		if retained.Text == "" {
			return source.MappedText{}, nil
		}
		return pageText(retained, source.ByteSpan{Start: 0, End: len(retained.Text)})
	case source.RegionLocation:
		if location.Page != retained.Original.Page {
			return source.MappedText{}, invalidLayout()
		}
		return regionText(retained, location.Region)
	case source.DocumentLocation, source.CellLocation, source.ImageLocation:
		return source.MappedText{}, invalidLayout()
	default:
		return source.MappedText{}, invalidLayout()
	}
}

func retainedRegion(retained Retained) source.Rectangle {
	if retained.Original.Kind == source.CellLocation {
		return retained.CellRegion
	}
	return retained.Original.Region
}
