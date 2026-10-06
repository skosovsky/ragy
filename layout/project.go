package layout

import (
	"context"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/source"
)

// Projected is source evidence ready for a host metadata/index projector.
// ID is revision/location identity, never retrieval rank. Mapping owns its supports.
type Projected struct {
	ID          string
	Location    source.Locator
	Text        source.MappedText
	Coverage    Coverage
	Diagnostics []Diagnostic
}

// ProjectionOptions carries the same trusted binding used for downstream consumers.
// ImageText is optional derived text; the core neither calls nor requires a model.
type ProjectionOptions struct {
	Read      access.Binding
	ImageText func(context.Context, access.Binding, Image) (source.MappedText, error)
}

// Project emits page/cell/available image evidence, retaining observed coverage.
// Callback, validation or freshness failure returns no partial evidence.
// Whole-document validation, including unique page text references, precedes
// any ImageText callback. Supplied payload ownership/authorization is host-owned.
func Project(ctx context.Context, document Document, options ProjectionOptions) ([]Projected, error) {
	if err := options.Read.Check(ctx); err != nil {
		return nil, err
	}
	if err := document.Validate(); err != nil {
		return nil, access.NonSkippable(err)
	}
	out, err := projectLayout(ctx, document, options)
	if gateErr := options.Read.Check(ctx); gateErr != nil {
		return nil, gateErr
	}
	if err != nil {
		return nil, access.NonSkippable(err)
	}
	return out, nil
}

func projectLayout(ctx context.Context, document Document, options ProjectionOptions) ([]Projected, error) {
	var out []Projected
	for _, page := range document.Pages {
		if err := options.Read.Check(ctx); err != nil {
			return nil, err
		}
		diagnostics := append(append([]Diagnostic(nil), document.Diagnostics...), page.Diagnostics...)
		projected, err := projectPage(ctx, page, document.Coverage, diagnostics, options)
		if err != nil {
			return nil, err
		}
		out = append(out, projected...)
	}
	return out, nil
}

func projectPage(
	ctx context.Context,
	page Page,
	coverage Coverage,
	diagnostics []Diagnostic,
	options ProjectionOptions,
) ([]Projected, error) {
	var out []Projected
	if page.Text != "" {
		var location source.Locator
		location.Reference, location.Kind, location.Span = page.Reference, source.TextLocation, source.ByteSpan{
			Start: 0,
			End:   len(page.Text),
		}
		mapped, err := source.OriginalText(location, page.Text)
		if err != nil {
			return nil, err
		}
		projected, err := projectEvidence(location, mapped, coverage, diagnostics)
		if err != nil {
			return nil, err
		}
		out = append(out, projected)
	}
	cells, err := projectCells(page.Cells, coverage, diagnostics)
	if err != nil {
		return nil, err
	}
	out = append(out, cells...)
	images, err := projectImages(ctx, page.Images, coverage, diagnostics, options)
	if err != nil {
		return nil, err
	}
	return append(out, images...), nil
}

func projectCells(cells []Cell, coverage Coverage, diagnostics []Diagnostic) ([]Projected, error) {
	out := make([]Projected, 0, len(cells))
	for _, cell := range cells {
		if cell.Text == "" {
			continue
		}
		mapped, err := source.SupportedOriginalText(cell.Text, []source.Locator{cell.Location})
		if err != nil {
			return nil, err
		}
		projected, err := projectEvidence(cell.Location, mapped, coverage, diagnostics)
		if err != nil {
			return nil, err
		}
		out = append(out, projected)
	}
	return out, nil
}

func projectImages(
	ctx context.Context,
	images []Image,
	coverage Coverage,
	diagnostics []Diagnostic,
	options ProjectionOptions,
) ([]Projected, error) {
	out := make([]Projected, 0, len(images))
	for _, image := range images {
		mapped, err := projectImage(ctx, image, options)
		if err != nil {
			return nil, err
		}
		if mapped.Text() == "" {
			continue
		}
		projected, err := projectEvidence(image.Location, mapped, coverage, diagnostics)
		if err != nil {
			return nil, err
		}
		out = append(out, projected)
	}
	return out, nil
}

func projectEvidence(
	location source.Locator,
	mapped source.MappedText,
	coverage Coverage,
	diagnostics []Diagnostic,
) (Projected, error) {
	identity, err := location.Identity()
	if err != nil {
		return Projected{}, err
	}
	return Projected{
		ID:          identity,
		Location:    location,
		Text:        mapped,
		Coverage:    coverage,
		Diagnostics: append([]Diagnostic(nil), diagnostics...),
	}, nil
}

func projectImage(ctx context.Context, image Image, options ProjectionOptions) (source.MappedText, error) {
	if options.ImageText == nil {
		return image.OCR.Mapping()
	}
	if err := options.Read.Check(ctx); err != nil {
		return source.MappedText{}, err
	}
	mapped, err := options.ImageText(ctx, options.Read, image)
	if gateErr := options.Read.Check(ctx); gateErr != nil {
		return source.MappedText{}, gateErr
	}
	if err != nil {
		return source.MappedText{}, err
	}
	if mapped.Text() == "" {
		return source.MappedText{}, nil
	}
	if err := mapped.Validate(); err != nil {
		return source.MappedText{}, err
	}
	for _, fragment := range mapped.Fragments() {
		if fragment.Origin != source.DerivedContent || fragment.Precision != source.UnavailablePrecision {
			return source.MappedText{}, ragy.ErrProtocol
		}
		for _, support := range fragment.Supports {
			if support != image.Location {
				return source.MappedText{}, ragy.ErrProtocol
			}
		}
	}
	return mapped, nil
}
