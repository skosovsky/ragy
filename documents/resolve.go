package documents

import (
	"context"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/source"
)

// TextResolveRequest addresses exact retained UTF-8 representations. Geometry,
// cell and image resolution belongs to the corresponding parser/source port and
// is rejected here before I/O; a text resolver never treats PDF binary as text.
type TextResolveRequest struct {
	Read      access.Binding
	Locations []source.Locator
	Filters   filter.Condition
}

// ResolvedText contains an exact retained locator and an owned source mapping.
// Source metadata/payload outside the requested span is not exported.
type ResolvedText struct {
	Location source.Locator
	Text     source.MappedText
}

// ResolveText admits the entire batch before materialization and returns no
// citations if any revision, access check, UTF-8 interval or gate fails. Duplicate
// locations are collapsed while preserving first occurrence order.
func (h *Hydrator[TAccess, TMeta]) ResolveText(
	ctx context.Context,
	request TextResolveRequest,
) ([]ResolvedText, error) {
	captured := request
	captured.Locations = append([]source.Locator(nil), request.Locations...)
	if err := captured.Read.Check(ctx); err != nil {
		return nil, err
	}
	resolved, err := h.resolveText(ctx, captured)
	if gateErr := captured.Read.Check(ctx); gateErr != nil {
		return nil, gateErr
	}
	if err != nil {
		return nil, access.NonSkippable(err)
	}
	return resolved, nil
}

func (h *Hydrator[TAccess, TMeta]) resolveText(
	ctx context.Context,
	request TextResolveRequest,
) ([]ResolvedText, error) {
	locations, references, err := textReferences(request.Locations)
	if err != nil {
		return nil, err
	}
	payloads, err := h.Lookup(
		ctx,
		source.LookupRequest{Read: request.Read, References: references, Filters: request.Filters},
	)
	if err != nil {
		return nil, err
	}
	content := make(map[source.Reference]string, len(payloads))
	for _, payload := range payloads {
		content[payload.Reference] = payload.Payload.Content
	}
	resolved := make([]ResolvedText, 0, len(locations))
	for _, location := range locations {
		if gateErr := request.Read.Check(ctx); gateErr != nil {
			return nil, gateErr
		}
		retained, exists := content[location.Reference]
		if !exists {
			return nil, ragy.ErrUnavailable
		}
		mapped, mappingErr := source.OriginalText(location, retained)
		if mappingErr != nil {
			return nil, mappingErr
		}
		resolved = append(resolved, ResolvedText{Location: location, Text: mapped})
	}
	return resolved, nil
}

func textReferences(input []source.Locator) ([]source.Locator, []source.Reference, error) {
	locations := make([]source.Locator, 0, len(input))
	references := make([]source.Reference, 0, len(input))
	seenLocations := make(map[source.Locator]struct{}, len(input))
	seenReferences := make(map[source.Reference]struct{}, len(input))
	for _, location := range input {
		if err := location.Validate(); err != nil {
			return nil, nil, err
		}
		if location.Kind != source.TextLocation {
			return nil, nil, ragy.ErrUnsupported
		}
		if _, exists := seenLocations[location]; exists {
			continue
		}
		seenLocations[location] = struct{}{}
		locations = append(locations, location)
		if _, exists := seenReferences[location.Reference]; !exists {
			seenReferences[location.Reference] = struct{}{}
			references = append(references, location.Reference)
		}
	}
	return locations, references, nil
}
