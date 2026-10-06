package evidence

import (
	"slices"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/source"
)

func hitLocations[TMeta any](hit Hit[TMeta]) []source.Locator {
	out := hit.Document.SourceLocations()
	locations := slices.Clone(hit.Locations)
	for _, contribution := range hit.Contributions {
		locations = append(locations, contribution.Locations...)
	}
	for _, loc := range locations {
		if !slices.Contains(out, loc) {
			out = append(out, loc)
		}
	}
	return out
}

func validateLocations[TMeta any](hit Hit[TMeta]) error {
	for _, loc := range hitLocations(hit) {
		if err := loc.Validate(); err != nil {
			return err
		}
		if !slices.Contains(hit.Sources, loc.Reference) {
			return ragy.ErrProtocol
		}
	}
	return nil
}

func captureLocations(c *capture, locations []source.Locator) (CaptureState, []Location) {
	if len(locations) == 0 {
		return Unavailable, nil
	}
	if c.policy.AllowLocation == nil || !c.policy.AllowNumbers {
		return Omitted, nil
	}
	var out []Location
	for _, loc := range locations {
		src := captureSource(c, loc.Reference)
		if c.err != nil || !fullSource(src) {
			continue
		}
		allowed := c.policy.AllowLocation(loc)
		c.err = c.read.Check(c.ctx)
		if c.err != nil || !allowed {
			continue
		}
		out = append(
			out,
			Location{Source: src, Kind: loc.Kind, Span: loc.Span, Page: loc.Page, Region: loc.Region, Cell: loc.Cell},
		)
	}
	if len(out) == 0 {
		return Omitted, nil
	}
	return Observed, out
}

func fullSource(src Source) bool {
	for _, field := range []Text{src.Namespace, src.ID, src.Revision, src.Transformation, src.Artifact, src.Representation} {
		if !validText(field) || field.State != Observed {
			return false
		}
	}
	return true
}

func validLocation(loc Location) bool {
	if !fullSource(loc.Source) {
		return false
	}
	// This sentinel verifies only the locator union shape. It attests no authority.
	original := source.Locator{Reference: source.Reference{
		Namespace:         *loc.Source.Namespace.Value,
		Source:            *loc.Source.ID.Value,
		Revision:          *loc.Source.Revision.Value,
		Transformation:    *loc.Source.Transformation.Value,
		AccessFingerprint: "wire-shape",
		Artifact:          *loc.Source.Artifact.Value,
		Representation:    *loc.Source.Representation.Value,
	}, Kind: loc.Kind, Span: loc.Span, Page: loc.Page, Region: loc.Region, Cell: loc.Cell}
	return original.Validate() == nil
}

func validLocations(hit WireHit) bool {
	return validLocationList(hit.LocationsState, hit.Locations, hit.Sources)
}

func validLocationList(state CaptureState, locations []Location, sources []Source) bool {
	switch state {
	case Observed:
		if len(locations) == 0 {
			return false
		}
	case Omitted, Unavailable, Unsupported:
		return len(locations) == 0
	default:
		return false
	}
	for _, loc := range locations {
		if !validLocation(loc) || !sourceCaptured(sources, loc.Source) {
			return false
		}
	}
	return true
}

func sourceCaptured(sources []Source, wanted Source) bool {
	for _, candidate := range sources {
		if !fullSource(candidate) {
			continue
		}
		if *candidate.Namespace.Value == *wanted.Namespace.Value && *candidate.ID.Value == *wanted.ID.Value &&
			*candidate.Revision.Value == *wanted.Revision.Value &&
			*candidate.Transformation.Value == *wanted.Transformation.Value &&
			*candidate.Artifact.Value == *wanted.Artifact.Value &&
			*candidate.Representation.Value == *wanted.Representation.Value {
			return true
		}
	}
	return false
}
