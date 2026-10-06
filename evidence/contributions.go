package evidence

import (
	"slices"
	"strings"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
)

type contributionKey struct {
	query    int
	document string
	rank     int
}

func validateContributions[TMeta any](hit Hit[TMeta]) error {
	seen := make(map[contributionKey]bool, len(hit.Contributions))
	for _, item := range hit.Contributions {
		key := contributionKey{query: item.QueryIndex, document: item.DocumentID, rank: item.Rank}
		if item.QueryIndex < 0 || item.Rank <= 0 || strings.TrimSpace(item.DocumentID) == "" ||
			!utf8.ValidString(item.DocumentID) ||
			seen[key] {
			return ragy.ErrProtocol
		}
		seen[key] = true
	}
	return nil
}

func captureContributions(c *capture, input []Contribution) (CaptureState, []WireContribution) {
	if len(input) == 0 {
		return Unavailable, nil
	}
	if c.policy.AllowContribution == nil || !c.policy.AllowNumbers {
		return Omitted, nil
	}
	var out []WireContribution
	for _, item := range input {
		id := c.identifier(DocumentIdentifier, item.DocumentID)
		if c.err != nil || id.State != Observed {
			continue
		}
		owned := item
		owned.Locations = slices.Clone(item.Locations)
		allowed := c.policy.AllowContribution(owned)
		c.err = c.read.Check(c.ctx)
		if c.err != nil || !allowed {
			continue
		}
		state, locations := captureLocations(c, item.Locations)
		out = append(
			out,
			WireContribution{
				QueryIndex:     item.QueryIndex,
				DocumentID:     id,
				Rank:           item.Rank,
				LocationsState: state,
				Locations:      locations,
			},
		)
	}
	if len(out) == 0 {
		return Omitted, nil
	}
	return Observed, out
}

func validContributions(hit WireHit) bool {
	switch hit.ContributionsState {
	case Observed:
		if len(hit.Contributions) == 0 {
			return false
		}
	case Omitted, Unavailable, Unsupported:
		return len(hit.Contributions) == 0
	default:
		return false
	}
	seen := make(map[contributionKey]bool, len(hit.Contributions))
	for _, item := range hit.Contributions {
		if !validText(item.DocumentID) || item.DocumentID.State != Observed || item.QueryIndex < 0 || item.Rank <= 0 {
			return false
		}
		key := contributionKey{query: item.QueryIndex, document: *item.DocumentID.Value, rank: item.Rank}
		if seen[key] {
			return false
		}
		seen[key] = true
		if !validLocationList(item.LocationsState, item.Locations, hit.Sources) {
			return false
		}
		for _, loc := range item.Locations {
			if !locationCaptured(hit.Locations, loc) {
				return false
			}
		}
	}
	return true
}

func locationCaptured(locations []Location, wanted Location) bool {
	for _, candidate := range locations {
		if candidate.Kind == wanted.Kind && candidate.Span == wanted.Span && candidate.Page == wanted.Page &&
			candidate.Region == wanted.Region &&
			candidate.Cell == wanted.Cell &&
			sourceCaptured([]Source{candidate.Source}, wanted.Source) {
			return true
		}
	}
	return false
}
