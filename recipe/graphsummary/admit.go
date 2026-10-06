package graphsummary

import (
	"context"
	"slices"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/source"
)

func (r *Recipe[TAccess]) admit(
	ctx context.Context,
	request Request[TAccess],
	gate func() error,
) ([]Community[TAccess], string, error) {
	if err := r.shape(request); err != nil {
		return nil, "", err
	}
	binding, err := bindingID(ctx, request.Read, r.config.Schema)
	if err != nil {
		return nil, "", err
	}
	mandatory, err := request.Read.Prepare(
		ctx,
		r.config.Schema,
		filter.Condition{},
		access.Capabilities{ScopeProfile: true, PinnedPublication: true, RequirePinnedPublication: true},
	)
	if err != nil {
		return nil, "", err
	}
	owned := make([]Community[TAccess], len(request.Communities))
	for i, community := range request.Communities {
		owned[i] = Community[TAccess]{
			ID:       community.ID,
			Members:  slices.Clone(community.Members),
			Snippets: make([]Snippet[TAccess], len(community.Snippets)),
		}
		for j, snippet := range community.Snippets {
			accessValue, accessErr := r.access(snippet.Access, mandatory, gate)
			if accessErr != nil {
				return nil, "", accessErr
			}
			owned[i].Snippets[j] = Snippet[TAccess]{
				Mapping: snippet.Mapping,
				Access:  accessValue,
				Members: slices.Clone(snippet.Members),
			}
		}
	}
	if err = r.refresh(ctx, request.Read, owned, gate); err != nil {
		return nil, "", err
	}
	for _, community := range owned {
		if err = gate(); err != nil {
			return nil, "", err
		}
		if err = r.config.Membership(ctx, request.Read, community.ID, slices.Clone(community.Members)); err != nil {
			return nil, "", access.NonSkippable(err)
		}
		if err = gate(); err != nil {
			return nil, "", err
		}
	}

	return owned, binding, nil
}

func (r *Recipe[TAccess]) shape(request Request[TAccess]) error {
	if request.Question == "" || !utf8.ValidString(request.Question) || len(request.Communities) == 0 ||
		len(request.Communities) > r.config.MaxCommunities {
		return ragy.ErrInvalidArgument
	}
	remainingBytes := r.config.MaxInputBytes - len(request.Question)
	remainingSupports := r.config.MaxSupports
	seen := make(map[string]bool)
	for _, community := range request.Communities {
		if community.ID == "" || seen[community.ID] || len(community.Members) == 0 ||
			len(community.Members) > r.config.MaxMembers ||
			len(community.Snippets) == 0 ||
			len(community.Snippets) > r.config.MaxSnippets {
			return ragy.ErrInvalidArgument
		}
		seen[community.ID] = true
		if !unique(community.Members) {
			return ragy.ErrInvalidArgument
		}
		for _, snippet := range community.Snippets {
			remainingBytes -= len(snippet.Mapping.Text())
			supports := snippet.Mapping.Supports()
			remainingSupports -= len(supports)
			if remainingBytes < 0 || remainingSupports < 0 || snippet.Mapping.Text() == "" || len(supports) == 0 ||
				!unique(snippet.Members) || !subset(community.Members, snippet.Members) {
				return ragy.ErrInvalidArgument
			}
			if err := validateLocations(request.Read.Publication(), supports); err != nil {
				return err
			}
		}
	}
	return nil
}

func unique(values []string) bool {
	if len(values) == 0 {
		return false
	}
	seen := make(map[string]bool)
	for _, value := range values {
		if value == "" || seen[value] {
			return false
		}
		seen[value] = true
	}
	return true
}

func subset(all, values []string) bool {
	for _, value := range values {
		if !slices.Contains(all, value) {
			return false
		}
	}
	return true
}

func (r *Recipe[TAccess]) access(
	input TAccess,
	mandatory filter.Condition,
	gate func() error,
) (TAccess, error) {
	var empty TAccess
	if err := gate(); err != nil {
		return empty, err
	}
	owned, err := r.config.CloneAccess(input)
	if err != nil {
		return empty, err
	}
	if err = gate(); err != nil {
		return empty, err
	}
	attributes, err := r.config.Attributes(owned)
	if err != nil {
		return empty, err
	}
	if err = gate(); err != nil {
		return empty, err
	}
	attributes, err = r.config.Schema.NormalizeAttributes(attributes)
	if err != nil {
		return empty, err
	}
	matched, err := filter.MatchCondition(
		mandatory,
		func(field string) (any, bool) { value, ok := attributes[field]; return value, ok },
	)
	if err != nil {
		return empty, err
	}
	if !matched {
		return empty, access.NonSkippable(ragy.ErrUnavailable)
	}
	return owned, nil
}

func mapSnippets[TAccess any](community Community[TAccess]) []ModelSnippet {
	result := make([]ModelSnippet, len(community.Snippets))
	for i, snippet := range community.Snippets {
		result[i] = ModelSnippet{Index: i, Text: snippet.Mapping.Text()}
	}
	return result
}

func selectedSupports[TAccess any](community Community[TAccess], selected []int) ([]source.Locator, bool) {
	var supports []source.Locator
	var members []string
	for _, index := range selected {
		snippet := community.Snippets[index]
		supports = union(supports, snippet.Mapping.Supports())
		for _, member := range snippet.Members {
			if !slices.Contains(members, member) {
				members = append(members, member)
			}
		}
	}
	return supports, subset(members, community.Members)
}

func validateLocations(publication access.Publication, supports []source.Locator) error {
	for _, location := range supports {
		if err := location.Validate(); err != nil {
			return err
		}
		if !published(publication, location.Reference) {
			return access.NonSkippable(ragy.ErrUnavailable)
		}
	}
	return nil
}

func (r *Recipe[TAccess]) refresh(
	ctx context.Context,
	read access.Binding,
	communities []Community[TAccess],
	gate func() error,
) error {
	for _, community := range communities {
		for _, snippet := range community.Snippets {
			for _, location := range snippet.Mapping.Supports() {
				if err := gate(); err != nil {
					return err
				}
				if err := r.config.AdmitSource(ctx, read, location); err != nil {
					return access.NonSkippable(err)
				}
				if err := gate(); err != nil {
					return err
				}
			}
		}
	}
	return nil
}
