package graphsummary

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"slices"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/source"
)

func (s Summary) CommunityIDs() []string     { return slices.Clone(s.communities) }
func (s Summary) Supports() []source.Locator { return slices.Clone(s.supports) }

// CoversMembership reports declared snippet-member coverage, not semantic
// correctness of generated prose. The latter belongs to external evaluation.
func (s Summary) CoversMembership() bool { return s.covered }

// Resolve returns support-only derived text, never an exact original quotation.
// A changed scope/publication, revoked source, deleted support or expiry fails
// closed. Revalidation never grants a new binding or refreshes a cached summary.
func (s Summary) Resolve(
	ctx context.Context,
	read access.Binding,
	admit func(context.Context, access.Binding, source.Locator) error,
) (source.MappedText, error) {
	var empty source.MappedText
	if s.text == "" || admit == nil {
		return empty, ragy.ErrInvalidArgument
	}
	binding, err := bindingID(ctx, read, s.schema)
	if err != nil {
		return empty, err
	}
	if binding != s.binding {
		return empty, access.NonSkippable(ragy.ErrUnavailable)
	}
	for _, location := range s.supports {
		if err = read.Check(ctx); err != nil {
			return empty, err
		}
		if err = admit(ctx, read, location); err != nil {
			return empty, access.NonSkippable(err)
		}
		if err = read.Check(ctx); err != nil {
			return empty, err
		}
	}
	mapping, err := source.DerivedText(s.text, s.supports)
	if err != nil {
		return empty, err
	}
	if err = read.Check(ctx); err != nil {
		return empty, err
	}
	return mapping, nil
}

func bindingID(ctx context.Context, read access.Binding, schema filter.Schema) (string, error) {
	mandatory, err := read.Prepare(
		ctx,
		schema,
		filter.Condition{},
		access.Capabilities{ScopeProfile: true, PinnedPublication: true, RequirePinnedPublication: true},
	)
	if err != nil {
		return "", err
	}
	predicate, err := mandatory.Fingerprint()
	if err != nil {
		return "", err
	}
	data, err := json.Marshal(struct {
		Snapshot    access.Snapshot         `json:"snapshot"`
		Predicate   string                  `json:"predicate"`
		Publication string                  `json:"publication"`
		Targets     []access.TargetRevision `json:"targets"`
	}{Snapshot: read.Snapshot(), Predicate: predicate, Publication: read.Publication().Reference(), Targets: read.Publication().Targets()})
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:]), nil
}

func published(publication access.Publication, reference source.Reference) bool {
	for _, target := range publication.Targets() {
		if target.Namespace == reference.Namespace && target.Source == reference.Source &&
			target.Revision == reference.Revision &&
			target.AccessFingerprint == reference.AccessFingerprint {
			return true
		}
	}
	return false
}

func union(all, incoming []source.Locator) []source.Locator {
	for _, location := range incoming {
		if !slices.Contains(all, location) {
			all = append(all, location)
		}
	}
	return all
}
