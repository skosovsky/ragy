package access

import (
	"slices"

	ragy "github.com/skosovsky/ragy"
)

// PinPartialPublication pins available targets and explicitly excludes complete
// configured branches. Excluded names are static labels, never source identifiers.
func PinPartialPublication(ref string, targets []TargetRevision, excluded []string) (Publication, error) {
	publication, err := PinPublication(ref, targets)
	if err != nil {
		return Publication{}, err
	}
	if len(excluded) == 0 {
		return Publication{}, Protect(ragy.ErrInvalidArgument)
	}
	names := slices.Clone(excluded)
	slices.Sort(names)
	for i, name := range names {
		if !staticTarget(name) || (i > 0 && names[i-1] == name) {
			return Publication{}, Protect(ragy.ErrInvalidArgument)
		}
		for _, target := range targets {
			if target.Target == name {
				return Publication{}, Protect(ragy.ErrInvalidArgument)
			}
		}
	}
	publication.excluded = names
	return publication, nil
}

func staticTarget(name string) bool {
	if name == "" || len(name) > 128 {
		return false
	}
	for _, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') ||
			r == '_' || r == '-' || r == '.' {
			continue
		}
		return false
	}
	return true
}

func (p Publication) IsPartial() bool           { return len(p.excluded) > 0 }
func (p Publication) ExcludedTargets() []string { return slices.Clone(p.excluded) }

// AdmitTarget rejects an excluded branch before target I/O. Available empty
// namespaces remain usable; the host target declaration is backed by conformance.
func (p Publication) AdmitTarget(name string) error {
	if !p.IsPartial() {
		return nil
	}
	available := false
	for _, target := range p.targets {
		if target.Target == name {
			available = true
			break
		}
	}
	if !available || slices.Contains(p.excluded, name) {
		return UnsupportedCapability(ragy.ErrUnsupported)
	}
	return nil
}
