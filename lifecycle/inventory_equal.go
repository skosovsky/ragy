package lifecycle

import "github.com/skosovsky/ragy/source"

// Clone owns all nested artifact/support slices in a manifest snapshot.
func (m Manifest) Clone() Manifest { return cloneManifest(m) }

// SameTargetInventory compares exact artifact/support sets, independent of order.
// Inputs must be valid manifests; absent targets never compare as empty inventory.
func SameTargetInventory(first, second Manifest, name string) bool {
	a, foundA := namedArtifacts(first, name)
	b, foundB := namedArtifacts(second, name)
	return foundA && foundB && SameArtifactInventory(a, b)
}

// SameArtifactInventory compares exact artifact/support sets. Inputs must already
// satisfy Artifact.Validate for their owning identity; this is not source admission.
func SameArtifactInventory(a, b []Artifact) bool {
	if len(a) != len(b) {
		return false
	}
	indexed := make(map[source.Reference]Artifact, len(a))
	for _, artifact := range a {
		if _, duplicate := indexed[artifact.Reference]; duplicate {
			return false
		}
		indexed[artifact.Reference] = artifact
	}
	for _, artifact := range b {
		previous, exists := indexed[artifact.Reference]
		if !exists || !sameSupports(previous.Supports, artifact.Supports) {
			return false
		}
		delete(indexed, artifact.Reference)
	}
	return len(indexed) == 0
}
func namedArtifacts(m Manifest, name string) ([]Artifact, bool) {
	for _, target := range m.Targets {
		if target.Name == name {
			return target.Artifacts, true
		}
	}
	return nil, false
}
func sameSupports(first, second []source.Reference) bool {
	if len(first) != len(second) {
		return false
	}
	seen := make(map[source.Reference]struct{}, len(first))
	for _, ref := range first {
		seen[ref] = struct{}{}
	}
	if len(seen) != len(first) {
		return false
	}
	for _, ref := range second {
		if _, found := seen[ref]; !found {
			return false
		}
		delete(seen, ref)
	}
	return len(seen) == 0
}
