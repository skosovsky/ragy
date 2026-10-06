package access

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
)

// Fingerprint identifies the immutable binding without exposing raw policy data.
// It is a cache partition identity, not authorization: callers must still Check.
func (b Binding) Fingerprint() (string, error) {
	if err := b.Validate(); err != nil {
		return "", err
	}
	mandatory, err := b.state.mandatory.Fingerprint()
	if err != nil {
		return "", Protect(err)
	}
	targets := b.state.publication.Targets()
	sort.Slice(targets, func(i, j int) bool {
		a, c := targets[i], targets[j]
		if a.Target != c.Target {
			return a.Target < c.Target
		}
		if a.Namespace != c.Namespace {
			return a.Namespace < c.Namespace
		}
		return a.Source < c.Source
	})
	value := struct {
		Scoped      bool             `json:"scoped"`
		Snapshot    Snapshot         `json:"snapshot"`
		Mandatory   string           `json:"mandatory"`
		Publication string           `json:"publication"`
		Current     bool             `json:"current"`
		Targets     []TargetRevision `json:"targets"`
		Excluded    []string         `json:"excluded_targets"`
	}{Scoped: b.state.scoped, Snapshot: b.state.snapshot, Mandatory: mandatory, Publication: b.state.publication.ref, Current: b.state.publication.current, Targets: targets, Excluded: b.state.publication.ExcludedTargets()}
	data, err := json.Marshal(value)
	if err != nil {
		return "", Protect(err)
	}
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:]), nil
}
