// Package lifecycle defines durable source publication and artifact checkpoints.
package lifecycle

import (
	"fmt"
	"time"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/source"
)

const SchemaIdentity = "ragy.lifecycle"

type State string

const (
	Planned        State = "planned"
	Staging        State = "staging"
	Ready          State = "ready"
	Published      State = "published"
	CleanupPending State = "cleanup-pending"
	Complete       State = "complete"
	Failed         State = "failed"
	Canceled       State = "canceled"
	Unknown        State = "unknown"
)

type TargetState string

const (
	TargetPending TargetState = "pending"
	TargetReady   TargetState = "ready"
	TargetFailed  TargetState = "failed"
	TargetUnknown TargetState = "unknown"
)

// Identity includes access even when content is unchanged.
type Identity struct {
	Namespace      string `json:"namespace"`
	Source         string `json:"source"`
	Revision       string `json:"revision"`
	Content        string `json:"content_fingerprint"`
	Transformation string `json:"transformation"`
	Access         string `json:"access_fingerprint"`
}

// Artifact records one managed record and its original source supports. A host-owned
// graph fact can retain an independent basis through the target cleanup contract.
type Artifact struct {
	Reference source.Reference   `json:"reference"`
	Supports  []source.Reference `json:"supports"`
}

type Target struct {
	Name      string      `json:"name"`
	Required  bool        `json:"required"`
	State     TargetState `json:"state"`
	Revision  string      `json:"revision"`
	Artifacts []Artifact  `json:"artifacts"`
}

// Manifest stores a complete retained inventory, never just replacement chunk IDs.
// Checkpoint is the last confirmed state when State is failed/canceled/unknown.
type Manifest struct {
	ID                  string    `json:"id"`
	Identity            Identity  `json:"identity"`
	Key                 string    `json:"idempotency_key"`
	Payload             string    `json:"payload_fingerprint"`
	ExpectedPublication string    `json:"expected_publication"`
	Tombstone           bool      `json:"tombstone"`
	Partial             bool      `json:"partial"`
	State               State     `json:"state"`
	Checkpoint          State     `json:"checkpoint"`
	Targets             []Target  `json:"targets"`
	PublishedAt         time.Time `json:"published_at"`
}

type Publication struct {
	Source   string `json:"source"`
	Manifest string `json:"manifest"`
}

// Snapshot is one namespace's atomically persisted publication/manifest inventory.
type Snapshot struct {
	Inventories  []InventoryReceipt `json:"inventories"`
	Schema       string             `json:"schema"`
	Namespace    string             `json:"namespace"`
	Generation   uint64             `json:"generation"`
	Manifests    []Manifest         `json:"manifests"`
	Publications []Publication      `json:"publications"`
	Cleanups     []CleanupJob       `json:"cleanups"`
}

func (i Identity) Validate() error {
	if !identities(i.Namespace, i.Source, i.Revision, i.Content, i.Transformation, i.Access) {
		return invalid()
	}
	return nil
}

func (m Manifest) Validate() error {
	if !identities(m.ID, m.Key, m.Payload) || !utf8.ValidString(m.ExpectedPublication) {
		return invalid()
	}
	if err := m.Identity.Validate(); err != nil {
		return err
	}
	effective, err := confirmedState(m.State, m.Checkpoint)
	if err != nil {
		return err
	}
	if afterPublished(effective) == m.PublishedAt.IsZero() {
		return invalid()
	}
	if len(m.Targets) == 0 && !m.Tombstone {
		return invalid()
	}
	names := make(map[string]struct{}, len(m.Targets))
	ready := 0
	for _, target := range m.Targets {
		if _, exists := names[target.Name]; exists {
			return invalid()
		}
		names[target.Name] = struct{}{}
		if err = target.validate(m.Identity); err != nil {
			return err
		}
		if target.State == TargetReady {
			ready++
		}
		if afterReady(effective) && target.Required && target.State != TargetReady && !m.Partial && !m.Tombstone {
			return invalid()
		}
	}
	if afterReady(effective) && !m.Tombstone && ready == 0 {
		return invalid()
	}
	return nil
}

func (t Target) validate(identity Identity) error {
	if !identities(t.Name) {
		return invalid()
	}
	switch t.State {
	case TargetPending, TargetFailed, TargetUnknown:
	case TargetReady:
		if t.Revision != identity.Revision {
			return invalid()
		}
	default:
		return invalid()
	}
	seen := make(map[source.Reference]struct{}, len(t.Artifacts))
	for _, artifact := range t.Artifacts {
		if err := artifact.Validate(identity); err != nil {
			return err
		}
		if _, exists := seen[artifact.Reference]; exists {
			return invalid()
		}
		seen[artifact.Reference] = struct{}{}
	}
	return nil
}

func (a Artifact) Validate(identity Identity) error {
	ref := a.Reference
	if err := ref.Validate(); err != nil {
		return err
	}
	if ref.Namespace != identity.Namespace || ref.Source != identity.Source || ref.Revision != identity.Revision ||
		ref.Transformation != identity.Transformation || ref.AccessFingerprint != identity.Access {
		return invalid()
	}
	if len(a.Supports) == 0 {
		return invalid()
	}
	supports := make(map[source.Reference]struct{}, len(a.Supports))
	for _, support := range a.Supports {
		if err := support.Validate(); err != nil {
			return err
		}
		if support.Namespace != identity.Namespace {
			return invalid()
		}
		if _, exists := supports[support]; exists {
			return invalid()
		}
		supports[support] = struct{}{}
	}
	return nil
}

func (s Snapshot) Validate() error {
	if s.Schema != SchemaIdentity || !identities(s.Namespace) {
		return invalid()
	}
	manifests := make(map[string]Manifest, len(s.Manifests))
	for _, manifest := range s.Manifests {
		if manifest.Identity.Namespace != s.Namespace {
			return invalid()
		}
		if err := manifest.Validate(); err != nil {
			return err
		}
		if _, exists := manifests[manifest.ID]; exists {
			return invalid()
		}
		manifests[manifest.ID] = manifest
	}
	if err := s.validatePublications(manifests); err != nil {
		return err
	}
	if err := s.validateInventories(manifests); err != nil {
		return err
	}
	return s.validateCleanups(manifests)
}

func confirmedState(state, checkpoint State) (State, error) {
	switch state {
	case Planned, Staging, Ready, Published, CleanupPending, Complete:
		if checkpoint != "" {
			return "", invalid()
		}
		return state, nil
	case Failed, Canceled, Unknown:
		switch checkpoint {
		case Planned, Staging, Ready, Published, CleanupPending, Complete:
			return checkpoint, nil
		case Failed, Canceled, Unknown:
			return "", invalid()
		default:
			return "", invalid()
		}
	default:
		return "", invalid()
	}
}
func afterReady(state State) bool { return state == Ready || afterPublished(state) }
func afterPublished(state State) bool {
	return state == Published || state == CleanupPending || state == Complete
}
func identities(values ...string) bool {
	for _, value := range values {
		if value == "" || !utf8.ValidString(value) {
			return false
		}
	}
	return true
}
func invalid() error { return fmt.Errorf("%w: lifecycle snapshot", ragy.ErrInvalidArgument) }

func (s Snapshot) validatePublications(manifests map[string]Manifest) error {
	sources := make(map[string]struct{}, len(s.Publications))
	for _, publication := range s.Publications {
		if !identities(publication.Source, publication.Manifest) {
			return invalid()
		}
		if _, exists := sources[publication.Source]; exists {
			return invalid()
		}
		sources[publication.Source] = struct{}{}
		manifest, exists := manifests[publication.Manifest]
		if !exists || manifest.Identity.Source != publication.Source {
			return invalid()
		}
		state, err := confirmedState(manifest.State, manifest.Checkpoint)
		if err != nil || !afterPublished(state) {
			return invalid()
		}
	}
	return nil
}
