package lifecycle

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"reflect"
	"slices"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/source"
)

// ArtifactFence reserves one exact target/reference identity after its inventory
// payload is compacted. Digest is lowercase SHA-256 of the JSON source.Reference.
type ArtifactFence struct {
	Target string `json:"target"`
	Digest string `json:"digest"`
}

// PublicationPin protects lifecycle metadata, not target/source availability or
// authorization. Released IDs remain reserved; there is no lease or clock expiry.
type PublicationPin struct {
	ID               string                  `json:"id"`
	Publication      string                  `json:"publication"`
	RequestedTargets []string                `json:"requested_targets"`
	Targets          []access.TargetRevision `json:"targets"`
	Released         bool                    `json:"released"`
}

func referenceDigest(ref source.Reference) string {
	data, _ := json.Marshal(ref)
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func (m Manifest) validateRetirement() error {
	if !m.Retired {
		if len(m.ArtifactFences) != 0 {
			return invalid()
		}
		return nil
	}
	if m.State == Unknown {
		return invalid()
	}
	seen := map[ArtifactFence]struct{}{}
	names := map[string]struct{}{}
	for _, target := range m.Targets {
		if target.State == TargetUnknown || len(target.Artifacts) != 0 {
			return invalid()
		}
		names[target.Name] = struct{}{}
	}
	for _, fence := range m.ArtifactFences {
		decoded, err := hex.DecodeString(fence.Digest)
		if err != nil || len(decoded) != sha256.Size ||
			hex.EncodeToString(decoded) != fence.Digest {
			return invalid()
		}
		if _, exists := names[fence.Target]; !exists {
			return invalid()
		}
		if _, exists := seen[fence]; exists {
			return invalid()
		}
		seen[fence] = struct{}{}
	}
	return nil
}

func targetMatches(identity Identity, target access.TargetRevision, name string) bool {
	return name == target.Target && identity.Namespace == target.Namespace &&
		identity.Source == target.Source &&
		identity.Revision == target.Revision &&
		identity.Transformation == target.Transformation &&
		identity.Access == target.AccessFingerprint
}

func (s Snapshot) validatePins(manifests map[string]Manifest) error {
	seen := map[string]struct{}{}
	for _, pin := range s.Pins {
		if _, exists := seen[pin.ID]; exists {
			return invalid()
		}
		seen[pin.ID] = struct{}{}
		if err := validatePin(pin, s.Namespace, manifests); err != nil {
			return err
		}
	}
	return nil
}

func validatePin(pin PublicationPin, namespace string, manifests map[string]Manifest) error {
	if !identities(pin.ID, pin.Publication) {
		return invalid()
	}
	requested, err := inventoryTargetSet(pin.RequestedTargets)
	if err != nil {
		return invalid()
	}
	if _, err = access.PinPublication(pin.Publication, pin.Targets); err != nil {
		return invalid()
	}
	for _, target := range pin.Targets {
		if _, exists := requested[target.Target]; !exists {
			return invalid()
		}
		if target.Namespace != namespace || !pinTargetPresent(pin, target, manifests) {
			return invalid()
		}
	}
	return nil
}

func pinTargetPresent(pin PublicationPin, target access.TargetRevision, manifests map[string]Manifest) bool {
	for _, m := range manifests {
		if m.Tombstone || m.PublishedAt.IsZero() || (m.Retired && !pin.Released) {
			continue
		}
		for _, inventory := range m.Targets {
			if targetMatches(m.Identity, target, inventory.Name) && inventory.State == TargetReady {
				return true
			}
		}
	}
	return false
}

// CompactHistory owns its returned snapshot and invalidates only exact selected
// handles after verifying metadata references and actual confirmed cleanup fences.
// It never persists, authorizes retention, dispatches cleanup or deletes payloads.
func CompactHistory(current Snapshot, ids []string) (Snapshot, error) {
	if err := current.Validate(); err != nil {
		return Snapshot{}, err
	}
	if len(ids) == 0 {
		return Snapshot{}, ragy.ErrInvalidArgument
	}
	selected, err := retirementSelection(current, ids)
	if err != nil {
		return Snapshot{}, err
	}
	// JSON roundtrip gives ownership of all nested wire-only collections.
	data, err := json.Marshal(current)
	if err != nil {
		return Snapshot{}, err
	}
	var next Snapshot
	if err = json.Unmarshal(data, &next); err != nil {
		return Snapshot{}, err
	}
	for i := range next.Manifests {
		m := &next.Manifests[i]
		if _, exists := selected[m.ID]; !exists || m.Retired {
			continue
		}
		m.Retired = true
		for j := range m.Targets {
			for _, artifact := range m.Targets[j].Artifacts {
				m.ArtifactFences = append(
					m.ArtifactFences,
					ArtifactFence{
						Target: m.Targets[j].Name,
						Digest: referenceDigest(artifact.Reference),
					},
				)
			}
			m.Targets[j].Artifacts = nil
		}
	}
	if err = next.Validate(); err != nil {
		return Snapshot{}, err
	}
	return next, nil
}

func retirementSelection(current Snapshot, ids []string) (map[string]struct{}, error) {
	selected := map[string]struct{}{}
	protected := protectedHistory(current)
	for _, id := range ids {
		if !identities(id) {
			return nil, ragy.ErrInvalidArgument
		}
		if _, exists := selected[id]; exists {
			return nil, ragy.ErrInvalidArgument
		}
		selected[id] = struct{}{}
		at := manifestIndex(current, id)
		if at < 0 {
			return nil, ragy.ErrUnavailable
		}
		if err := retirementEligible(current, current.Manifests[at], protected); err != nil {
			return nil, err
		}
	}
	return selected, nil
}

func retirementEligible(s Snapshot, m Manifest, protected map[string]struct{}) error {
	if m.Retired {
		return nil
	}
	if activePublication(s, m.Identity.Source) == m.ID || m.State == Unknown {
		return ErrProtected
	}
	for _, target := range m.Targets {
		if target.State == TargetUnknown || !cleanedInventory(s, m.ID, target.Name) {
			return ErrProtected
		}
	}
	if _, exists := protected[m.ID]; exists {
		return ErrProtected
	}
	for _, pin := range s.Pins {
		if pin.Released {
			continue
		}
		for _, target := range pin.Targets {
			for _, inventory := range m.Targets {
				if targetMatches(m.Identity, target, inventory.Name) {
					return ErrProtected
				}
			}
		}
	}
	return nil
}

func protectedHistory(s Snapshot) map[string]struct{} {
	// Every still-live operation and unfinished cleanup keeps its full reference
	// closure. A failed/canceled checkpoint is not evidence of remote termination.
	manifests := make(map[string]Manifest, len(s.Manifests))
	for _, other := range s.Manifests {
		manifests[other.ID] = other
	}
	protected := map[string]struct{}{}
	protectChain := func(id string) {
		visited := map[string]struct{}{}
		for id != "" {
			if _, exists := visited[id]; exists {
				break
			}
			visited[id] = struct{}{}
			protected[id] = struct{}{}
			row, exists := manifests[id]
			if !exists {
				break
			}
			id = row.ExpectedPublication
		}
	}
	for _, other := range s.Manifests {
		state, _ := confirmedState(other.State, other.Checkpoint)
		if !other.Retired && (other.State == Unknown || !afterPublished(state)) {
			protectChain(other.ID)
		}
	}
	for _, job := range s.Cleanups {
		if !job.Complete {
			protectChain(job.Owner)
			for _, item := range job.Items {
				protectChain(item.Manifest)
			}
		}
	}
	return protected
}

// sameReservedPlan compares immutable inventory without serializing/copying it.
func sameReservedPlan(first, second Manifest) bool {
	if first.ID != second.ID || first.Identity != second.Identity || first.Key != second.Key ||
		first.Payload != second.Payload ||
		first.ExpectedPublication != second.ExpectedPublication ||
		first.Tombstone != second.Tombstone ||
		first.Partial != second.Partial ||
		first.Retired != second.Retired ||
		len(first.Targets) != len(second.Targets) ||
		!slices.Equal(first.ArtifactFences, second.ArtifactFences) {
		return false
	}
	for i, target := range first.Targets {
		other := second.Targets[i]
		if target.Name != other.Name || target.Required != other.Required ||
			len(target.Artifacts) != len(other.Artifacts) {
			return false
		}
		for j, artifact := range target.Artifacts {
			if artifact.Reference != other.Artifacts[j].Reference ||
				!slices.Equal(artifact.Supports, other.Artifacts[j].Supports) {
				return false
			}
		}
	}
	return true
}

// ValidateReplacement is mandatory for Store implementations. Retired manifests
// and released pin IDs cannot disappear or be rebound. Every prior operation plan
// and bootstrap watermark receipt stays reserved. Newly compacted manifests
// must be exactly the safe compaction of the prior inventory; registered live pins
// can only be explicitly released with their original publication tuple intact.
func ValidateReplacement(current, next Snapshot) error {
	if current.Namespace != next.Namespace || next.Generation != current.Generation {
		return ragy.ErrInvalidArgument
	}
	if err := current.Validate(); err != nil {
		return err
	}
	if err := next.Validate(); err != nil {
		return err
	}
	if err := validateManifestReplacements(current, next); err != nil {
		return err
	}
	if err := validateCleanupReplacements(current, next); err != nil {
		return err
	}
	if err := validateInventoryReplacements(current, next); err != nil {
		return err
	}
	return validatePinReplacements(current, next)
}

func validateManifestReplacements(current, next Snapshot) error {
	currentByID := make(map[string]Manifest, len(current.Manifests))
	nextByID := make(map[string]Manifest, len(next.Manifests))
	for _, row := range current.Manifests {
		currentByID[row.ID] = row
	}
	for _, row := range next.Manifests {
		nextByID[row.ID] = row
	}
	if err := validateNewInventories(current, next.Manifests, currentByID); err != nil {
		return err
	}
	var retiredIDs []string
	for _, row := range next.Manifests {
		prior, exists := currentByID[row.ID]
		if row.Retired && !exists {
			return ragy.ErrInvalidArgument
		}
		if row.Retired && !prior.Retired {
			retiredIDs = append(retiredIDs, row.ID)
		}
	}
	var compacted Snapshot
	if len(retiredIDs) != 0 {
		var err error
		compacted, err = CompactHistory(current, retiredIDs)
		if err != nil {
			return err
		}
	}
	for _, prior := range current.Manifests {
		replacement, exists := nextByID[prior.ID]
		if err := validateManifestReplacement(prior, replacement, exists, compacted); err != nil {
			return err
		}
	}
	return nil
}

// validateNewInventories reserves current identities and each earlier new row in
// one index. Supports are not ownership keys: independent facts may share them.
func validateNewInventories(current Snapshot, rows []Manifest, currentByID map[string]Manifest) error {
	var fresh []Manifest
	for _, row := range rows {
		if _, exists := currentByID[row.ID]; !exists {
			fresh = append(fresh, row)
		}
	}
	if len(fresh) == 0 {
		return nil
	}
	occupied := make(map[inventoryReservation]struct{})
	fences := make(map[ArtifactFence]struct{})
	for _, row := range current.Manifests {
		for _, fence := range row.ArtifactFences {
			fences[fence] = struct{}{}
		}
		for _, target := range row.Targets {
			for _, artifact := range target.Artifacts {
				occupied[inventoryReservation{Target: target.Name, Reference: artifact.Reference}] = struct{}{}
			}
		}
	}
	for _, row := range fresh {
		if err := reserveNewInventory(row, occupied, fences); err != nil {
			return err
		}
	}
	return nil
}

type inventoryReservation struct {
	Target    string
	Reference source.Reference
}

func reserveNewInventory(
	row Manifest,
	occupied map[inventoryReservation]struct{},
	fences map[ArtifactFence]struct{},
) error {
	for _, target := range row.Targets {
		for _, artifact := range target.Artifacts {
			key := inventoryReservation{Target: target.Name, Reference: artifact.Reference}
			if _, exists := occupied[key]; exists {
				return ErrConflict
			}
			if len(fences) != 0 {
				fence := ArtifactFence{Target: target.Name, Digest: referenceDigest(artifact.Reference)}
				if _, exists := fences[fence]; exists {
					return ErrConflict
				}
			}
			occupied[key] = struct{}{}
		}
	}
	return nil
}

func validateManifestReplacement(prior, replacement Manifest, exists bool, compacted Snapshot) error {
	switch {
	case prior.Retired:
		if !exists || !reflect.DeepEqual(prior, replacement) {
			return ErrRetired
		}
	case !exists:
		return ErrProtected
	case replacement.Retired:
		if !reflect.DeepEqual(compacted.Manifests[manifestIndex(compacted, prior.ID)], replacement) {
			return ragy.ErrInvalidArgument
		}
	case !sameReservedPlan(prior, replacement) || (!prior.PublishedAt.IsZero() && !prior.PublishedAt.Equal(replacement.PublishedAt)):
		return ErrIdempotencyConflict
	}
	return nil
}
func validateCleanupReplacements(current, next Snapshot) error {
	for _, prior := range current.Cleanups {
		at := jobIndex(next, prior.Owner)
		if at < 0 {
			return ErrProtected
		}
		replacement := next.Cleanups[at]
		if err := validateCleanupReplacement(prior, replacement); err != nil {
			return err
		}
	}
	return nil
}
func validateCleanupReplacement(prior, replacement CleanupJob) error {
	if !prior.StartedAt.Equal(replacement.StartedAt) || !prior.Deadline.Equal(replacement.Deadline) ||
		len(prior.Items) != len(replacement.Items) ||
		(prior.Complete && !replacement.Complete) {
		return ErrProtected
	}
	for i, item := range prior.Items {
		nextItem := replacement.Items[i]
		if item.Manifest != nextItem.Manifest || item.Target != nextItem.Target ||
			nextItem.Attempts < item.Attempts ||
			(item.State == CleanupDone && nextItem.State != CleanupDone) {
			return ErrProtected
		}
	}
	return nil
}
func validateInventoryReplacements(current, next Snapshot) error {
	for _, prior := range current.Inventories {
		found := false
		for _, receipt := range next.Inventories {
			if receipt.Kind == prior.Kind && receipt.Watermark == prior.Watermark {
				found = reflect.DeepEqual(receipt, prior)
				break
			}
		}
		if !found {
			return ErrIdempotencyConflict
		}
	}
	return nil
}
func validatePinReplacements(current, next Snapshot) error {
	for _, prior := range current.Pins {
		found := false
		for _, pin := range next.Pins {
			if pin.ID != prior.ID {
				continue
			}
			found = true
			if prior.Publication != pin.Publication || !slices.Equal(prior.Targets, pin.Targets) ||
				!slices.Equal(prior.RequestedTargets, pin.RequestedTargets) ||
				(prior.Released && !pin.Released) {
				return ErrRetired
			}
		}
		if !found {
			return ErrRetired
		}
	}
	return nil
}
