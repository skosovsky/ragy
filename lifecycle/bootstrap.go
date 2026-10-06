package lifecycle

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"slices"

	ragy "github.com/skosovsky/ragy"
)

type InventoryKind string

const (
	DeltaInventory    InventoryKind = "delta"
	CompleteInventory InventoryKind = "complete"
)

type InventoryCoverage string

const (
	PartialInventory InventoryCoverage = "partial"
	FullInventory    InventoryCoverage = "complete"
)

// UnmanagedRecord is an opaque backend key with no inferred source/revision.
type UnmanagedRecord struct {
	Target string `json:"target"`
	Key    string `json:"key"`
}

type Inventory struct {
	Namespace string            `json:"namespace"`
	Kind      InventoryKind     `json:"kind"`
	Watermark string            `json:"watermark"`
	Coverage  InventoryCoverage `json:"coverage"`
	Targets   []string          `json:"targets"`
	Manifests []Manifest        `json:"manifests"`
	Unmanaged []UnmanagedRecord `json:"unmanaged"`
}

// InventoryConfirmation attests a fenced observation of the exact input fingerprint.
type InventoryConfirmation struct {
	Namespace   string
	Watermark   string
	Fingerprint string
	Coverage    InventoryCoverage
}

// InventoryVerifier observes backend ownership, exact revisions/artifacts/supports
// and complete namespace coverage. It must not mutate or guess unmanaged records.
type InventoryVerifier interface {
	VerifyInventory(context.Context, Inventory) (InventoryConfirmation, error)
}

// InventoryReceipt retains deletion proposals against their original publication.
// They are not tombstone acknowledgements and never authorize raw-ID deletion.
type InventoryReceipt struct {
	Kind        InventoryKind     `json:"kind"`
	Watermark   string            `json:"watermark"`
	Fingerprint string            `json:"fingerprint"`
	Coverage    InventoryCoverage `json:"coverage"`
	Targets     []string          `json:"targets"`
	Imported    []string          `json:"imported"`
	Unmanaged   []UnmanagedRecord `json:"unmanaged"`
	Missing     []Publication     `json:"missing"`
}

type Bootstrapper struct {
	store    Store
	verifier InventoryVerifier
}

func NewBootstrapper(store Store, verifier InventoryVerifier) (*Bootstrapper, error) {
	if nilPort(store) || nilPort(verifier) {
		return nil, ragy.ErrInvalidArgument
	}
	return &Bootstrapper{store: store, verifier: verifier}, nil
}

func (i Inventory) Validate() error {
	if !identities(i.Namespace, i.Watermark) || !inventoryProfile(i.Kind, i.Coverage) || len(i.Targets) == 0 {
		return invalid()
	}
	targets, err := inventoryTargetSet(i.Targets)
	if err != nil {
		return err
	}
	seen := make(map[string]struct{})
	ids := make(map[string]struct{})
	keys := make(map[string]struct{})
	for _, manifest := range i.Manifests {
		if err := manifest.Validate(); err != nil {
			return err
		}
		if manifest.Identity.Namespace != i.Namespace || manifest.State != Published || manifest.Tombstone {
			return invalid()
		}
		if _, exists := seen[manifest.Identity.Source]; exists {
			return invalid()
		}
		if _, exists := ids[manifest.ID]; exists {
			return invalid()
		}
		if _, exists := keys[manifest.Key]; exists {
			return invalid()
		}
		seen[manifest.Identity.Source] = struct{}{}
		ids[manifest.ID] = struct{}{}
		keys[manifest.Key] = struct{}{}
		if err := inventoryTargets(manifest, targets); err != nil {
			return err
		}
	}
	return validateUnmanaged(i.Unmanaged, targets)
}

// Fingerprint is stable over an owned exact envelope, including unmanaged keys.
func (i Inventory) Fingerprint() (string, error) {
	if err := i.Validate(); err != nil {
		return "", err
	}
	data, err := json.Marshal(i)
	if err != nil {
		return "", ragy.ErrProtocol
	}
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:]), nil
}

// Import persists verified ownership only. Missing complete-inventory sources stay
// published until an explicit expected-publication tombstone succeeds.
func (b *Bootstrapper) Import(ctx context.Context, input Inventory) (InventoryReceipt, error) {
	if b == nil {
		return InventoryReceipt{}, ragy.ErrInvalidArgument
	}
	captured := cloneInventory(input)
	fingerprint, err := captured.Fingerprint()
	if err != nil {
		return InventoryReceipt{}, err
	}
	if err = ctx.Err(); err != nil {
		return InventoryReceipt{}, err
	}
	snapshot, err := b.store.Load(ctx, captured.Namespace)
	if err != nil {
		return InventoryReceipt{}, err
	}
	if snapshot.Namespace != captured.Namespace || snapshot.Validate() != nil {
		return InventoryReceipt{}, ragy.ErrProtocol
	}
	if receipt, found, replayErr := inventoryReplay(snapshot, captured, fingerprint); found || replayErr != nil {
		if replayErr != nil {
			return InventoryReceipt{}, replayErr
		}
		if err = ctx.Err(); err != nil {
			return InventoryReceipt{}, err
		}
		return receipt, nil
	}
	confirmation, err := b.verifier.VerifyInventory(ctx, cloneInventory(captured))
	if err != nil {
		return InventoryReceipt{}, err
	}
	if err = ctx.Err(); err != nil {
		return InventoryReceipt{}, err
	}
	if confirmation.Namespace != captured.Namespace || confirmation.Watermark != captured.Watermark ||
		confirmation.Fingerprint != fingerprint || confirmation.Coverage != captured.Coverage {
		return InventoryReceipt{}, ragy.ErrProtocol
	}
	receipt := InventoryReceipt{
		Kind:        captured.Kind,
		Watermark:   captured.Watermark,
		Fingerprint: fingerprint,
		Coverage:    captured.Coverage,
		Targets: slices.Clone(
			captured.Targets,
		),
		Imported:  nil,
		Unmanaged: slices.Clone(captured.Unmanaged),
		Missing:   nil,
	}
	if err = mergeInventory(&snapshot, captured, &receipt); err != nil {
		return InventoryReceipt{}, err
	}
	snapshot.Inventories = append(snapshot.Inventories, receipt)
	if _, err = b.store.CompareSwap(ctx, snapshot.Generation, snapshot); err != nil {
		return InventoryReceipt{}, mutationError(err)
	}
	if err = ctx.Err(); err != nil {
		return InventoryReceipt{}, mutationError(err)
	}
	return cloneReceipt(receipt), nil
}
func mergeInventory(snapshot *Snapshot, input Inventory, receipt *InventoryReceipt) error {
	incoming := make(map[string]struct{}, len(input.Manifests))
	for _, manifest := range input.Manifests {
		incoming[manifest.Identity.Source] = struct{}{}
		if err := mergeInventorySource(snapshot, manifest); err != nil {
			return err
		}
		receipt.Imported = append(receipt.Imported, manifest.ID)
	}
	if input.Kind == CompleteInventory {
		receipt.Missing = missingInventorySources(*snapshot, incoming)
	}
	return nil
}
func replacePublication(snapshot *Snapshot, sourceID, manifestID string) {
	for i, publication := range snapshot.Publications {
		if publication.Source == sourceID {
			snapshot.Publications[i].Manifest = manifestID
			return
		}
	}
	snapshot.Publications = append(snapshot.Publications, Publication{Source: sourceID, Manifest: manifestID})
}
func inventoryProfile(kind InventoryKind, coverage InventoryCoverage) bool {
	switch kind {
	case DeltaInventory:
		return coverage == PartialInventory || coverage == FullInventory
	case CompleteInventory:
		return coverage == FullInventory
	default:
		return false
	}
}
func inventoryTargets(manifest Manifest, targets map[string]struct{}) error {
	found := make(map[string]struct{})
	for _, target := range manifest.Targets {
		if _, exists := targets[target.Name]; !exists || target.State != TargetReady {
			return invalid()
		}
		found[target.Name] = struct{}{}
	}
	if len(found) != len(targets) {
		return invalid()
	}
	return nil
}
func validateUnmanaged(records []UnmanagedRecord, targets map[string]struct{}) error {
	seen := make(map[UnmanagedRecord]struct{}, len(records))
	for _, record := range records {
		if _, exists := targets[record.Target]; !exists || !identities(record.Key) {
			return invalid()
		}
		if _, exists := seen[record]; exists {
			return invalid()
		}
		seen[record] = struct{}{}
	}
	return nil
}
func cloneInventory(input Inventory) Inventory {
	input.Targets = slices.Clone(input.Targets)
	input.Manifests = slices.Clone(input.Manifests)
	input.Unmanaged = slices.Clone(input.Unmanaged)
	for i := range input.Manifests {
		input.Manifests[i] = cloneManifest(input.Manifests[i])
	}
	return input
}
func cloneReceipt(receipt InventoryReceipt) InventoryReceipt {
	receipt.Targets = slices.Clone(receipt.Targets)
	receipt.Imported = slices.Clone(receipt.Imported)
	receipt.Unmanaged = slices.Clone(receipt.Unmanaged)
	receipt.Missing = slices.Clone(receipt.Missing)
	return receipt
}

func inventoryTargetSet(names []string) (map[string]struct{}, error) {
	if len(names) == 0 {
		return nil, invalid()
	}
	targets := make(map[string]struct{}, len(names))
	for _, target := range names {
		if !identities(target) {
			return nil, invalid()
		}
		if _, exists := targets[target]; exists {
			return nil, invalid()
		}
		targets[target] = struct{}{}
	}
	return targets, nil
}
func inventoryReplay(snapshot Snapshot, input Inventory, fingerprint string) (InventoryReceipt, bool, error) {
	for _, receipt := range snapshot.Inventories {
		if receipt.Kind != input.Kind || receipt.Watermark != input.Watermark {
			continue
		}
		if receipt.Fingerprint != fingerprint {
			return InventoryReceipt{}, false, ErrIdempotencyConflict
		}
		return cloneReceipt(receipt), true, nil
	}
	return InventoryReceipt{}, false, nil
}
func mergeInventorySource(snapshot *Snapshot, manifest Manifest) error {
	current := activePublication(*snapshot, manifest.Identity.Source)
	if existing := manifestIndex(*snapshot, manifest.ID); existing >= 0 {
		previous := snapshot.Manifests[existing]
		state, err := confirmedState(previous.State, previous.Checkpoint)
		if err != nil || !afterPublished(state) || !samePublishedManifest(previous, manifest) ||
			current != manifest.ID {
			return ErrConflict
		}
		return nil
	}
	if current != manifest.ExpectedPublication || overlappingInventory(*snapshot, manifest) {
		return ErrConflict
	}
	for _, previous := range snapshot.Manifests {
		if previous.Key == manifest.Key {
			return ErrIdempotencyConflict
		}
	}
	snapshot.Manifests = append(snapshot.Manifests, cloneManifest(manifest))
	replacePublication(snapshot, manifest.Identity.Source, manifest.ID)
	return nil
}
func samePublishedManifest(first, second Manifest) bool {
	first.State, first.Checkpoint = Published, ""
	second.State, second.Checkpoint = Published, ""
	a, err := json.Marshal(first)
	if err != nil {
		return false
	}
	b, err := json.Marshal(second)
	return err == nil && slices.Equal(a, b)
}
func missingInventorySources(snapshot Snapshot, incoming map[string]struct{}) []Publication {
	var missing []Publication
	for _, publication := range snapshot.Publications {
		if _, exists := incoming[publication.Source]; exists {
			continue
		}
		manifest := snapshot.Manifests[manifestIndex(snapshot, publication.Manifest)]
		if !manifest.Tombstone {
			missing = append(missing, publication)
		}
	}
	return missing
}
