package lifecycle

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"slices"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
)

// CapturePublication fixes the active inventory once before query fan-out. It
// requires each requested target ready and omits tombstoned sources. An empty
// namespace/tombstone-only inventory remains pinned complete-empty. Partial target
// capture is not implicitly enabled by a partial source publication.
func CapturePublication(
	ctx context.Context,
	store Store,
	namespace string,
	targets []string,
) (access.Publication, error) {
	return capturePublication(ctx, store, namespace, targets, false)
}

// CapturePartialPublication explicitly excludes an entire target branch when any
// active source lacks its ready checkpoint. Coverage travels in the immutable pin.
func CapturePartialPublication(
	ctx context.Context, store Store, namespace string, targets []string,
) (access.Publication, error) {
	return capturePublication(ctx, store, namespace, targets, true)
}

func capturePublication(
	ctx context.Context, store Store, namespace string, targets []string, partial bool,
) (access.Publication, error) {
	if nilPort(store) || !identities(namespace) || len(targets) == 0 {
		return access.Publication{}, ragy.ErrInvalidArgument
	}
	targets = slices.Clone(targets)
	requested := make(map[string]struct{}, len(targets))
	for _, target := range targets {
		if !identities(target) {
			return access.Publication{}, ragy.ErrInvalidArgument
		}
		if _, exists := requested[target]; exists {
			return access.Publication{}, ragy.ErrInvalidArgument
		}
		requested[target] = struct{}{}
	}
	if err := ctx.Err(); err != nil {
		return access.Publication{}, err
	}
	snapshot, err := store.Load(ctx, namespace)
	if err != nil {
		return access.Publication{}, err
	}
	return publicationFromSnapshot(ctx, snapshot, targets, partial)
}

func publicationFromSnapshot(
	ctx context.Context,
	snapshot Snapshot,
	targets []string,
	partial bool,
) (access.Publication, error) {
	namespace := snapshot.Namespace
	if !identities(namespace) || len(targets) == 0 {
		return access.Publication{}, ragy.ErrInvalidArgument
	}
	targets = slices.Clone(targets)
	requested, err := inventoryTargetSet(targets)
	if err != nil {
		return access.Publication{}, err
	}
	if snapshot.Namespace != namespace || snapshot.Validate() != nil {
		return access.Publication{}, ragy.ErrProtocol
	}
	var excluded []string
	if partial {
		excluded = excludeUnavailableTargets(snapshot, targets, requested)
	}
	inventory, err := publicationInventory(snapshot, requested)
	if err != nil {
		return access.Publication{}, err
	}
	identity := struct {
		Namespace    string        `json:"namespace"`
		Publications []Publication `json:"publications"`
		Targets      []string      `json:"targets"`
	}{
		Namespace: namespace, Publications: slices.Clone(snapshot.Publications), Targets: slices.Clone(targets),
	}
	slices.SortFunc(
		identity.Publications,
		func(first, second Publication) int { return compareStrings(first.Source, second.Source) },
	)
	slices.Sort(identity.Targets)
	data, err := json.Marshal(identity)
	if err != nil {
		return access.Publication{}, ragy.ErrProtocol
	}
	hash := sha256.Sum256(data)
	if err = ctx.Err(); err != nil {
		return access.Publication{}, err
	}
	if len(excluded) > 0 {
		return access.PinPartialPublication(hex.EncodeToString(hash[:]), inventory, excluded)
	}
	return access.PinPublication(hex.EncodeToString(hash[:]), inventory)
}
func publicationInventory(snapshot Snapshot, requested map[string]struct{}) ([]access.TargetRevision, error) {
	var inventory []access.TargetRevision
	for _, publication := range snapshot.Publications {
		manifest := snapshot.Manifests[manifestIndex(snapshot, publication.Manifest)]
		if manifest.Tombstone {
			continue
		}
		for name := range requested {
			found := false
			for _, target := range manifest.Targets {
				if target.Name != name {
					continue
				}
				if target.State != TargetReady {
					return nil, ragy.ErrUnsupported
				}
				found = true
				inventory = append(inventory, access.TargetRevision{
					Target: name, Namespace: manifest.Identity.Namespace, Source: manifest.Identity.Source,
					Revision: manifest.Identity.Revision, Transformation: manifest.Identity.Transformation,
					AccessFingerprint: manifest.Identity.Access,
				})
			}
			if !found {
				return nil, ragy.ErrUnsupported
			}
		}
	}
	return inventory, nil
}
func compareStrings(first, second string) int {
	if first < second {
		return -1
	}
	if first > second {
		return 1
	}
	return 0
}

func excludeUnavailableTargets(snapshot Snapshot, targets []string, requested map[string]struct{}) []string {
	var excluded []string
	for _, name := range targets {
		if _, err := publicationInventory(snapshot, map[string]struct{}{name: {}}); err != nil {
			excluded = append(excluded, name)
			delete(requested, name)
		}
	}
	return excluded
}
