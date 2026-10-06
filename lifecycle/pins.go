package lifecycle

import (
	"context"
	"slices"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
)

// AcquirePublicationPin captures active publication metadata and registers its
// exact target revisions in the same namespace generation CAS. IDs are durable:
// a live ID replays its original publication, while a released ID stays retired.
// This protects metadata only; target availability and authorization are separate.
func AcquirePublicationPin(
	ctx context.Context,
	store Store,
	namespace, id string,
	targets []string,
) (access.Publication, error) {
	if ctx == nil || nilPort(store) || !identities(namespace, id) {
		return access.Publication{}, ragy.ErrInvalidArgument
	}
	if _, err := inventoryTargetSet(targets); err != nil {
		return access.Publication{}, err
	}
	profile := slices.Clone(targets)
	slices.Sort(profile)
	if err := ctx.Err(); err != nil {
		return access.Publication{}, err
	}
	snapshot, err := store.Load(ctx, namespace)
	if err != nil {
		return access.Publication{}, err
	}
	if snapshot.Namespace != namespace || snapshot.Validate() != nil {
		return access.Publication{}, ragy.ErrProtocol
	}
	if prior, found, replayErr := replayPublicationPin(ctx, snapshot, id, profile); found {
		return prior, replayErr
	}
	if err = admitPublicationPin(snapshot, profile); err != nil {
		return access.Publication{}, err
	}
	publication, err := publicationFromSnapshot(ctx, snapshot, profile, false)
	if err != nil {
		return access.Publication{}, err
	}
	snapshot.Pins = append(slices.Clone(snapshot.Pins), PublicationPin{
		ID:               id,
		Publication:      publication.Reference(),
		Targets:          publication.Targets(),
		RequestedTargets: profile,
		Released:         false,
	})
	if err := ctx.Err(); err != nil {
		return access.Publication{}, err
	}
	if _, err := store.CompareSwap(ctx, snapshot.Generation, snapshot); err != nil {
		return access.Publication{}, mutationError(err)
	}
	return publication, nil
}

func replayPublicationPin(
	ctx context.Context,
	snapshot Snapshot,
	id string,
	profile []string,
) (access.Publication, bool, error) {
	for _, prior := range snapshot.Pins {
		if prior.ID != id {
			continue
		}
		if prior.Released {
			return access.Publication{}, true, ErrRetired
		}
		requested := slices.Clone(prior.RequestedTargets)
		slices.Sort(requested)
		if !slices.Equal(requested, profile) {
			return access.Publication{}, true, ErrIdempotencyConflict
		}
		if err := ctx.Err(); err != nil {
			return access.Publication{}, true, err
		}
		publication, err := access.PinPublication(prior.Publication, prior.Targets)
		return publication, true, err
	}
	return access.Publication{}, false, nil
}

func admitPublicationPin(snapshot Snapshot, profile []string) error {
	for _, publication := range snapshot.Publications {
		manifest := snapshot.Manifests[manifestIndex(snapshot, publication.Manifest)]
		if manifest.Retired {
			return ErrRetired
		}
		if manifest.State == Unknown {
			return ragy.ErrUnsupported
		}
		if manifest.Tombstone {
			continue
		}
		for _, target := range profile {
			if cleanedInventory(snapshot, manifest.ID, target) {
				return ragy.ErrUnsupported
			}
		}
	}
	return nil
}

// ReleasePublicationPin durably ends metadata protection. Releasing an already
// released ID is a read-only success; the ID can never be acquired again.
func ReleasePublicationPin(ctx context.Context, store Store, namespace, id string) error {
	if ctx == nil || nilPort(store) || !identities(namespace, id) {
		return ragy.ErrInvalidArgument
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	snapshot, err := store.Load(ctx, namespace)
	if err != nil {
		return err
	}
	if snapshot.Namespace != namespace || snapshot.Validate() != nil {
		return ragy.ErrProtocol
	}
	for i, pin := range snapshot.Pins {
		if pin.ID != id {
			continue
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if pin.Released {
			return nil
		}
		snapshot.Pins = slices.Clone(snapshot.Pins)
		snapshot.Pins[i].Released = true
		if _, err := store.CompareSwap(ctx, snapshot.Generation, snapshot); err != nil {
			return mutationError(err)
		}
		return nil
	}
	return ragy.ErrUnavailable
}
