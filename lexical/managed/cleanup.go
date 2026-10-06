package managed

import (
	"context"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/lifecycle"
)

// Cleanup is idempotent exact-version deletion, serialized with Stage. Changed
// active publication is explicit conflict; no source or logical-ID cascade occurs.
func (a *Adapter[TMeta]) Cleanup(
	ctx context.Context,
	request lifecycle.CleanupRequest,
) (lifecycle.CleanupState, error) {
	if err := a.validateCleanup(ctx, request); err != nil {
		return lifecycle.CleanupUnknown, err
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := a.checkCleanupPublication(ctx, request, true); err != nil {
		return lifecycle.CleanupUnknown, err
	}
	key := keyForIdentity(request.Retired.Identity)
	previous, exists := a.versions[key]
	if exists && previous.manifest.ID != request.Retired.ID {
		return lifecycle.CleanupUnknown, lifecycle.ErrConflict
	}
	delete(a.versions, key)
	a.invalidateCacheLocked()
	if err := ctx.Err(); err != nil {
		return lifecycle.CleanupUnknown, err
	}
	return lifecycle.CleanupDone, nil
}

func (a *Adapter[TMeta]) InspectCleanup(
	ctx context.Context,
	request lifecycle.CleanupRequest,
) (lifecycle.CleanupState, error) {
	if err := a.validateCleanup(ctx, request); err != nil {
		return lifecycle.CleanupUnknown, err
	}
	a.mu.RLock()
	defer a.mu.RUnlock()
	if err := a.checkCleanupPublication(ctx, request, false); err != nil {
		return lifecycle.CleanupUnknown, err
	}
	previous, exists := a.versions[keyForIdentity(request.Retired.Identity)]
	if !exists {
		return lifecycle.CleanupDone, nil
	}
	if previous.manifest.ID != request.Retired.ID {
		return lifecycle.CleanupUnknown, lifecycle.ErrConflict
	}
	return lifecycle.CleanupWaiting, nil
}
func (a *Adapter[TMeta]) validateCleanup(ctx context.Context, request lifecycle.CleanupRequest) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if a == nil || request.Target != a.config.Target || request.Retired.Identity.Namespace != a.config.Namespace ||
		request.Owner.Identity.Namespace != a.config.Namespace || request.Owner.Identity.Source != request.Retired.Identity.Source ||
		request.Owner.ID == request.Retired.ID {
		return ragy.ErrInvalidArgument
	}
	if err := request.Retired.Validate(); err != nil {
		return err
	}
	return request.Owner.Validate()
}

func (a *Adapter[TMeta]) checkCleanupPublication(
	ctx context.Context,
	request lifecycle.CleanupRequest,
	dispatch bool,
) error {
	snapshot, err := a.config.Store.Load(ctx, a.config.Namespace)
	if err != nil {
		return err
	}
	if snapshot.Namespace != a.config.Namespace || snapshot.Validate() != nil {
		return ragy.ErrProtocol
	}
	if !cleanupInventory(snapshot, request) || !cleanupRegistration(snapshot, request, dispatch) {
		return ragy.ErrProtocol
	}
	for _, publication := range snapshot.Publications {
		if publication.Source == request.Retired.Identity.Source {
			if publication.Manifest != request.ActivePublication || publication.Manifest == request.Retired.ID {
				return lifecycle.ErrConflict
			}
			return ctx.Err()
		}
	}
	return lifecycle.ErrConflict
}

func cleanupInventory(snapshot lifecycle.Snapshot, request lifecycle.CleanupRequest) bool {
	owner, retired := false, false
	for _, manifest := range snapshot.Manifests {
		if manifest.ID == request.Owner.ID && manifest.Identity == request.Owner.Identity {
			owner = true
		}
		if manifest.ID == request.Retired.ID && manifest.Identity == request.Retired.Identity &&
			lifecycle.SameTargetInventory(manifest, request.Retired, request.Target) {
			retired = true
		}
	}
	return owner && retired
}
func cleanupRegistration(snapshot lifecycle.Snapshot, request lifecycle.CleanupRequest, dispatch bool) bool {
	for _, job := range snapshot.Cleanups {
		if job.Owner != request.Owner.ID {
			continue
		}
		for _, item := range job.Items {
			if item.Manifest == request.Retired.ID && item.Target == request.Target &&
				(!dispatch || item.State == lifecycle.CleanupUnknown) {
				return true
			}
		}
	}
	return false
}
