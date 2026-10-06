package managed

import (
	"context"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/lifecycle"
)

// Cleanup releases only the retired source inventory. Shared facts remain in
// other retained inventories; no node/edge cascade by logical ID is performed.
func (a *Adapter[TMeta]) Cleanup(
	ctx context.Context,
	request lifecycle.CleanupRequest,
) (lifecycle.CleanupState, error) {
	if a == nil {
		return lifecycle.CleanupUnknown, ragy.ErrInvalidArgument
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := a.checkCleanup(ctx, request, true); err != nil {
		return lifecycle.CleanupUnknown, err
	}
	delete(a.versions, request.Retired.ID)
	if err := ctx.Err(); err != nil {
		return lifecycle.CleanupUnknown, err
	}
	return lifecycle.CleanupDone, nil
}

func (a *Adapter[TMeta]) InspectCleanup(
	ctx context.Context,
	request lifecycle.CleanupRequest,
) (lifecycle.CleanupState, error) {
	if a == nil {
		return lifecycle.CleanupUnknown, ragy.ErrInvalidArgument
	}
	a.mu.RLock()
	defer a.mu.RUnlock()
	if err := a.checkCleanup(ctx, request, false); err != nil {
		return lifecycle.CleanupUnknown, err
	}
	if _, exists := a.versions[request.Retired.ID]; exists {
		return lifecycle.CleanupWaiting, nil
	}
	return lifecycle.CleanupDone, nil
}
func (a *Adapter[TMeta]) checkCleanup(ctx context.Context, request lifecycle.CleanupRequest, dispatch bool) error {
	if request.Target != a.config.Target || request.Retired.Identity.Namespace != a.config.Namespace ||
		request.Owner.Identity.Namespace != a.config.Namespace ||
		request.Retired.ID == request.Owner.ID {
		return ragy.ErrInvalidArgument
	}
	snapshot, err := a.config.Store.Load(ctx, a.config.Namespace)
	if err != nil {
		return err
	}
	if snapshot.Namespace != a.config.Namespace || snapshot.Validate() != nil {
		return ragy.ErrProtocol
	}
	if !cleanupInventory(snapshot, request) {
		return ragy.ErrProtocol
	}
	if !cleanupRegistration(snapshot, request, dispatch) {
		return ragy.ErrProtocol
	}
	for _, publication := range snapshot.Publications {
		if publication.Source != request.Retired.Identity.Source {
			continue
		}
		if publication.Manifest != request.ActivePublication || publication.Manifest == request.Retired.ID {
			return lifecycle.ErrConflict
		}
		return ctx.Err()
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
