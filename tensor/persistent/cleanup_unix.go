//go:build darwin || linux

package persistent

import (
	"context"
	"errors"
	"os"
	"path/filepath"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/internal/durablefs"
	"github.com/skosovsky/ragy/lifecycle"
)

// Cleanup removes one retired manifest directory, never a source-wide prefix.
// The durable rename makes retained reads unavailable before physical deletion.
func (a *Adapter[TMeta]) Cleanup(
	ctx context.Context,
	request lifecycle.CleanupRequest,
) (lifecycle.CleanupState, error) {
	if a == nil {
		return lifecycle.CleanupUnknown, ragy.ErrInvalidArgument
	}
	lock, err := durablefs.Lock(ctx, filepath.Join(a.root, "target.lock"), true)
	if err != nil {
		return lifecycle.CleanupUnknown, err
	}
	defer func() { _ = lock.Close() }()
	if err = a.checkCleanup(ctx, request, true); err != nil {
		return lifecycle.CleanupUnknown, err
	}
	trash := a.trash(request.Retired.ID)
	if err = a.retireDirectory(ctx, request, trash); err != nil {
		return lifecycle.CleanupUnknown, err
	}
	if err = ctx.Err(); err != nil {
		return lifecycle.CleanupUnknown, err
	}
	if err = os.RemoveAll(a.staging(request.Retired.ID)); err != nil {
		return lifecycle.CleanupUnknown, err
	}
	if err = os.RemoveAll(trash); err != nil {
		return lifecycle.CleanupUnknown, err
	}
	if err = durablefs.SyncDirectory(a.root); err != nil {
		return lifecycle.CleanupUnknown, err
	}
	if err = ctx.Err(); err != nil {
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
	lock, err := durablefs.Lock(ctx, filepath.Join(a.root, "target.lock"), false)
	if err != nil {
		return lifecycle.CleanupUnknown, err
	}
	defer func() { _ = lock.Close() }()
	if err = a.checkCleanup(ctx, request, false); err != nil {
		return lifecycle.CleanupUnknown, err
	}
	for _, path := range []string{a.path(request.Retired.ID), a.trash(request.Retired.ID), a.staging(request.Retired.ID)} {
		if _, err = os.Stat(path); err == nil {
			return lifecycle.CleanupWaiting, nil
		}
		if !errors.Is(err, os.ErrNotExist) {
			return lifecycle.CleanupUnknown, err
		}
	}
	if err = ctx.Err(); err != nil {
		return lifecycle.CleanupUnknown, err
	}
	return lifecycle.CleanupDone, nil
}

func (a *Adapter[TMeta]) trash(id string) string {
	return filepath.Join(a.root, ".retired-"+digest([]byte(id)))
}

func (a *Adapter[TMeta]) checkCleanup(ctx context.Context, request lifecycle.CleanupRequest, dispatch bool) error {
	if request.Target != a.config.Target || request.Owner.Identity.Namespace != a.config.Namespace ||
		request.Retired.Identity.Namespace != a.config.Namespace ||
		request.Owner.ID == request.Retired.ID {
		return ragy.ErrInvalidArgument
	}
	snapshot, err := a.config.Store.Load(ctx, a.config.Namespace)
	if err != nil {
		return err
	}
	if snapshot.Namespace != a.config.Namespace || snapshot.Validate() != nil {
		return ragy.ErrProtocol
	}
	if !cleanupManifests(snapshot, request) {
		return ragy.ErrProtocol
	}
	if err = cleanupPublication(snapshot, request); err != nil {
		return err
	}
	if !cleanupRegistered(snapshot, request, dispatch) {
		return ragy.ErrProtocol
	}
	return ctx.Err()
}

func catalogRequestInventory(manifest lifecycle.Manifest, request lifecycle.CleanupRequest) bool {
	return lifecycle.SameTargetInventory(manifest, request.Retired, request.Target)
}

func (a *Adapter[TMeta]) retireDirectory(ctx context.Context, request lifecycle.CleanupRequest, trash string) error {
	_, err := os.Stat(a.path(request.Retired.ID))
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	entry, err := a.readCatalog(ctx, request.Retired.ID)
	if err != nil {
		return err
	}
	if entry.Identity != request.Retired.Identity || !catalogInventory(entry, request.Retired, a.config.Target) {
		return ragy.ErrProtocol
	}
	if _, err = os.Stat(trash); err == nil {
		return ragy.ErrProtocol
	}
	if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if err = ctx.Err(); err != nil {
		return err
	}
	if err = os.Rename(a.path(request.Retired.ID), trash); err != nil {
		return err
	}
	return durablefs.SyncDirectory(a.root)
}

func cleanupManifests(snapshot lifecycle.Snapshot, request lifecycle.CleanupRequest) bool {
	ownerFound, retiredFound := false, false
	for _, manifest := range snapshot.Manifests {
		if manifest.ID == request.Owner.ID && manifest.Identity == request.Owner.Identity {
			ownerFound = true
		}
		if manifest.ID == request.Retired.ID && manifest.Identity == request.Retired.Identity &&
			catalogRequestInventory(manifest, request) {
			retiredFound = true
		}
	}
	return ownerFound && retiredFound
}
func cleanupPublication(snapshot lifecycle.Snapshot, request lifecycle.CleanupRequest) error {
	for _, publication := range snapshot.Publications {
		if publication.Source != request.Retired.Identity.Source {
			continue
		}
		if publication.Manifest != request.ActivePublication || publication.Manifest == request.Retired.ID {
			return lifecycle.ErrConflict
		}
		return nil
	}
	return lifecycle.ErrConflict
}
func cleanupRegistered(snapshot lifecycle.Snapshot, request lifecycle.CleanupRequest, dispatch bool) bool {
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
