//go:build darwin || linux

package persistent

import (
	"context"
	"path/filepath"
	"slices"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/internal/durablefs"
	"github.com/skosovsky/ragy/lifecycle"
)

func (a *Adapter[TMeta]) InventoryTarget() string {
	if a == nil {
		return ""
	}
	return a.config.Target
}

// ObserveInventory holds the same cross-process fence as Stage/Cleanup. Unknown
// entries are opaque keys: no source identity, payload read or deletion is inferred.
func (a *Adapter[TMeta]) ObserveInventory(ctx context.Context, input lifecycle.Inventory, next func() error) error {
	if a == nil || next == nil || input.Namespace != a.config.Namespace ||
		!slices.Contains(input.Targets, a.config.Target) {
		return ragy.ErrInvalidArgument
	}
	if err := input.Validate(); err != nil {
		return err
	}
	lock, err := durablefs.Lock(ctx, filepath.Join(a.root, "target.lock"), false)
	if err != nil {
		return err
	}
	defer func() { _ = lock.Close() }()
	keys, err := durablefs.InventoryKeys(ctx, a.root, "target.lock", a.config.MaxRecords)
	if err != nil {
		return err
	}
	if err = a.verifyInventory(ctx, input, keys); err != nil {
		return err
	}
	if err = ctx.Err(); err != nil {
		return err
	}
	if callbackErr := next(); callbackErr != nil {
		return callbackErr
	}
	return ctx.Err()
}

func (a *Adapter[TMeta]) verifyInventory(
	ctx context.Context,
	input lifecycle.Inventory,
	keys map[string]struct{},
) error {
	verifiedRecords := 0
	for _, manifest := range input.Manifests {
		entry, err := a.readCatalog(ctx, manifest.ID)
		if err != nil {
			return err
		}
		if entry.Identity != manifest.Identity || entry.PayloadFingerprint != manifest.Payload ||
			!catalogInventory(entry, manifest, a.config.Target) {
			return ragy.ErrProtocol
		}
		if len(entry.Records) > a.config.MaxRecords-verifiedRecords {
			return ragy.ErrUnavailable
		}
		verifiedRecords += len(entry.Records)
		if err = a.verifyPayloads(ctx, entry); err != nil {
			return err
		}
		key := filepath.Base(a.path(manifest.ID))
		if _, exists := keys[key]; !exists {
			return ragy.ErrProtocol
		}
		delete(keys, key)
	}
	return verifyOpaqueInventory(input, keys, a.config.Target)
}

func verifyOpaqueInventory(input lifecycle.Inventory, keys map[string]struct{}, target string) error {
	for _, record := range input.Unmanaged {
		if record.Target != target {
			continue
		}
		if !durablefs.ValidInventoryKey(record.Key) {
			return ragy.ErrInvalidArgument
		}
		if _, exists := keys[record.Key]; !exists {
			return ragy.ErrProtocol
		}
		delete(keys, record.Key)
	}
	if input.Kind == lifecycle.CompleteInventory && len(keys) != 0 {
		return ragy.ErrProtocol
	}
	return nil
}
