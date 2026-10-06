package managed

import (
	"context"
	"slices"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/lifecycle"
)

type inventoryObserver[TMeta any] struct {
	adapter    *Adapter[TMeta]
	maxEntries int
	maxRecords int
}

// InventoryObserver configures explicit bounds for observation of this retained
// in-process target. Missing data after restart is never reconstructed from a ledger.
func (a *Adapter[TMeta]) InventoryObserver(maxEntries, maxRecords int) (lifecycle.InventoryObserver, error) {
	if a == nil || maxEntries <= 0 || maxRecords <= 0 {
		return nil, ragy.ErrInvalidArgument
	}
	return &inventoryObserver[TMeta]{adapter: a, maxEntries: maxEntries, maxRecords: maxRecords}, nil
}
func (o *inventoryObserver[TMeta]) InventoryTarget() string { return o.adapter.config.Target }

// ObserveInventory holds the mutation fence through the synchronous next callback.
// Opaque manifest/host keys remain outside imported ownership and physical cleanup.
func (o *inventoryObserver[TMeta]) ObserveInventory(
	ctx context.Context,
	input lifecycle.Inventory,
	next func() error,
) error {
	a := o.adapter
	if next == nil || input.Namespace != a.config.Namespace || !slices.Contains(input.Targets, a.config.Target) {
		return ragy.ErrInvalidArgument
	}
	if err := input.Validate(); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if !a.mu.TryRLock() {
		return lifecycle.ErrConflict
	}
	defer a.mu.RUnlock()
	if err := o.verify(ctx, input); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if callbackErr := next(); callbackErr != nil {
		return callbackErr
	}
	return ctx.Err()
}

func (o *inventoryObserver[TMeta]) verify(ctx context.Context, input lifecycle.Inventory) error {
	a := o.adapter
	if len(a.versions) > o.maxEntries || len(a.bases) > o.maxEntries-len(a.versions) {
		return ragy.ErrUnavailable
	}
	keys := make(map[string]struct{}, len(a.versions)+len(a.bases))
	for id := range a.versions {
		keys["manifest:"+id] = struct{}{}
	}
	for id := range a.bases {
		keys["host:"+id] = struct{}{}
	}
	records := 0
	for _, manifest := range input.Manifests {
		if err := ctx.Err(); err != nil {
			return err
		}
		retained, exists := a.versions[manifest.ID]
		if !exists {
			return ragy.ErrUnavailable
		}
		if retained.manifest.ID != manifest.ID || retained.manifest.Identity != manifest.Identity ||
			retained.manifest.Payload != manifest.Payload ||
			!lifecycle.SameTargetInventory(retained.manifest, manifest, a.config.Target) {
			return ragy.ErrProtocol
		}
		count := len(retained.nodes) + len(retained.edges)
		if count > o.maxRecords-records {
			return ragy.ErrUnavailable
		}
		records += count
		if err := a.verifyRetainedRecords(retained); err != nil {
			return err
		}
		key := "manifest:" + manifest.ID
		if _, exists = keys[key]; !exists {
			return ragy.ErrProtocol
		}
		delete(keys, key)
	}
	return verifyOpaqueKeys(input, a.config.Target, keys)
}

func verifyOpaqueKeys(input lifecycle.Inventory, target string, keys map[string]struct{}) error {
	for _, record := range input.Unmanaged {
		if record.Target != target {
			continue
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

func (a *Adapter[TMeta]) verifyRetainedRecords(retained version[TMeta]) error {
	refs := inventory(retained.manifest, a.config.Target)
	if refs == nil || len(refs) != len(retained.nodes)+len(retained.edges) {
		return ragy.ErrProtocol
	}
	for _, node := range retained.nodes {
		if err := consumeReference(refs, node.record.Reference, node.record.Value.ID, "graph-node"); err != nil {
			return err
		}
	}
	for _, edge := range retained.edges {
		if err := consumeReference(refs, edge.record.Reference, edge.record.Value.ID, "graph-edge"); err != nil {
			return err
		}
	}
	return nil
}
