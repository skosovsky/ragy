package lifecycle

func (s Snapshot) validateInventories(manifests map[string]Manifest) error {
	seen := make(map[[2]string]struct{}, len(s.Inventories))
	for _, receipt := range s.Inventories {
		key := [2]string{string(receipt.Kind), receipt.Watermark}
		if _, exists := seen[key]; exists {
			return invalid()
		}
		seen[key] = struct{}{}
		if err := receipt.validate(manifests); err != nil {
			return err
		}
	}
	return nil
}
func (r InventoryReceipt) validate(manifests map[string]Manifest) error {
	if !inventoryProfile(r.Kind, r.Coverage) || !identities(r.Watermark, r.Fingerprint) || len(r.Targets) == 0 {
		return invalid()
	}
	targets, err := inventoryTargetSet(r.Targets)
	if err != nil {
		return err
	}
	if err := validateUnmanaged(r.Unmanaged, targets); err != nil {
		return err
	}
	imported := make(map[string]struct{}, len(r.Imported))
	for _, id := range r.Imported {
		if _, exists := manifests[id]; !exists {
			return invalid()
		}
		if _, exists := imported[id]; exists {
			return invalid()
		}
		imported[id] = struct{}{}
	}
	return r.validateMissing(manifests, imported)
}
func (r InventoryReceipt) validateMissing(manifests map[string]Manifest, imported map[string]struct{}) error {
	if r.Kind == DeltaInventory && len(r.Missing) != 0 {
		return invalid()
	}
	missing := make(map[string]struct{}, len(r.Missing))
	for _, publication := range r.Missing {
		manifest, exists := manifests[publication.Manifest]
		if !exists || manifest.Identity.Source != publication.Source || manifest.Tombstone {
			return invalid()
		}
		if _, exists := missing[publication.Source]; exists {
			return invalid()
		}
		if _, exists := imported[publication.Manifest]; exists {
			return invalid()
		}
		missing[publication.Source] = struct{}{}
	}
	return nil
}
