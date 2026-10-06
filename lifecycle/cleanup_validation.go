package lifecycle

func (s Snapshot) validateCleanups(manifests map[string]Manifest) error {
	owners := make(map[string]struct{}, len(s.Cleanups))
	for _, job := range s.Cleanups {
		if _, exists := owners[job.Owner]; exists {
			return invalid()
		}
		owners[job.Owner] = struct{}{}
		owner, exists := manifests[job.Owner]
		if !exists || !job.StartedAt.Equal(owner.PublishedAt) || !job.Deadline.After(job.StartedAt) {
			return invalid()
		}
		state, err := confirmedState(owner.State, owner.Checkpoint)
		if err != nil || (state != CleanupPending && state != Complete) || job.Complete != (state == Complete) {
			return invalid()
		}
		if err = job.validateItems(manifests, owner, s.Publications); err != nil {
			return err
		}
	}
	return nil
}
func (j CleanupJob) validateItems(manifests map[string]Manifest, owner Manifest, publications []Publication) error {
	ancestors, err := retirementAncestry(manifests, owner)
	if err != nil {
		return err
	}
	seen := make(map[[2]string]struct{}, len(j.Items))
	complete := true
	for _, item := range j.Items {
		key := [2]string{item.Manifest, item.Target}
		if _, exists := seen[key]; exists {
			return invalid()
		}
		seen[key] = struct{}{}
		retired, exists := manifests[item.Manifest]
		if !exists || !retirable(retired, owner, ancestors) {
			return invalid()
		}
		if err = item.validate(retired, publications, j); err != nil {
			return err
		}
		if item.State != CleanupDone {
			complete = false
		}
	}
	for _, retired := range manifests {
		if !retirable(retired, owner, ancestors) {
			continue
		}
		for _, target := range retired.Targets {
			if _, exists := seen[[2]string{retired.ID, target.Name}]; !exists {
				return invalid()
			}
		}
	}
	if j.Complete != complete {
		return invalid()
	}
	return nil
}
func (i RetiredTarget) validate(retired Manifest, publications []Publication, job CleanupJob) error {
	for _, publication := range publications {
		if publication.Manifest == retired.ID {
			return invalid()
		}
	}
	found := false
	for _, target := range retired.Targets {
		if target.Name == i.Target {
			found = true
		}
	}
	if !found || i.NextAt.Before(job.StartedAt) {
		return invalid()
	}
	switch i.State {
	case CleanupWaiting:
		return nil
	case CleanupUnknown, CleanupDone:
		if i.Attempts == 0 {
			return invalid()
		}
		return nil
	default:
		return invalid()
	}
}
func retirementAncestry(manifests map[string]Manifest, owner Manifest) (map[string]struct{}, error) {
	ancestors := map[string]struct{}{"": {}}
	next := owner.ExpectedPublication
	for next != "" {
		if _, exists := ancestors[next]; exists {
			return nil, invalid()
		}
		retired, exists := manifests[next]
		if !exists || retired.Identity.Source != owner.Identity.Source || retired.ID == owner.ID {
			return nil, invalid()
		}
		state, err := confirmedState(retired.State, retired.Checkpoint)
		if err != nil || !afterPublished(state) {
			return nil, invalid()
		}
		ancestors[next] = struct{}{}
		next = retired.ExpectedPublication
	}
	return ancestors, nil
}
func retirable(retired, owner Manifest, ancestors map[string]struct{}) bool {
	if retired.ID == owner.ID || retired.Identity.Source != owner.Identity.Source {
		return false
	}
	if _, exists := ancestors[retired.ID]; exists {
		return true
	}
	state, err := confirmedState(retired.State, retired.Checkpoint)
	if err != nil || afterPublished(state) {
		return false
	}
	_, exists := ancestors[retired.ExpectedPublication]
	return exists
}
