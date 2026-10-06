package main

func validatePreparation(prep preparation, f fixture, execution string, cliProfile ...bool) error {
	if !validFingerprint(prep.ResolutionHistoryID) || len(prep.Extractions) != len(f.Sources) ||
		prep.MembershipGraphCalls != uint64(len(f.Communities)) {
		return errInvalid
	}
	seen := make(map[string]bool)
	for _, row := range prep.Extractions {
		exists := false
		for _, src := range f.Sources {
			if row.Reference == originalReference(src.ID) {
				exists = true
			}
		}
		if !exists || seen[row.Reference.Source] || row.Failed || row.Nanos <= 0 ||
			!validFingerprint(row.Configuration) {
			return errInvalid
		}
		if execution == liveExecution && !row.TransportCallsKnown && (len(cliProfile) == 0 || !cliProfile[0]) {
			return errInvalid
		}
		seen[row.Reference.Source] = true
	}
	return nil
}
func preparationBudgets(prep preparation) bool {
	for _, row := range prep.Extractions {
		if row.Failed || !row.UsageKnown || row.Nanos > int64(attemptDuration) || row.ModelCalls > 1 ||
			row.InputTokens > extractionInputTokens ||
			row.OutputTokens > extractionOutputTokens ||
			row.Cost > referenceCostCap {
			return false
		}
		if row.TransportCallsKnown && row.TransportCalls > row.ModelCalls {
			return false
		}
	}
	return true
}
