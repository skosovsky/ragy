package main

import (
	"github.com/skosovsky/ragy/evidence"
	"github.com/skosovsky/ragy/source"
)

const supportStage = "delivered-source-supports"

// Offline verification checks association, not producer authenticity or current authorization.
func validateEvidence(sample observation) error {
	if len(sample.Evidence) == 0 {
		return nil
	}
	record, err := evidence.Decode(sample.Evidence)
	if err != nil {
		return errInvalid
	}
	snapshot, err := record.Snapshot()
	outcome, reason := supportOutcome(sample)
	if err != nil || !observedText(snapshot.RetrievalID, sample.Query+"/"+sample.Profile) ||
		!observedText(snapshot.Scope, sample.Scope) || !observedText(snapshot.Publication, sample.Publication) ||
		!observedText(
			snapshot.Recipe,
			sample.Configuration,
		) || snapshot.Outcome != outcome || snapshot.Reason != reason {
		return errInvalid
	}
	found := false
	for _, stage := range snapshot.Stages {
		if !observedText(stage.Name, supportStage) {
			continue
		}
		if found || stage.Status != evidence.StageObserved || stage.HitsState != evidence.Observed ||
			len(stage.Hits) != len(sample.Supports) {
			return errInvalid
		}
		found = true
		for i, hit := range stage.Hits {
			if !matchesHit(hit, sample.Supports[i], i+1) {
				return errInvalid
			}
		}
	}
	if !found {
		return errInvalid
	}
	return nil
}
func matchesHit(hit evidence.WireHit, ref source.Reference, rank int) bool {
	if !observedText(hit.ID, ref.Source) || hit.Rank.State != evidence.Observed || hit.Rank.Value == nil {
		return false
	}
	if *hit.Rank.Value != float64(rank) || hit.SourcesState != evidence.Observed || len(hit.Sources) != 1 {
		return false
	}
	return matchesReference(hit.Sources[0], ref)
}
func observedText(value evidence.Text, want string) bool {
	return value.State == evidence.Observed && value.Value != nil && *value.Value == want
}
func matchesReference(value evidence.Source, ref source.Reference) bool {
	return observedText(value.Namespace, ref.Namespace) && observedText(value.ID, ref.Source) &&
		observedText(value.Revision, ref.Revision) && observedText(value.Transformation, ref.Transformation) &&
		observedText(value.Artifact, ref.Artifact) && observedText(value.Representation, ref.Representation)
}

func supportOutcome(sample observation) (evidence.Outcome, evidence.Reason) {
	if sample.Failed {
		return evidence.Failed, evidence.TargetFailure
	}
	switch sample.Outcome {
	case "partial":
		return evidence.Partial, evidence.Budget
	case "insufficient":
		return evidence.Insufficient, evidence.MissingEvidence
	default:
		if len(sample.Supports) == 0 {
			return evidence.CompleteEmpty, evidence.NoReason
		}
		return evidence.Complete, evidence.NoReason
	}
}
