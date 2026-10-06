package lifecycle

import (
	"errors"

	"github.com/skosovsky/ragy/observation"
)

func lifecycleCompletion(err error) observation.Completion {
	completion := observation.Finish(err, observation.Count{Known: false, Value: 0})
	if errors.Is(err, ErrOutcomeUnknown) && completion.Outcome == observation.OutcomeFailed {
		completion.Outcome = observation.OutcomePartial
	}
	return completion
}

func manifestCompletion(manifest Manifest, err error) observation.Completion {
	completion := lifecycleCompletion(err)
	if err == nil && manifest.Partial {
		completion.Outcome = observation.OutcomePartial
	}
	return completion
}

func stageCompletion(result StageResult, err error) observation.Completion {
	completion := lifecycleCompletion(err)
	if err != nil {
		return completion
	}
	switch result.State {
	case TargetPending, TargetUnknown:
		completion.Outcome = observation.OutcomePartial
		if result.State == TargetUnknown {
			completion.Error = observation.ErrorUnknown
		}
	case TargetFailed:
		completion.Outcome = observation.OutcomeFailed
		completion.Error = observation.ErrorUnavailable
	case TargetReady:
	default:
		completion.Outcome = observation.OutcomeFailed
		completion.Error = observation.ErrorProtocol
	}
	return completion
}

func cleanupCompletion(job CleanupJob, err error) observation.Completion {
	completion := lifecycleCompletion(err)
	if err == nil && !job.Complete {
		completion.Outcome = observation.OutcomePartial
	}
	return completion
}

func cleanupStateCompletion(state CleanupState, err error) observation.Completion {
	completion := lifecycleCompletion(err)
	if err != nil {
		return completion
	}
	switch state {
	case CleanupWaiting, CleanupUnknown:
		completion.Outcome = observation.OutcomePartial
		if state == CleanupUnknown {
			completion.Error = observation.ErrorUnknown
		}
	case CleanupDone:
	default:
		completion.Outcome = observation.OutcomeFailed
		completion.Error = observation.ErrorProtocol
	}
	return completion
}

func reuseCompletion(decision ReuseDecision, err error) observation.Completion {
	completion := lifecycleCompletion(err)
	if err != nil {
		return completion
	}
	switch decision.Reason {
	case ReuseAbsent:
		completion.Outcome = observation.OutcomeEmpty
	case ReuseChanged:
		completion.Outcome = observation.OutcomeSkipped
	case ReuseIncomplete:
		completion.Outcome = observation.OutcomePartial
	case ReuseConfirmed:
	default:
		completion.Outcome = observation.OutcomeUnknown
	}
	return completion
}

func inventoryCompletion(receipt InventoryReceipt, err error) observation.Completion {
	completion := lifecycleCompletion(err)
	if err == nil && receipt.Coverage == PartialInventory {
		completion.Outcome = observation.OutcomePartial
	}
	return completion
}
