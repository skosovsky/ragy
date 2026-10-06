package recipe

import (
	"errors"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/observation"
	"github.com/skosovsky/ragy/recipe/budget"
)

func diagnosticCompletion(err error, count observation.Count) observation.Completion {
	completion := observation.Finish(err, count)
	if errors.Is(err, budget.ErrExhausted) || errors.Is(err, budget.ErrUnknownPrice) {
		completion.Outcome, completion.Error = observation.OutcomeExhausted, observation.ErrorResource
	} else if access.IsProtectionFailure(err) {
		completion.Error = observation.ErrorProtection
	}
	return completion
}

func modelCompletion(err error, usage Usage) observation.Completion {
	completion := diagnosticCompletion(err, observation.Count{Known: false, Value: 0})
	completion.Usage = observation.Usage{
		BilledUnits:  observation.Count{Known: false, Value: 0},
		InputTokens:  observation.Count{Known: usage.Known, Value: usage.Value.InputTokens},
		OutputTokens: observation.Count{Known: usage.Known, Value: usage.Value.OutputTokens},
	}
	// Host pricing cost is not a provider billed-unit counter.
	return completion
}

func attemptCompletion[TMeta any](output Result[TMeta], err error) observation.Completion {
	completion := diagnosticCompletion(err, observation.Count{Known: err == nil, Value: uint64(len(output.Selected))})
	if err != nil {
		return completion
	}
	switch output.Stop {
	case BudgetExhausted, PriceUnavailable:
		completion.Outcome, completion.Error = observation.OutcomeExhausted, observation.ErrorResource
	case DeadlineReached:
		completion.Outcome, completion.Error = observation.OutcomeCanceled, observation.ErrorDeadline
	case Assessed, StageFailure:
		if output.Outcome == Partial || output.Outcome == Insufficient && len(output.Selected) > 0 {
			completion.Outcome = observation.OutcomePartial
		}
	}
	return completion
}
