package retrieval

import (
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/observation"
)

// observationCompletion observes only the gated result's cardinality. Usage is
// unavailable at this generic BYOT boundary; documents and errors stay private.
func observationCompletion[TMeta any](err error, rs ResultSet[TMeta]) observation.Completion {
	if access.IsProtectionFailure(err) {
		completion := observation.Finish(err, observation.Count{Known: true, Value: 0})
		if completion.Outcome != observation.OutcomeCanceled && completion.Outcome != observation.OutcomeUnsupported {
			completion.Error = observation.ErrorProtection
			completion.Outcome = observation.OutcomeFailed
		}
		return completion
	}
	count := observation.Count{Known: false, Value: 0}
	if rs != nil {
		length := rs.Len()
		if length >= 0 {
			count = observation.Count{Known: true, Value: observationCardinality(length)}
		}
	}
	completion := observation.Finish(err, count)
	return completion
}

func branchOrdinals(length int) []int {
	indices := make([]int, length)
	for i := range indices {
		indices[i] = i
	}
	return indices
}

// observationCardinality converts only a validated nonnegative cardinality.
func observationCardinality(value int) uint64 {
	if value < 0 {
		return 0
	}
	return uint64(value)
}
func observationOrdinal(index int) uint64 { return observationCardinality(index) + 1 }
