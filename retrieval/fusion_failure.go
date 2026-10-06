package retrieval

import (
	"context"
	"errors"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/internal/nilvalue"
)

// FusionFailureError retains admitted observations when fusion fails ordinarily.
// Observations are diagnostic input, never a substitute for the returned result.
// Protection and cancellation do not expose observations.
type FusionFailureError[TMeta any] struct {
	Cause error

	observations []ResultSet[TMeta]
}

func (e *FusionFailureError[TMeta]) Error() string { return fmt.Sprintf("fusion failed: %v", e.Cause) }
func (e *FusionFailureError[TMeta]) Unwrap() error { return e.Cause }

// Observations returns independently owned ragy containers in branch order.
// Arbitrary BYOT metadata remains host-owned and stable.
func (e *FusionFailureError[TMeta]) Observations() []ResultSet[TMeta] {
	sets := make([]ResultSet[TMeta], 0, len(e.observations))
	for _, set := range e.observations {
		sets = append(sets, NewResultSet(set.Documents(), ResolverFor(set)))
	}
	return sets
}

// DegradingMerger explicitly selects a fallback merger on ordinary fusion error.
// Both causes remain observable; a fallback result is delivered with an error.
// It never retries providers or degrades protection/cancellation failures.
// ScoreMerger is appropriate only for host-attested comparable score scales.
type DegradingMerger[TMeta any] struct {
	Primary  ResultMerger[TMeta]
	Fallback ResultMerger[TMeta]
}

// Merge invokes Fallback only when Primary fails ordinarily and context is live.
func (m DegradingMerger[TMeta]) Merge(ctx context.Context, sets ...ResultSet[TMeta]) (ResultSet[TMeta], error) {
	empty := NewResultSet[TMeta](nil, nil)
	if err := m.validate(); err != nil {
		return empty, err
	}
	if err := ctx.Err(); err != nil {
		return empty, err
	}
	result, primaryErr := m.Primary.Merge(ctx, sets...)
	if err := errors.Join(primaryErr, ctx.Err()); stopsDegradation(err) {
		return empty, suppressErrorPayload(err, empty)
	}
	if primaryErr == nil {
		return result, nil
	}
	result, fallbackErr := m.Fallback.Merge(ctx, sets...)
	err := errors.Join(primaryErr, fallbackErr, ctx.Err())
	if stopsDegradation(err) {
		return empty, suppressErrorPayload(err, empty)
	}
	return result, err
}

func stopsDegradation(err error) bool {
	return access.IsProtectionFailure(err) || errors.Is(err, context.Canceled) ||
		errors.Is(err, context.DeadlineExceeded)
}

func (m DegradingMerger[TMeta]) validate() error {
	if nilvalue.IsNil(m.Primary) || nilvalue.IsNil(m.Fallback) {
		return fmt.Errorf("%w: degradation merger ports", ragy.ErrInvalidArgument)
	}
	for _, port := range []ResultMerger[TMeta]{m.Primary, m.Fallback} {
		if _, err := resolveAggregateMerger(port, DefaultResolver[TMeta](nil)); err != nil {
			return err
		}
	}
	return nil
}
