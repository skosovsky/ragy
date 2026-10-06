package retrieval

import (
	"errors"
	"fmt"
	"strings"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/internal/nilvalue"
)

// PartialFailureError reports failed aggregate branches. Result is a synchronized
// diagnostic view; the separately returned ResultSet is authoritative, including
// an empty set. Error-carried documents never authorize payload resurrection.
type PartialFailureError[TMeta any] struct {
	Errors []error
	Result ResultSet[TMeta]
}

func (e *PartialFailureError[TMeta]) Error() string {
	if e == nil {
		return "aggregate partial failure"
	}
	msgs := make([]string, 0, len(e.Errors))
	for _, err := range e.Errors {
		if err != nil {
			msgs = append(msgs, err.Error())
		}
	}
	return fmt.Sprintf(
		"aggregate partial failure (%d child error(s)): %s",
		len(e.Errors),
		strings.Join(msgs, "; "),
	)
}

// Unwrap returns child errors for [errors.Is] / [errors.As] traversal.
func (e *PartialFailureError[TMeta]) Unwrap() []error {
	if e == nil {
		return nil
	}
	return e.Errors
}

// syncPartialFailureResult updates PartialFailureError.Result to match the post-processed set.
func syncPartialFailureResult[TMeta any](err error, rs ResultSet[TMeta]) error {
	partial, ok := AsPartialFailure[TMeta](err)
	if !ok {
		return err
	}
	if partial == nil {
		return err
	}
	if nilvalue.IsNil(rs) {
		rs = NewResultSet[TMeta](nil, nil)
	}
	return &partialResultError[TMeta]{
		cause: err,
		view: &PartialFailureError[TMeta]{
			Errors: append([]error(nil), partial.Errors...),
			Result: rs,
		},
	}
}

// partialResultError provides the current diagnostic view while retaining the
// original error graph for [errors.Is]/[errors.As] and without mutating host-owned errors.
type partialResultError[TMeta any] struct {
	cause error
	view  *PartialFailureError[TMeta]
}

func (e *partialResultError[TMeta]) Error() string { return e.cause.Error() }
func (e *partialResultError[TMeta]) Unwrap() error { return e.cause }
func (e *partialResultError[TMeta]) As(target any) bool {
	if out, ok := target.(**PartialFailureError[TMeta]); ok {
		*out = e.view
		return true
	}
	return false
}

// AsPartialFailure reports whether err is a PartialFailureError and returns it when true.
func AsPartialFailure[TMeta any](err error) (*PartialFailureError[TMeta], bool) {
	if partial, ok := errors.AsType[*PartialFailureError[TMeta]](err); ok {
		return partial, true
	}
	return nil, false
}

// PreserveResultOnError keeps a non-empty ResultSet when err signals partial success.
func PreserveResultOnError[TMeta any](
	rs ResultSet[TMeta],
	err error,
	resolver IdentityResolver[TMeta],
) (ResultSet[TMeta], error) {
	return preserveResultOnError(rs, err, resolver)
}

// preserveResultOnError preserves only the separately returned set; nested error
// results are diagnostic and protection always suppresses payload.
func preserveResultOnError[TMeta any](
	rs ResultSet[TMeta],
	err error,
	resolver IdentityResolver[TMeta],
) (ResultSet[TMeta], error) {
	if err == nil {
		if nilvalue.IsNil(rs) {
			return NewResultSet[TMeta](nil, resolver), nil
		}
		return RewrapResultSet(rs, resolver), nil
	}
	resolver = DefaultResolver(resolver)
	if access.IsProtectionFailure(err) {
		empty := NewResultSet[TMeta](nil, resolver)
		return empty, &access.ProtectionError{Cause: suppressErrorPayload(err, empty)}
	}
	err = syncPartialFailureResult(err, rs)
	if !nilvalue.IsNil(rs) && !rs.IsEmpty() {
		return RewrapResultSet(rs, resolver), err
	}
	return NewResultSet[TMeta](nil, resolver), err
}

// suppressErrorPayload overrides public diagnostic views without changing causes.
func suppressErrorPayload[TMeta any](err error, empty ResultSet[TMeta]) error {
	err = syncPartialFailureResult(err, empty)
	if fusion, ok := errors.AsType[*FusionFailureError[TMeta]](err); ok && fusion != nil {
		return &fusionDiagnosticError[TMeta]{
			cause: err,
			view:  &FusionFailureError[TMeta]{Cause: fusion.Cause, observations: nil},
		}
	}
	return err
}

type fusionDiagnosticError[TMeta any] struct {
	cause error
	view  *FusionFailureError[TMeta]
}

func (e *fusionDiagnosticError[TMeta]) Error() string { return e.cause.Error() }
func (e *fusionDiagnosticError[TMeta]) Unwrap() error { return e.cause }
func (e *fusionDiagnosticError[TMeta]) As(target any) bool {
	if out, ok := target.(**FusionFailureError[TMeta]); ok {
		*out = e.view
		return true
	}
	return false
}
