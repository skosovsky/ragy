package retrieval

import (
	"errors"
	"fmt"
	"strings"
)

// PartialFailureError reports that an aggregate node returned documents while one or
// more child branches failed. The ResultSet is still valid and may be used by callers
// that tolerate partial success.
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
	return fmt.Sprintf("aggregate partial failure (%d child error(s)): %s", len(e.Errors), strings.Join(msgs, "; "))
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
	return &PartialFailureError[TMeta]{Errors: partial.Errors, Result: rs}
}

// AsPartialFailure reports whether err is a PartialFailureError and returns it when true.
func AsPartialFailure[TMeta any](err error) (*PartialFailureError[TMeta], bool) {
	var partial *PartialFailureError[TMeta]
	if errors.As(err, &partial) {
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

// preserveResultOnError keeps a non-empty ResultSet when err signals partial success.
func preserveResultOnError[TMeta any](
	rs ResultSet[TMeta],
	err error,
	resolver IdentityResolver[TMeta],
) (ResultSet[TMeta], error) {
	if err == nil {
		if rs == nil {
			return NewResultSet[TMeta](nil, resolver), nil
		}
		return RewrapResultSet(rs, resolver), nil
	}
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	if partial, ok := AsPartialFailure[TMeta](err); ok && partial != nil && !partial.Result.IsEmpty() {
		return NewResultSet(partial.Result.Documents(), resolver), err
	}
	if rs != nil && !rs.IsEmpty() {
		return RewrapResultSet(rs, resolver), err
	}
	return NewResultSet[TMeta](nil, resolver), err
}
