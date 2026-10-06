// Package readfailure retains callback classifications across protected delivery
// without exposing callback error text or payload-bearing error objects.
package readfailure

import (
	"context"
	"errors"

	"github.com/skosovsky/ragy/access"
)

// Join preserves gate protection and callback identity/classification through
// [errors.Is]. Only gate errors are unwrapped: [errors.As] cannot recover a private
// callback payload. Custom error Is/Unwrap remains cooperative host code.
func Join(gate, callback error) error {
	if gate == nil {
		return callback
	}
	if callback == nil {
		return gate
	}
	return &access.ProtectionError{Cause: &callbackError{gate: gate, callback: callback}}
}

// Check applies a final read gate without erasing an already observed callback
// cause. It sanitizes independently protected failures under the access contract.
func Check(ctx context.Context, read access.Binding, callback error) error {
	if gate := read.Check(ctx); gate != nil {
		return Join(gate, callback)
	}
	if access.IsProtectionFailure(callback) {
		return access.Protect(callback)
	}
	return callback
}

type callbackError struct {
	gate     error
	callback error
}

func (*callbackError) Error() string          { return "protected callback failure" }
func (e *callbackError) Unwrap() error        { return e.gate }
func (e *callbackError) Is(target error) bool { return errors.Is(e.callback, target) }
