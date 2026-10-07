package observation

import (
	"context"
	"testing"
)

type cooperativeClassificationError struct {
	isCalls     *int
	unwrapCalls *int
}

func (cooperativeClassificationError) Error() string   { panic("raw message must not be formatted") }
func (e cooperativeClassificationError) Is(error) bool { *e.isCalls++; return false }
func (e cooperativeClassificationError) Unwrap() error { *e.unwrapCalls++; return context.Canceled }

func TestClassificationInvokesCooperativeHostMethodsWithoutErrorText(t *testing.T) {
	// Arrange.
	isCalls, unwrapCalls := 0, 0
	err := cooperativeClassificationError{isCalls: &isCalls, unwrapCalls: &unwrapCalls}
	// Act.
	class := Classify(err)
	// Assert: host methods run; payload-free classification is not a sandbox boundary.
	if class != ErrorCanceled || isCalls == 0 || unwrapCalls == 0 {
		t.Fatalf("class=%v Is=%d Unwrap=%d", class, isCalls, unwrapCalls)
	}
}
