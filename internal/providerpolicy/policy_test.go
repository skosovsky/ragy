package providerpolicy

import (
	"context"
	"errors"
	"strings"
	"testing"

	ragy "github.com/skosovsky/ragy"
)

func TestPrivateSDKCauseAndCancellationIdentity(t *testing.T) {
	// Arrange.
	cause := errors.New("private source and credential")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	// Act.
	failure := Failure(ctx, cause, 403)
	// Assert: protected context wins without erasing the SDK diagnostic identity.
	if !errors.Is(failure, context.Canceled) || !errors.Is(failure, cause) ||
		strings.Contains(failure.Error(), "private") {
		t.Fatal("unsafe/lost SDK cause", failure)
	}
}

func TestSDKStatusClassAndPrivateCause(t *testing.T) {
	// Arrange.
	cause := errors.New("private provider body")
	// Act.
	failure := Failure(context.Background(), cause, 429)
	// Assert.
	if !errors.Is(failure, ragy.ErrUnavailable) || !errors.Is(failure, cause) ||
		strings.Contains(failure.Error(), "private") {
		t.Fatal("status or privacy", failure)
	}
}
