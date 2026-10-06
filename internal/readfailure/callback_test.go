package readfailure_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/internal/readfailure"
)

type payloadError struct{ secret string }

func (e *payloadError) Error() string { return e.secret }
func (*payloadError) Unwrap() error   { return ragy.ErrProtocol }

func TestCallbackCausesRemainClassifiableWithoutPayloadUnwrap(t *testing.T) {
	// Arrange: callback error carries private payload and a public classification.
	private := &payloadError{secret: "private document content"}
	gate := access.Protect(context.Canceled)
	// Act: callback boundary then final delivery under the canceled read.
	joined := readfailure.Join(gate, private)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	err := readfailure.Check(ctx, access.Unrestricted(), joined)
	// Assert: both causes inspectable with errors.Is, typed payload/text suppressed.
	var leaked *payloadError
	var protection *access.ProtectionError
	if !errors.Is(err, context.Canceled) || !errors.Is(err, ragy.ErrProtocol) || !errors.Is(err, private) ||
		!errors.As(err, &protection) || errors.As(err, &leaked) || strings.Contains(err.Error(), private.secret) {
		t.Fatal(err, leaked)
	}
}

func TestValidGatePreservesOrdinaryCallbackPolicy(t *testing.T) {
	// Arrange.
	ordinary := errors.New("ordinary callback")
	// Act.
	joined := readfailure.Join(nil, ordinary)
	delivered := readfailure.Check(t.Context(), access.Unrestricted(), joined)
	// Assert: no new protection/ordinary partial policy classification.
	if !errors.Is(delivered, ordinary) || access.IsProtectionFailure(delivered) || readfailure.Join(nil, nil) != nil {
		t.Fatal(delivered)
	}
}
