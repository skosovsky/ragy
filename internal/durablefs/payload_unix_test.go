//go:build darwin || linux

package durablefs_test

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/internal/durablefs"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/source"
)

type payloadReaderFunc func(context.Context, lifecycle.PayloadRead) ([]byte, error)

func (f payloadReaderFunc) ReadPayload(ctx context.Context, input lifecycle.PayloadRead) ([]byte, error) {
	return f(ctx, input)
}
func payloadInput() lifecycle.PayloadRead {
	return lifecycle.PayloadRead{
		Reference: source.Reference{
			Namespace:         "n",
			Source:            "s",
			Revision:          "r",
			Transformation:    "t",
			AccessFingerprint: "a",
			Artifact:          "id",
			Representation:    "vector",
		},
		Path:     "payload.json",
		MaxBytes: 3,
	}
}
func TestQueryPayloadRejectsInvalidInputBeforeHostIO(t *testing.T) {
	for _, name := range []string{"reference", "path", "zero", "negative", "overflow", "typed-nil", "canceled"} {
		t.Run(name, func(t *testing.T) {
			// Arrange.
			input := payloadInput()
			calls := 0
			reader := payloadReaderFunc(
				func(context.Context, lifecycle.PayloadRead) ([]byte, error) { calls++; return []byte("abc"), nil },
			)
			ctx := t.Context()
			expected := ragy.ErrInvalidArgument
			switch name {
			case "reference":
				input.Reference.Artifact = ""
			case "path":
				input.Path = ""
			case "zero":
				input.MaxBytes = 0
			case "negative":
				input.MaxBytes = -1
			case "overflow":
				input.MaxBytes = math.MaxInt64
			case "typed-nil":
				reader = nil
			case "canceled":
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
				expected = context.Canceled
			}
			// Act.
			data, err := durablefs.QueryPayload(ctx, reader, input)
			// Assert.
			if !errors.Is(err, expected) || data != nil || calls != 0 {
				t.Fatalf("rejected input performed I/O: %q %v calls=%d", data, err, calls)
			}
		})
	}
}
func TestQueryPayloadHostBoundary(t *testing.T) {
	for _, name := range []string{"bounded", "oversized", "error", "canceled-during"} {
		t.Run(name, func(t *testing.T) {
			// Arrange.
			input := payloadInput()
			calls := 0
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			failure := errors.New("host failure")
			reader := payloadReaderFunc(func(gotCtx context.Context, got lifecycle.PayloadRead) ([]byte, error) {
				calls++
				if got != input || gotCtx != ctx {
					t.Fatal("read identity or context changed")
				}
				return hostPayloadResult(name, cancel, failure)
			})
			expected := hostPayloadError(name, failure)
			// Act.
			data, err := durablefs.QueryPayload(ctx, reader, input)
			// Assert: no hidden retry and no bytes on failure.
			if !errors.Is(err, expected) || calls != 1 {
				t.Fatalf("boundary result %q %v calls=%d", data, err, calls)
			}
			if expected != nil && data != nil {
				t.Fatal("failure exposed bytes")
			}
			if expected == nil && string(data) != "abc" {
				t.Fatal("bounded bytes changed")
			}
		})
	}
}
func TestQueryPayloadDefaultReadsActualBoundedFile(t *testing.T) {
	// Arrange.
	input := payloadInput()
	input.Path = filepath.Join(t.TempDir(), "payload.json")
	if err := os.WriteFile(input.Path, []byte("abc"), 0o600); err != nil {
		t.Fatal(err)
	}
	// Act.
	data, err := durablefs.QueryPayload(t.Context(), nil, input)
	// Assert.
	if err != nil || string(data) != "abc" {
		t.Fatal(data, err)
	}
	input.MaxBytes = 2
	if data, err = durablefs.QueryPayload(t.Context(), nil, input); !errors.Is(err, ragy.ErrProtocol) || data != nil {
		t.Fatal("oversized file read", data, err)
	}
}
func TestQueryPayloadErrorsPreserveContractsAndRedactStorage(t *testing.T) {
	for _, expected := range []error{context.Canceled, context.DeadlineExceeded, ragy.ErrProtocol, ragy.ErrUnavailable} {
		// Arrange: host details must not leak to adapter errors.
		wrapped := errors.Join(errors.New("private storage path"), expected)
		// Act.
		got := durablefs.QueryPayloadError(wrapped)
		// Assert.
		if !errors.Is(got, expected) || got.Error() != expected.Error() {
			t.Fatal("contract error changed or host details exposed", got)
		}
	}
}

func hostPayloadResult(name string, cancel context.CancelFunc, failure error) ([]byte, error) {
	switch name {
	case "oversized":
		return []byte("abcd"), nil
	case "error":
		return nil, failure
	case "canceled-during":
		cancel()
	}
	return []byte("abc"), nil
}

func hostPayloadError(name string, failure error) error {
	switch name {
	case "oversized":
		return ragy.ErrProtocol
	case "error":
		return failure
	case "canceled-during":
		return context.Canceled
	default:
		return nil
	}
}
