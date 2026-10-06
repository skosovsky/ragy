//go:build darwin || linux

package persistent_test

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/tensor/persistent"
)

type queryPayloadReader func(context.Context, lifecycle.PayloadRead) ([]byte, error)

func (f queryPayloadReader) ReadPayload(ctx context.Context, input lifecycle.PayloadRead) ([]byte, error) {
	return f(ctx, input)
}

func TestPersistentQueryHostPayloadPortChecksAndNoRetry(t *testing.T) {
	for _, name := range []string{"valid", "digest", "oversized", "canceled", "deadline", "storage-error"} {
		t.Run(name, func(t *testing.T) {
			// Arrange: stage/publish verify physical bytes independently of the query reader.
			config := newConfig(t)
			input := records()
			calls := 0
			config.PayloadReader = queryPayloadReader(
				func(ctx context.Context, got lifecycle.PayloadRead) ([]byte, error) {
					calls++
					if got.Reference.Validate() != nil || got.MaxBytes != config.MaxPayloadBytes ||
						!strings.HasPrefix(got.Path, config.Root) ||
						ctx != t.Context() {
						t.Fatal("payload authority or context changed")
					}
					return queryPayloadBytes(name, got)
				},
			)
			adapter := published(t, config, input)
			if calls != 0 {
				t.Fatal("query port used for lifecycle verification")
			}
			// Act.
			out, err := adapter.Query(t.Context(), query(pin(t, config), input))
			result := out.Documents
			assertQueryPayload(t, name, result.Len(), calls, err)
		})
	}
}
func TestPersistentConfigRejectsTypedNilPayloadPort(t *testing.T) {
	// Arrange.
	config := newConfig(t)
	var reader queryPayloadReader
	config.PayloadReader = reader
	// Act.
	adapter, err := persistent.New(config)
	// Assert.
	if adapter != nil || !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatal("typed nil accepted", err)
	}
}

func queryPayloadBytes(name string, got lifecycle.PayloadRead) ([]byte, error) {
	switch name {
	case "oversized":
		return make([]byte, got.MaxBytes+1), nil
	case "canceled":
		return nil, context.Canceled
	case "deadline":
		return nil, context.DeadlineExceeded
	case "storage-error":
		return nil, errors.New("private storage detail")
	case "digest":
		return []byte("{}"), nil
	default:
		return os.ReadFile(got.Path)
	}
}

func assertQueryPayload(t *testing.T, name string, hits, calls int, err error) {
	t.Helper()
	// Assert: failure never delivers prior hits or retries a materialization.
	if name == "valid" {
		if err != nil || hits != 3 || calls != 3 {
			t.Fatal("actual payload reader", hits, calls, err)
		}
		return
	}
	expected := ragy.ErrProtocol
	switch name {
	case "canceled":
		expected = context.Canceled
	case "deadline":
		expected = context.DeadlineExceeded
	case "storage-error":
		expected = ragy.ErrUnavailable
	}
	if !errors.Is(err, expected) || hits != 0 || calls != 1 {
		t.Fatalf("unsafe reader failure: hits=%d calls=%d error=%v", hits, calls, err)
	}
	if err != nil && strings.Contains(err.Error(), "private storage detail") {
		t.Fatal("host error leaked")
	}
}
