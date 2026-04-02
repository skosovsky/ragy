package ragy

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"testing"
)

func TestErrorFromHTTPResponse(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		code   int
		wantIs error
	}{
		{"400", http.StatusBadRequest, ErrInvalidArgument},
		{"401", http.StatusUnauthorized, ErrInvalidArgument},
		{"404", http.StatusNotFound, ErrInvalidArgument},
		{"429", http.StatusTooManyRequests, ErrUnavailable},
		{"500", http.StatusInternalServerError, ErrUnavailable},
		{"502", http.StatusBadGateway, ErrUnavailable},
		{"200", http.StatusOK, ErrProtocol},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := ErrorFromHTTPResponse(tt.code, "test", "body")
			if !errors.Is(err, tt.wantIs) {
				t.Fatalf("ErrorFromHTTPResponse(%d): %v, want errors.Is(..., %v)", tt.code, err, tt.wantIs)
			}
		})
	}
}

func TestWrapTransportError_CanceledUnchanged(t *testing.T) {
	t.Parallel()
	in := context.Canceled
	out := WrapTransportError(in)
	if !errors.Is(out, context.Canceled) {
		t.Fatalf("WrapTransportError(canceled) = %v", out)
	}
}

func TestWrapTransportError_DeadlineExceeded(t *testing.T) {
	t.Parallel()
	in := context.DeadlineExceeded
	out := WrapTransportError(in)
	if !errors.Is(out, ErrUnavailable) || !errors.Is(out, context.DeadlineExceeded) {
		t.Fatalf("WrapTransportError(deadline) = %v", out)
	}
}

type timeoutError struct{}

func (timeoutError) Error() string   { return "timeout" }
func (timeoutError) Timeout() bool   { return true }
func (timeoutError) Temporary() bool { return true }

func TestWrapTransportError_NetTimeout(t *testing.T) {
	t.Parallel()
	var err net.Error = timeoutError{}
	out := WrapTransportError(err)
	if !errors.Is(out, ErrUnavailable) {
		t.Fatalf("WrapTransportError(timeout) = %v", out)
	}
}

func TestWrapTransportError_Generic(t *testing.T) {
	t.Parallel()
	in := errors.New("connection refused")
	out := WrapTransportError(in)
	if !errors.Is(out, ErrUnavailable) || !errors.Is(out, in) {
		t.Fatalf("WrapTransportError(refused) = %v", out)
	}
}

func TestWrapBackendError(t *testing.T) {
	t.Parallel()
	if WrapBackendError(nil, "op") != nil {
		t.Fatal("WrapBackendError(nil) != nil")
	}
	if err := WrapBackendError(context.Canceled, "op"); !errors.Is(err, context.Canceled) {
		t.Fatalf("WrapBackendError(canceled) = %v", err)
	}
	if err := WrapBackendError(ErrInvalidArgument, "op"); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("WrapBackendError(ErrInvalidArgument) = %v", err)
	}
	raw := errors.New("db down")
	wrapped := WrapBackendError(raw, "search")
	if !errors.Is(wrapped, ErrUnavailable) || !errors.Is(wrapped, raw) {
		t.Fatalf("WrapBackendError(raw) = %v", wrapped)
	}
}

func TestWrapBackendError_AlreadyUnavailable(t *testing.T) {
	t.Parallel()
	inner := fmt.Errorf("%w: inner", ErrUnavailable)
	out := WrapBackendError(inner, "op")
	if !errors.Is(out, ErrUnavailable) {
		t.Fatalf("got %v", out)
	}
}
