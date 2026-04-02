package ragy

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
)

// WrapTransportError classifies errors from [http.Client.Do] and similar transports.
// [context.Canceled] is returned unchanged (caller cancellation, not a retry target).
// Other errors are wrapped with ErrUnavailable for retry policies.
func WrapTransportError(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.Canceled) {
		return err
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return fmt.Errorf("%w: %w", ErrUnavailable, err)
	}
	var ne net.Error
	if errors.As(err, &ne) && ne.Timeout() {
		return fmt.Errorf("%w: %w", ErrUnavailable, err)
	}
	return fmt.Errorf("%w: %w", ErrUnavailable, err)
}

// ErrorFromHTTPResponse maps an HTTP status and response body snippet to a domain error.
// 4xx except 429 → ErrInvalidArgument (do not retry). 429 and 5xx → ErrUnavailable.
// Other status codes → ErrProtocol (unexpected success or redirect without following).
func ErrorFromHTTPResponse(statusCode int, prefix, body string) error {
	switch {
	case statusCode == http.StatusTooManyRequests:
		return fmt.Errorf("%w: %s: status %d: %s", ErrUnavailable, prefix, statusCode, body)
	case statusCode >= http.StatusInternalServerError:
		return fmt.Errorf("%w: %s: status %d: %s", ErrUnavailable, prefix, statusCode, body)
	case statusCode >= http.StatusBadRequest:
		return fmt.Errorf("%w: %s: status %d: %s", ErrInvalidArgument, prefix, statusCode, body)
	default:
		return fmt.Errorf("%w: %s: unexpected status %d: %s", ErrProtocol, prefix, statusCode, body)
	}
}

// WrapBackendError wraps database or RPC errors as ErrUnavailable when not already classified.
// [context.Canceled] is returned unchanged. Errors already matching ErrInvalidArgument,
// ErrUnavailable, or ErrProtocol are returned unchanged.
func WrapBackendError(err error, operation string) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.Canceled) {
		return err
	}
	if errors.Is(err, ErrInvalidArgument) || errors.Is(err, ErrUnavailable) || errors.Is(err, ErrProtocol) {
		return err
	}
	return fmt.Errorf("%w: %s: %w", ErrUnavailable, operation, err)
}
