package ragy

import "errors"

var (
	ErrInvalidArgument = errors.New("invalid argument")
	ErrUnsupported     = errors.New("unsupported")
	ErrProtocol        = errors.New("protocol error")
	// ErrUnavailable signals a transient or infrastructure failure (network, timeout, HTTP 429/5xx,
	// rate limits). Callers may retry with backoff; use [errors.Is] with err and ErrUnavailable.
	ErrUnavailable     = errors.New("unavailable")
	ErrMissingID       = errors.New("missing id")
	ErrMissingSourceID = errors.New("missing source id")
	ErrEmptyText       = errors.New("empty text")
	ErrEmptyVector     = errors.New("empty vector")
	ErrInvalidPage     = errors.New("invalid page")
	ErrInvalidGraph    = errors.New("invalid graph")
)
