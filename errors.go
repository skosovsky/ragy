package ragy

import "errors"

var (
	ErrInvalidArgument = errors.New("invalid argument")
	ErrUnsupported     = errors.New("unsupported")
	ErrProtocol        = errors.New("protocol error")
	ErrMissingID       = errors.New("missing id")
	ErrMissingSourceID = errors.New("missing source id")
	ErrEmptyText       = errors.New("empty text")
	ErrEmptyVector     = errors.New("empty vector")
	ErrInvalidPage     = errors.New("invalid page")
	ErrInvalidGraph    = errors.New("invalid graph")
)
