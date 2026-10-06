//go:build darwin || linux

package durablefs

import (
	"context"
	"errors"
	"math"
	"reflect"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/lifecycle"
)

// QueryPayload executes an optional host read port only for an admitted reference.
// The default retains the actual bounded local filesystem reader.
func QueryPayload(ctx context.Context, reader lifecycle.PayloadReader, input lifecycle.PayloadRead) ([]byte, error) {
	if input.Reference.Validate() != nil || input.Path == "" || input.MaxBytes <= 0 ||
		input.MaxBytes == math.MaxInt64 || ValidatePayloadReader(reader) != nil {
		return nil, ragy.ErrInvalidArgument
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if reader == nil {
		return ReadBounded(ctx, input.Path, input.MaxBytes)
	}
	data, err := reader.ReadPayload(ctx, input)
	if err != nil {
		return nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	if int64(len(data)) > input.MaxBytes {
		return nil, ragy.ErrProtocol
	}
	return data, nil
}

// QueryPayloadError preserves contract errors without exposing host storage details.
func QueryPayloadError(err error) error {
	switch {
	case errors.Is(err, context.Canceled):
		return context.Canceled
	case errors.Is(err, context.DeadlineExceeded):
		return context.DeadlineExceeded
	case errors.Is(err, ragy.ErrProtocol):
		return ragy.ErrProtocol
	default:
		return ragy.ErrUnavailable
	}
}

// ValidatePayloadReader allows the absent default port but rejects a typed nil.
func ValidatePayloadReader(reader lifecycle.PayloadReader) error {
	if reader == nil {
		return nil
	}
	value := reflect.ValueOf(reader)
	kind := value.Kind()
	if kind == reflect.Chan || kind == reflect.Func || kind == reflect.Interface || kind == reflect.Map ||
		kind == reflect.Pointer ||
		kind == reflect.Slice {
		if value.IsNil() {
			return ragy.ErrInvalidArgument
		}
	}
	return nil
}
