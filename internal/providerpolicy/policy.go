// Package providerpolicy provides domain admission and private diagnostic causes.
// It contains no provider transport or envelope parser.
package providerpolicy

import (
	"context"
	"encoding/json"
	"strings"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/embedding"
)

func ValidateTexts(ctx context.Context, texts []string, limits embedding.Limits) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(texts) == 0 || len(texts) > limits.MaxInputs {
		return ragy.ErrInvalidArgument
	}
	total := 0
	for _, text := range texts {
		if err := ctx.Err(); err != nil {
			return err
		}
		if len(text) > limits.MaxInputBytes-total || !utf8.ValidString(text) || strings.TrimSpace(text) == "" {
			return ragy.ErrInvalidArgument
		}
		total += len(text)
	}
	return ctx.Err()
}

// Size bounds a public SDK representation, not encoded wire bytes (SDK-L01).
func Size(value any, limit int64, class error) error {
	b, err := json.Marshal(value)
	if err != nil || int64(len(b)) > limit {
		return class
	}
	return nil
}

type SDKError struct {
	Class error
	Cause error
}

func (e *SDKError) Error() string   { return e.Class.Error() }
func (e *SDKError) Unwrap() []error { return []error{e.Class, e.Cause} }

func Failure(ctx context.Context, cause error, status int) error {
	class := ragy.ErrUnavailable
	if status > 0 {
		class = ragy.ErrorFromHTTPResponse(status, "provider SDK", "")
	}
	if err := ctx.Err(); err != nil {
		class = err
	}
	return &SDKError{Class: class, Cause: cause}
}
