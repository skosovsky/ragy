package recipe

import (
	"context"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
)

// UnsupportedQueryEncoderBridge documents the strict integration boundary for
// adapters which cannot enforce recipe-reserved remote input/output token caps.
// Admit rejects before pricing, reservation or provider I/O.
type UnsupportedQueryEncoderBridge struct{}

func (UnsupportedQueryEncoderBridge) Admit(context.Context, dense.Request) error {
	return ragy.ErrUnsupported
}
func (UnsupportedQueryEncoderBridge) Encode(context.Context, dense.Request, ModelLimits) (dense.Result, Usage, error) {
	return dense.Result{}, Usage{}, ragy.ErrUnsupported
}
