package lifecycle

import (
	"context"

	"github.com/skosovsky/ragy/source"
)

// PayloadRead identifies an already-admitted managed artifact and bounded local
// payload path. It grants no source ownership, write/delete or publication rights.
type PayloadRead struct {
	Reference source.Reference
	Path      string
	MaxBytes  int64
}

// PayloadReader materializes one admitted index payload. Implementations must honor
// context and byte bounds, own returned bytes and avoid implicit retries. Target
// adapters validate the payload digest/reference/shape and gate final delivery.
// A nil optional port in supplied filesystem targets selects their bounded file reader.
type PayloadReader interface {
	ReadPayload(context.Context, PayloadRead) ([]byte, error)
}
