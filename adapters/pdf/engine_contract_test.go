package pdf

import (
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/source"
)

func TestSanitizedEngineErrorClasses(t *testing.T) {
	for _, tc := range []struct {
		code string
		want error
	}{
		{"engine_internal_error", ragy.ErrUnavailable},
		{"invalid_pdf", ragy.ErrInvalidArgument},
		{"limit_exceeded", ragy.ErrInvalidArgument},
		{"unsupported_geometry", ragy.ErrUnsupported},
		{"private exception", ragy.ErrProtocol},
	} {
		t.Run(tc.code, func(t *testing.T) {
			// Arrange.
			wire := engineOutput{Error: tc.code}
			// Act.
			doc, err := normalize(wire, source.Reference{})
			// Assert.
			if !errors.Is(err, tc.want) || len(doc.Pages) != 0 || doc.Schema != "" {
				t.Fatalf("class or payload: %#v %v", doc, err)
			}
			if err.Error() == tc.code {
				t.Fatal("raw engine error exposed")
			}
		})
	}
}
