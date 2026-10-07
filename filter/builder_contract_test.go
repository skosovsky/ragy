package filter

import (
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
)

func TestBuilderInvalidStateSentinels(t *testing.T) {
	// Arrange.
	var schema Schema
	var builder *Builder
	// Act.
	constructed, constructorErr := NewBuilder(schema)
	condition, buildErr := builder.Build()
	_, zeroErr := (&Builder{}).Build()
	// Assert: invalid states share the public argument classification.
	if constructed != nil || !errors.Is(constructorErr, ragy.ErrInvalidArgument) ||
		!errors.Is(buildErr, ragy.ErrInvalidArgument) ||
		!errors.Is(zeroErr, ragy.ErrInvalidArgument) ||
		!IsEmpty(condition.IR()) {
		t.Fatal("builder sentinels", constructorErr, buildErr)
	}
}
