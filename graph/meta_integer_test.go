package graph

import (
	"math"
	"testing"

	"github.com/skosovsky/ragy/filter"
)

func TestNormalizeMetaPreservesIntegerIdentity(t *testing.T) {
	t.Parallel()
	// Arrange.
	builder := filter.NewSchema()
	if _, err := builder.Int("tenant"); err != nil {
		t.Fatal(err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatal(err)
	}
	type meta struct {
		Tenant int64 `json:"tenant"`
	}
	for _, tenant := range []int64{9007199254740993, math.MinInt64, math.MaxInt64} {
		// Act.
		got, normalizeErr := NormalizeMeta(schema, meta{Tenant: tenant})
		// Assert.
		if normalizeErr != nil || got.Tenant != tenant {
			t.Fatalf("tenant=%d got=%v error=%v", tenant, got, normalizeErr)
		}
	}
}
