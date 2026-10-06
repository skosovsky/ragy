package resolution_test

import (
	"context"
	"fmt"
	"strconv"
	"testing"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/graphingest/resolution"
)

// Counts are exact declared host maxima, not universal ragy capacities.
func BenchmarkResolutionDeclaredUpperBound(b *testing.B) {
	for _, size := range []int{64, 256} {
		for _, distinct := range []bool{false, true} {
			b.Run(fmt.Sprintf("mentions=%d/distinct=%t", size, distinct), func(b *testing.B) {
				benchmarkResolution(b, size, distinct)
			})
		}
	}
}

func benchmarkResolution(b *testing.B, size int, distinct bool) {
	cfg := config()
	cfg.MaxEntities, cfg.MaxRelations, cfg.MaxSupports = size, 1, size
	input := resolution.Extraction[kind, relation, attributes]{}
	for i := range size {
		owner := "same"
		if distinct {
			owner = strconv.Itoa(i)
		}
		input.Entities = append(input.Entities,
			entity(strconv.Itoa(i), "prod", "Billing", strconv.Itoa(i), owner, "Service"),
		)
	}
	resolver, err := resolution.New(cfg)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		result, runErr := resolver.Resolve(context.Background(), access.Unrestricted(), input)
		if runErr != nil || len(result.Entities) != 1 {
			b.Fatal(runErr)
		}
	}
}
