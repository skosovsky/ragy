package graphsummary

import (
	"context"
	"fmt"
	"strconv"
	"testing"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/source"
)

// Isolated stable-order union kernel; does not include JSON/model/host I/O costs.
func BenchmarkSupportUnionDeclaredUpperBound(b *testing.B) {
	for _, size := range []int{64, 256} {
		b.Run(fmt.Sprintf("supports=%d", size), func(b *testing.B) {
			incoming := make([]source.Locator, size)
			for i := range incoming {
				incoming[i] = source.Locator{Kind: source.DocumentLocation, Reference: source.Reference{
					Namespace: "n", Source: strconv.Itoa(i), Revision: "r1", Transformation: "original",
					AccessFingerprint: "acl", Artifact: "original", Representation: "text",
				}}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				result := union(nil, incoming)
				if len(result) != size {
					b.Fatal("lost support")
				}
			}
		})
	}
}

// Measures repeated freshness callback dispatch with a bounded no-I/O host port.
func BenchmarkSummaryAdmissionDeclaredUpperBound(b *testing.B) {
	for _, size := range []int{64, 256} {
		b.Run(fmt.Sprintf("supports=%d/calls=8", size), func(b *testing.B) {
			benchmarkSummaryAdmission(b, size)
		})
	}
}

func benchmarkSummaryAdmission(b *testing.B, size int) {
	supports := make([]source.Locator, size)
	for i := range supports {
		supports[i] = source.Locator{Kind: source.DocumentLocation, Reference: source.Reference{
			Namespace: "n", Source: strconv.Itoa(i), Revision: "r1", Transformation: "original",
			AccessFingerprint: "acl", Artifact: "original", Representation: "text",
		}}
	}
	mapping, err := source.DerivedText("evidence", supports)
	if err != nil {
		b.Fatal(err)
	}
	communities := []Community[struct{}]{
		{ID: "c", Members: []string{"m"},
			Snippets: []Snippet[struct{}]{{Mapping: mapping, Members: []string{"m"}}}},
	}
	admitted := 0
	r := Recipe[struct{}]{config: Config[struct{}]{
		AdmitSource: func(context.Context, access.Binding, source.Locator) error {
			admitted++
			return nil
		},
	}}
	gate := func() error { return nil }
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		admitted = 0
		for range 8 {
			if err = r.refresh(context.Background(), access.Unrestricted(), communities, gate); err != nil {
				b.Fatal(err)
			}
		}
		if admitted != size*8 {
			b.Fatal("admission count", admitted)
		}
	}
}
