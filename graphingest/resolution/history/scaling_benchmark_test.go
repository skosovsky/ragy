package history

import (
	"fmt"
	"strconv"
	"testing"

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
