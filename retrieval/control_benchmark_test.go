package retrieval

import (
	"strconv"
	"testing"

	"github.com/skosovsky/ragy/source"
)

//nolint:gochecknoglobals // Retain the owned benchmark result so measured output cannot be optimized away.
var retrievalControlSink ResultSet[struct{}]

// BenchmarkRetrievalControls is self-contained and can be copied unchanged to
// the baseline checkout. Fixtures are built outside timing; every measured call
// still validates, resolves, aggregates, sorts and creates its owned output.
//
//nolint:gocognit // Explicit workload cross-product keeps baseline-compatible fixture construction outside timing.
func BenchmarkRetrievalControls(b *testing.B) {
	for _, fixture := range []struct {
		name     string
		size     int
		distinct bool
		supports bool
	}{
		{"distinct-128", 128, true, true},
		{"distinct-small-8", 8, true, true},
		{"duplicate-small-8", 8, false, true},
		{"duplicate-small-no-supports-8", 8, false, false},
	} {
		for _, operation := range []string{"dedup", "merge"} {
			b.Run(fixture.name+"/"+operation, func(b *testing.B) {
				docs := make([]Document[struct{}], fixture.size)
				for i := range docs {
					id, content := "same", "same"
					if fixture.distinct {
						id = strconv.Itoa(i)
						content = "payload-" + id
					}
					docs[i] = Document[struct{}]{
						ID:             id,
						Content:        content,
						Score:          float64(i),
						ScoreState:     ScorePresent,
						ScoreSemantics: "control",
					}
					if fixture.supports {
						docs[i].SourceSupports = []source.Locator{
							{
								Kind: source.DocumentLocation,
								Reference: source.Reference{
									Namespace:         "n",
									Source:            strconv.Itoa(i),
									Revision:          "r",
									Transformation:    "t",
									AccessFingerprint: "a",
									Artifact:          "a",
									Representation:    "text",
								},
							},
						}
					}
				}
				all := NewResultSet(docs, nil)
				left := NewResultSet(docs[:len(docs)/2], nil)
				right := NewResultSet(docs[len(docs)/2:], nil)
				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					var err error
					if operation == "dedup" {
						retrievalControlSink, err = all.Dedup()
					} else {
						retrievalControlSink, err = left.Merge(right)
					}
					if err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
