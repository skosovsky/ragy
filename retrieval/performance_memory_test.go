package retrieval

import (
	"fmt"
	"strconv"
	"testing"

	"github.com/skosovsky/ragy/source"
)

func auditDocs(n int, supports bool) []Document[struct{}] {
	docs := make([]Document[struct{}], n)
	for i := range docs {
		docs[i] = Document[struct{}]{
			ID:             "same",
			Content:        "same",
			Score:          1,
			ScoreState:     ScorePresent,
			ScoreSemantics: "test",
		}
		if supports {
			docs[i].SourceSupports = []source.Locator{
				{
					Reference: source.Reference{
						Namespace:         "n",
						Source:            strconv.Itoa(i),
						Revision:          "r",
						Transformation:    "t",
						AccessFingerprint: "a",
						Artifact:          "a",
						Representation:    "text",
					},
					Kind: source.DocumentLocation,
				},
			}
		}
	}
	return docs
}

func TestAuditDedupEvidenceIsolation(t *testing.T) {
	// Arrange: equal payloads with distinct source evidence.
	docs := auditDocs(128, true)
	rs := NewResultSet(docs, nil)
	// Act.
	out, err := rs.Dedup()
	// Assert: all evidence and existing score-history semantics survive.
	if err != nil {
		t.Fatal(err)
	}
	got := out.Documents()
	if len(got) != 1 || len(got[0].SourceSupports) != 128 || len(got[0].ScoreHistory) != 127 {
		t.Fatalf("unexpected result: %d/%d/%d", len(got), len(got[0].SourceSupports), len(got[0].ScoreHistory))
	}
	got[0].SourceSupports[0].Reference.Source = "mutated"
	if out.Documents()[0].SourceSupports[0].Reference.Source == "mutated" ||
		rs.Documents()[0].SourceSupports[0].Reference.Source == "mutated" {
		t.Fatal("isolation broken")
	}
}

//nolint:gochecknoglobals // Retain the owned benchmark output across measured calls, matching the audit fixture.
var auditSink ResultSet[struct{}]

func BenchmarkAuditDedup(b *testing.B) {
	for _, supports := range []bool{false, true} {
		for _, n := range []int{64, 128, 256, 512} {
			b.Run(fmt.Sprintf("supports=%t/n=%d", supports, n), func(b *testing.B) {
				rs := NewResultSet(auditDocs(n, supports), nil)
				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					var err error
					auditSink, err = rs.Dedup()
					if err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
