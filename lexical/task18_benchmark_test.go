package lexical

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"

	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

// task18Corpus is deterministic: 16 repeated common terms, four bucket terms,
// and one unique term per document. No random generator or external corpus.
func task18Corpus(n int) []retrieval.Document[struct{}] {
	docs := make([]retrieval.Document[struct{}], n)
	for i := range docs {
		docs[i] = retrieval.Document[struct{}]{
			ID: fmt.Sprintf("doc-%06d", i),
			Content: fmt.Sprintf(
				"common common common common common common common common common common common common common common common common bucket%d bucket%d bucket%d bucket%d unique%d",
				i%100,
				i%100,
				i%100,
				i%100,
				i,
			),
		}
	}
	return docs
}
func task18Index(b *testing.B) *BM25Index[struct{}] {
	b.Helper()
	schema, err := filter.NewSchema().Build()
	if err != nil {
		b.Fatal(err)
	}
	idx, err := NewBM25Index(schema, Config[struct{}]{SearchFields: []string{"content"}}, DefaultTokenizer{}, nil)
	if err != nil {
		b.Fatal(err)
	}
	return idx
}

//nolint:gocognit // Keep fixed workload and measurement boundaries visible in one reproducible harness.
func BenchmarkTask18Lexical(b *testing.B) {
	for _, n := range []int{100, 1000, 10000} {
		docs := task18Corpus(n)
		b.Run(fmt.Sprintf("Bulk/N%d", n), func(b *testing.B) {
			idx := task18Index(b)
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				if err := idx.Index(docs); err != nil {
					b.Fatal(err)
				}
			}
		})
		for _, text := range []string{"unique0", "common"} {
			b.Run(fmt.Sprintf("Read/%s/N%d", text, n), func(b *testing.B) {
				idx := task18Index(b)
				if err := idx.Index(docs); err != nil {
					b.Fatal(err)
				}
				req := retrieval.Query[struct{}]{
					Read:    retrieval.UnrestrictedRead(),
					Text:    text,
					Options: retrieval.RetrieveOptions{TopK: 10},
				}
				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					out, err := idx.Retrieve(context.Background(), req)
					if err != nil || out.Len() == 0 {
						b.Fatalf("read: %v", err)
					}
				}
			})
		}
		b.Run(fmt.Sprintf("Update/N%d", n), func(b *testing.B) {
			idx := task18Index(b)
			if err := idx.Index(docs); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := range b.N {
				if err := idx.Upsert(docs[i%n]); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(fmt.Sprintf("ConcurrentReadUpdate/N%d", n), func(b *testing.B) {
			idx := task18Index(b)
			if err := idx.Index(docs); err != nil {
				b.Fatal(err)
			}
			req := retrieval.Query[struct{}]{
				Read:    retrieval.UnrestrictedRead(),
				Text:    "unique0",
				Options: retrieval.RetrieveOptions{TopK: 10},
			}
			var operations atomic.Uint64
			b.ReportAllocs()
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					i := operations.Add(1)
					if i%10 == 0 {
						if err := idx.Upsert(docs[int(i)%n]); err != nil {
							b.Error(err)
						}
					} else {
						out, err := idx.Retrieve(context.Background(), req)
						if err != nil || out.Len() != 1 {
							b.Errorf("read: %v", err)
						}
					}
				}
			})
		})
	}
}
