package ragy

import (
	"context"
	"iter"
)

// YieldDocuments returns an iterator over docs. On non-nil err, yields a single (zero, err) pair.
// It checks ctx.Done() before each yield and stops early if yield returns false.
func YieldDocuments(ctx context.Context, docs []Document, err error) iter.Seq2[Document, error] {
	return func(yield func(Document, error) bool) {
		if err != nil {
			yield(Document{}, err)
			return
		}
		for _, d := range docs {
			select {
			case <-ctx.Done():
				yield(Document{}, ctx.Err())
				return
			default:
			}
			if !yield(d, nil) {
				return
			}
		}
	}
}
