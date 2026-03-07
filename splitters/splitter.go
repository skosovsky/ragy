package splitters

import (
	"context"
	"iter"

	"github.com/skosovsky/ragy"
)

// Splitter splits a source document into chunk documents lazily.
// Implementations must respect yield-safety: if yield returns false, stop immediately.
// Returned chunks inherit metadata from the source doc (e.g. ParentID = doc.ID).
type Splitter interface {
	Split(ctx context.Context, doc ragy.Document) iter.Seq2[ragy.Document, error]
}
