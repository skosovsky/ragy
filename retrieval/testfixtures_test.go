package retrieval

import (
	"context"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

type partialBackend[TIntent, TMeta any] struct {
	docs []Document[TMeta]
	err  error
}

func (b partialBackend[TIntent, TMeta]) Schema() filter.Schema { return filter.EmptySchema() }

func (b partialBackend[TIntent, TMeta]) Retrieve(
	_ context.Context,
	_ Query[TIntent],
) (ResultSet[TMeta], error) {
	rs := NewResultSet(b.docs, DocumentIDResolver[TMeta]{})
	return rs, b.err
}

type partialFailureBackend[TIntent, TMeta any] struct {
	docs []Document[TMeta]
}

func (b partialFailureBackend[TIntent, TMeta]) Schema() filter.Schema { return filter.EmptySchema() }

func (b partialFailureBackend[TIntent, TMeta]) Retrieve(
	_ context.Context,
	_ Query[TIntent],
) (ResultSet[TMeta], error) {
	rs := NewResultSet(b.docs, DocumentIDResolver[TMeta]{})
	return rs, &PartialFailureError[TMeta]{Errors: []error{ragy.ErrUnavailable}, Result: rs}
}
