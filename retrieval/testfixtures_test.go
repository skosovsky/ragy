package retrieval

import (
	"context"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

type partialBackend[TMeta any] struct {
	docs []Document[TMeta]
	err  error
}

func (b partialBackend[TMeta]) Schema() filter.Schema { return filter.EmptySchema() }

func (b partialBackend[TMeta]) Retrieve(
	_ context.Context,
	_ string,
	_ RetrieveOptions,
) (ResultSet[TMeta], error) {
	rs := NewResultSet(b.docs, DocumentIDResolver[TMeta]{})
	return rs, b.err
}

type partialFailureBackend[TMeta any] struct {
	docs []Document[TMeta]
}

func (b partialFailureBackend[TMeta]) Schema() filter.Schema { return filter.EmptySchema() }

func (b partialFailureBackend[TMeta]) Retrieve(
	_ context.Context,
	_ string,
	_ RetrieveOptions,
) (ResultSet[TMeta], error) {
	rs := NewResultSet(b.docs, DocumentIDResolver[TMeta]{})
	return rs, &PartialFailureError[TMeta]{Errors: []error{ragy.ErrUnavailable}, Result: rs}
}
