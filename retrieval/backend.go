package retrieval

import (
	"context"
	"fmt"

	ragy "github.com/skosovsky/ragy"
)

// Backend executes retrieval against a concrete store without post-processing.
// Prefer retrieval.Pipeline for orchestration; this path is for direct backend access.
type Backend[TIntent, TMeta any] = RequestBackend[TIntent, NoRequestMeta, TMeta]

// RequestBackend executes retrieval against a concrete store with the complete
// typed request envelope.
type RequestBackend[TIntent, TRequestMeta, TMeta any] interface {
	Retrieve(ctx context.Context, req Request[TIntent, TRequestMeta]) (ResultSet[TMeta], error)
}

// RequestProjector adapts a richer request shape to a backend-specific request.
type RequestProjector[TIntent, TRequestMeta, TBackendIntent, TBackendMeta any] func(
	Request[TIntent, TRequestMeta],
) Request[TBackendIntent, TBackendMeta]

// ProjectedBackend lets callers use a backend with a different request envelope
// without hiding the projection policy in context.
type ProjectedBackend[TIntent, TRequestMeta, TBackendIntent, TBackendMeta, TMeta any] struct {
	Next    RequestBackend[TBackendIntent, TBackendMeta, TMeta]
	Project RequestProjector[TIntent, TRequestMeta, TBackendIntent, TBackendMeta]
}

// Retrieve implements RequestBackend.
func (b ProjectedBackend[TIntent, TRequestMeta, TBackendIntent, TBackendMeta, TMeta]) Retrieve(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (ResultSet[TMeta], error) {
	if b.Next == nil {
		return NewResultSet[TMeta](nil, DocumentIDResolver[TMeta]{}),
			fmt.Errorf("%w: projected backend next", ragy.ErrInvalidArgument)
	}
	if b.Project == nil {
		return NewResultSet[TMeta](nil, DocumentIDResolver[TMeta]{}),
			fmt.Errorf("%w: projected backend request projector", ragy.ErrInvalidArgument)
	}
	projected := b.Project(req)
	if projected.Plan == nil {
		projected.Plan = ProjectPlannedQuery(req.Plan, projected.Intent)
	}
	return b.Next.Retrieve(ctx, projected)
}

// PostProcessor transforms a ranked result set.
type PostProcessor[TMeta any] interface {
	Process(rs ResultSet[TMeta]) (ResultSet[TMeta], error)
}
