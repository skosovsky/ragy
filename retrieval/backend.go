package retrieval

import (
	"context"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
)

// Backend executes retrieval against a concrete store without post-processing.
// Prefer ExecutionPipeline for orchestration; this path is for direct backend access.
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
	// AdmissionProject is a pure metadata/options projection for request-aware
	// admission across different request types. It is required for every cross-type
	// scoped/pinned projection, regardless of target or decorator capabilities. It must not perform payload/model
	// I/O, mutate inputs or grant access. Same-type admission needs no callback.
	AdmissionProject RequestProjector[TIntent, TRequestMeta, TBackendIntent, TBackendMeta]
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
	if _, err := b.AdmitRead(ctx, req); err != nil {
		return NewResultSet[TMeta](nil, nil), err
	}
	projected := b.Project(req)
	projected.Read = req.Read
	if projected.Plan == nil {
		projected.Plan = ProjectPlannedQuery(req.Plan, projected.Intent)
	}
	rs, err := b.Next.Retrieve(ctx, projected)
	return DeliverRead(ctx, req.Read, rs, err, nil)
}

// AdmitRead preserves target admission without invoking the payload projector.
func (b ProjectedBackend[TIntent, TRequestMeta, TBackendIntent, TBackendMeta, TMeta]) AdmitRead(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (ReadCoverage, error) {
	if err := req.Read.Check(ctx); err != nil {
		return UnobservedReadCoverage(), err
	}
	if isNilReadTarget(b.Next) {
		return UnobservedReadCoverage(), access.Protect(ragy.ErrInvalidArgument)
	}
	if !req.Read.IsScoped() && req.Read.Publication().IsCurrent() {
		return ReadCoverage{state: CoverageUnrestricted, skipped: nil}, nil
	}
	if same, ok := any(b.Next).(RequestBackend[TIntent, TRequestMeta, TMeta]); ok && b.AdmissionProject == nil {
		return inspectBackendRead(ctx, req, same)
	}
	if b.AdmissionProject == nil {
		return UnobservedReadCoverage(), access.UnsupportedCapability(ragy.ErrUnsupported)
	}
	projected := b.AdmissionProject(CopyRequestOptions(req))
	projected.Read = req.Read
	if projected.Plan == nil {
		projected.Plan = ProjectPlannedQuery(req.Plan, projected.Intent)
	}
	return inspectBackendRead(ctx, projected, b.Next)
}

// PostProcessor transforms a ranked result set.
type PostProcessor[TMeta any] interface {
	Process(ctx context.Context, read access.Binding, rs ResultSet[TMeta]) (ResultSet[TMeta], error)
}

// Schema forwards target schema admission through request projection.
func (b ProjectedBackend[TIntent, TRequestMeta, TBackendIntent, TBackendMeta, TMeta]) Schema() filter.Schema {
	if provider, ok := b.Next.(ReadCapabilityProvider); ok {
		return provider.Schema()
	}
	return filter.Schema{}
}

// ReadCapabilities forwards only guarantees actually declared by the target.
func (b ProjectedBackend[TIntent, TRequestMeta, TBackendIntent, TBackendMeta, TMeta]) ReadCapabilities() access.Capabilities {
	if provider, ok := b.Next.(ReadCapabilityProvider); ok {
		return provider.ReadCapabilities()
	}
	return access.Capabilities{RequirePinnedPublication: false, ScopeProfile: false, PinnedPublication: false}
}

// AdmitPublication forwards partial pin admission through projection.
func (b ProjectedBackend[TIntent, TRequestMeta, TBackendIntent, TBackendMeta, TMeta]) AdmitPublication(
	publication access.Publication,
) error {
	if admission, ok := b.Next.(PublicationAdmission); ok {
		return admission.AdmitPublication(publication)
	}
	return access.UnsupportedCapability(ragy.ErrUnsupported)
}
