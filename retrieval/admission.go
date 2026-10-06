package retrieval

import (
	"context"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
)

// RequestReadAdmission negotiates all reachable leaves before dispatch and reports
// explicit partial skips. No planners/predicates/payload I/O or request mutation
// may run during admission. Required host freshness validation remains mandatory.
type RequestReadAdmission[TIntent, TRequestMeta any] interface {
	AdmitRead(context.Context, Request[TIntent, TRequestMeta]) (ReadCoverage, error)
}

// InspectRead negotiates immutable coverage for a complete composition. Unknown
// scoped/pinned nodes fail closed; unrestricted live reads are selected explicitly.
func InspectRead[TIntent, TRequestMeta any](
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
	node any,
) (ReadCoverage, error) {
	if err := req.Read.Check(ctx); err != nil {
		return UnobservedReadCoverage(), err
	}
	if isNilReadTarget(node) {
		return UnobservedReadCoverage(), access.Protect(ragy.ErrInvalidArgument)
	}
	if !req.Read.IsScoped() && req.Read.Publication().IsCurrent() {
		return ReadCoverage{state: CoverageUnrestricted, skipped: nil}, nil
	}
	admission, ok := node.(RequestReadAdmission[TIntent, TRequestMeta])
	if !ok {
		return UnobservedReadCoverage(), access.UnsupportedCapability(ragy.ErrUnsupported)
	}
	coverage, err := admission.AdmitRead(ctx, req)
	if err != nil {
		if access.IsUnsupportedCapability(err) {
			return UnobservedReadCoverage(), err
		}
		return UnobservedReadCoverage(), access.NonSkippable(err)
	}
	if err := coverage.validate(); err != nil {
		return UnobservedReadCoverage(), access.Protect(err)
	}
	if err := req.Read.Check(ctx); err != nil {
		return UnobservedReadCoverage(), err
	}
	return BindPublicationCoverage(req.Read, coverage), nil
}

// PreflightRead negotiates a composition before dispatch; use InspectRead when
// transporting admission coverage to a custom result envelope.
func PreflightRead[TIntent, TRequestMeta any](ctx context.Context, req Request[TIntent, TRequestMeta], node any) error {
	_, err := InspectRead(ctx, req, node)
	return err
}

func admitChildren[TIntent, TRequestMeta any](
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
	children ...any,
) (ReadCoverage, error) {
	coverage := CompleteReadCoverage()
	for _, child := range children {
		if child == nil {
			continue
		}
		current, err := InspectRead(ctx, req, child)
		if err != nil {
			return UnobservedReadCoverage(), err
		}
		coverage = MergeReadCoverage(coverage, current)
	}
	return coverage, nil
}

// AdmitRead negotiates the target before payload I/O.
func (n RequestBackendNode[TIntent, TRequestMeta, TMeta, TExecMeta]) AdmitRead(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (ReadCoverage, error) {
	return inspectBackendRead(ctx, req, n.Backend)
}

// AdmitRead negotiates the target before payload I/O.
func (n RequestExecutionRetrieverNode[TIntent, TRequestMeta, TMeta, TExecMeta]) AdmitRead(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (ReadCoverage, error) {
	return inspectBackendRead(ctx, req, n.Backend)
}

// AdmitRead negotiates the target before payload I/O.
func (n resultRetrieverNode[TIntent, TRequestMeta, TMeta]) AdmitRead(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (ReadCoverage, error) {
	return inspectBackendRead(ctx, req, n.Backend)
}

// AdmitRead negotiates every reachable branch without executing callbacks.
func (n RequestFallbackNode[TIntent, TRequestMeta, TMeta, TExecMeta]) AdmitRead(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (ReadCoverage, error) {
	return admitChildren(ctx, req, n.Primary, n.Secondary)
}

// AdmitRead negotiates every reachable branch without executing callbacks.
func (n RequestRescueNode[TIntent, TRequestMeta, TMeta, TExecMeta]) AdmitRead(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (ReadCoverage, error) {
	return admitChildren(ctx, req, n.Primary, n.Secondary)
}

// AdmitRead negotiates every reachable branch without executing callbacks.
func (n RequestConditionalNode[TIntent, TRequestMeta, TMeta, TExecMeta]) AdmitRead(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (ReadCoverage, error) {
	return admitChildren(ctx, req, n.Child)
}

// AdmitRead negotiates every reachable branch without executing callbacks.
func (n requestNodeExecutionAdapter[TIntent, TRequestMeta, TMeta, TExecMeta]) AdmitRead(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (ReadCoverage, error) {
	return admitChildren(ctx, req, n.Node)
}

// AdmitRead negotiates every reachable branch without executing callbacks.
func (n resultFallbackNode[TIntent, TRequestMeta, TMeta]) AdmitRead(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (ReadCoverage, error) {
	return admitChildren(ctx, req, n.Primary, n.Secondary)
}

// AdmitRead negotiates every reachable branch without executing callbacks.
func (n resultRescueNode[TIntent, TRequestMeta, TMeta]) AdmitRead(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (ReadCoverage, error) {
	return admitChildren(ctx, req, n.Primary, n.Secondary)
}

// AdmitRead negotiates every reachable branch without executing callbacks.
func (n resultConditionalNode[TIntent, TRequestMeta, TMeta]) AdmitRead(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (ReadCoverage, error) {
	return admitChildren(ctx, req, n.Child)
}

// AdmitRead negotiates all aggregate leaves before parallel dispatch.
func (n RequestExecutionAggregateNode[TIntent, TRequestMeta, TMeta, TExecMeta]) AdmitRead(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (ReadCoverage, error) {
	coverage := CompleteReadCoverage()
	for _, child := range n.Nodes {
		current, err := InspectRead(ctx, req, child)
		if err != nil {
			return UnobservedReadCoverage(), err
		}
		coverage = MergeReadCoverage(coverage, current)
	}
	return coverage, nil
}

// AdmitRead negotiates all aggregate leaves before parallel dispatch.
func (n resultAggregateNode[TIntent, TRequestMeta, TMeta]) AdmitRead(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (ReadCoverage, error) {
	coverage := CompleteReadCoverage()
	for _, child := range n.Nodes {
		current, err := InspectRead(ctx, req, child)
		if err != nil {
			return UnobservedReadCoverage(), err
		}
		coverage = MergeReadCoverage(coverage, current)
	}
	return coverage, nil
}

// AdmitRead negotiates every route case/default before planning or dispatch.
func (n RequestRouteSwitchNode[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) AdmitRead(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (ReadCoverage, error) {
	coverage := CompleteReadCoverage()
	for _, branch := range n.Cases {
		current, err := InspectRead(ctx, req, branch.Node)
		if err != nil {
			return UnobservedReadCoverage(), err
		}
		coverage = MergeReadCoverage(coverage, current)
	}
	other, err := admitChildren(ctx, req, n.Default)
	return MergeReadCoverage(coverage, other), err
}

func validateReadRequest[TIntent, TRequestMeta any](
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
	node any,
) error {
	if err := req.Options.Validate(); err != nil {
		return err
	}
	return PreflightRead(ctx, req, node)
}
