package retrieval

import (
	"context"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
)

// RequestPartialReadNode explicitly permits skipping one configured branch only
// when its required read capability is unsupported during admission. Name is a
// static configuration label, never a document/source/policy identity. Runtime
// errors, authority denial, revocation and cancellation are not skippable.
type RequestPartialReadNode[TIntent, TRequestMeta, TMeta, TExecMeta any] struct {
	Child    RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta]
	Name     string
	Resolver IdentityResolver[TMeta]
}

// PartialReadNode is the no-request-metadata explicit partial branch profile.
type PartialReadNode[TIntent, TMeta, TExecMeta any] = RequestPartialReadNode[TIntent, NoRequestMeta, TMeta, TExecMeta]

// AdmitRead retains scope and reports only a pre-dispatch capability skip.
func (n RequestPartialReadNode[TIntent, TRequestMeta, TMeta, TExecMeta]) AdmitRead(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (ReadCoverage, error) {
	coverage, _, err := n.inspectChild(ctx, req)
	return coverage, err
}

func (n RequestPartialReadNode[TIntent, TRequestMeta, TMeta, TExecMeta]) inspectChild(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (ReadCoverage, bool, error) {
	if err := validateCoverageBranch(n.Name); err != nil {
		return UnobservedReadCoverage(), false, err
	}
	if n.Child == nil {
		return UnobservedReadCoverage(), false, fmt.Errorf("%w: partial read child", ragy.ErrInvalidArgument)
	}
	coverage, err := InspectRead(ctx, req, n.Child)
	if access.IsUnsupportedCapability(err) {
		partial, partialErr := PartialReadCoverage(n.Name)
		return partial, true, partialErr
	}
	return coverage, false, err
}

// Execute never invokes an unsupported child and never converts an I/O failure to
// a capability skip. Even an empty partial result retains its admission coverage.
func (n RequestPartialReadNode[TIntent, TRequestMeta, TMeta, TExecMeta]) Execute(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
	exec TExecMeta,
) (RetrievalResult[TMeta, TExecMeta], error) {
	resolver := executionResolver(n.Resolver)
	if err := req.Read.Check(ctx); err != nil {
		var zero TExecMeta
		return emptyRetrievalResult(resolver, zero), err
	}
	coverage, skip, err := n.inspectChild(ctx, req)
	if err != nil {
		var zero TExecMeta
		return emptyRetrievalResult(resolver, zero), access.Protect(err)
	}
	result := emptyRetrievalResult(resolver, exec)
	if skip {
		result.Coverage = coverage
		result.BranchTrace = []BranchStep{
			{Node: n.Name, Kind: BranchKindNode, Route: "", State: BranchStateUnsupported, Error: ""},
		}
		return finishReadResult(ctx, req.Read, result, nil, resolver)
	}
	result, err = n.Child.Execute(ctx, req, exec)
	result.Coverage = MergeReadCoverage(coverage, result.Coverage)
	return finishReadResult(ctx, req.Read, result, err, resolver)
}

func (n RequestPartialReadNode[TIntent, TRequestMeta, TMeta, TExecMeta]) validateExecutionNode() error {
	if err := validateCoverageBranch(n.Name); err != nil {
		return err
	}
	if n.Child == nil {
		return fmt.Errorf("%w: partial read child", ragy.ErrInvalidArgument)
	}
	return validateExecutionNodeTree[TIntent, TRequestMeta, TMeta, TExecMeta](n.Child)
}

func (n RequestPartialReadNode[TIntent, TRequestMeta, TMeta, TExecMeta]) withExecutionResolver(
	resolver IdentityResolver[TMeta],
) (RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta], error) {
	n.Resolver = resolver
	child, err := injectExecutionNodeResolver[TIntent, TRequestMeta, TMeta, TExecMeta](n.Child, resolver)
	if err != nil {
		return nil, err
	}
	n.Child = child
	return n, nil
}
