package retrieval

import (
	"context"
	"errors"
	"fmt"
	"reflect"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/internal/parallel"
)

// NoExecutionMeta is the default execution metadata type for pipelines that
// only need ResultSet, diagnostics, and branch trace.
type NoExecutionMeta struct{}

// ExecutionDiagnostic captures planner, binder, route, and node decisions in a
// pipeline result without binding ragy to an application observability model.
type ExecutionDiagnostic struct {
	Stage string
	Key   string
	Value string
}

// BranchStep describes one graph branch decision or execution outcome.
type BranchStep struct {
	Node  string
	Kind  string
	Route string
	State string
	Error string
}

const (
	BranchKindRoute    = "route"
	BranchKindCase     = "case"
	BranchKindFallback = "fallback"
	BranchKindRescue   = "rescue"
	BranchKindNode     = "node"

	BranchStateSelected = "selected"
	BranchStateSkipped  = "skipped"
	BranchStateEmpty    = "empty"
	BranchStateReturned = "returned"
	BranchStateErrored  = "errored"
)

func pipelineErrorTrace(node string, err error) []BranchStep {
	return []BranchStep{nodeBranchStep(node, BranchStateErrored, err)}
}

// RetrievalResult is the typed result envelope for execution-aware pipelines.
// ResultSet carries retrieved documents; Executed is caller-owned metadata for
// route decisions, side outputs, and execution summaries.
//
//nolint:revive // RetrievalResult is the public contract name used by the task and docs.
type RetrievalResult[TMeta, TExecMeta any] struct {
	ResultSet   ResultSet[TMeta]
	Executed    TExecMeta
	Diagnostics []ExecutionDiagnostic
	BranchTrace []BranchStep
}

// Documents returns the result documents for callers that do not need direct
// ResultSet access.
func (r RetrievalResult[TMeta, TExecMeta]) Documents() []Document[TMeta] {
	if r.ResultSet == nil {
		return nil
	}
	return r.ResultSet.Documents()
}

// Len returns the number of result documents.
func (r RetrievalResult[TMeta, TExecMeta]) Len() int {
	if r.ResultSet == nil {
		return 0
	}
	return r.ResultSet.Len()
}

// IsEmpty reports whether the result has no documents.
func (r RetrievalResult[TMeta, TExecMeta]) IsEmpty() bool {
	return r.ResultSet == nil || r.ResultSet.IsEmpty()
}

// BoundRequest is the output of a plan binding stage.
type BoundRequest[TIntent, TRequestMeta, TExecMeta any] struct {
	Request     Request[TIntent, TRequestMeta]
	Executed    TExecMeta
	Diagnostics []ExecutionDiagnostic
}

// RequestPlanBinder binds planner output back into typed request/options before
// retrieval execution.
type RequestPlanBinder[TIntent, TRequestMeta, TExecMeta any] interface {
	BindPlan(
		ctx context.Context,
		req Request[TIntent, TRequestMeta],
		plan *PlannedQuery[TIntent],
		exec TExecMeta,
	) (BoundRequest[TIntent, TRequestMeta, TExecMeta], error)
}

// RequestPlanBinderFunc adapts a function into RequestPlanBinder.
type RequestPlanBinderFunc[TIntent, TRequestMeta, TExecMeta any] func(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
	plan *PlannedQuery[TIntent],
	exec TExecMeta,
) (BoundRequest[TIntent, TRequestMeta, TExecMeta], error)

// BindPlan implements RequestPlanBinder.
func (f RequestPlanBinderFunc[TIntent, TRequestMeta, TExecMeta]) BindPlan(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
	plan *PlannedQuery[TIntent],
	exec TExecMeta,
) (BoundRequest[TIntent, TRequestMeta, TExecMeta], error) {
	return f(ctx, req, plan, exec)
}

// RequestExecutionNode executes retrieval and returns the full result envelope.
type RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta any] interface {
	Execute(
		ctx context.Context,
		req Request[TIntent, TRequestMeta],
		exec TExecMeta,
	) (RetrievalResult[TMeta, TExecMeta], error)
}

// ExecutionNode is the no-request-metadata execution node shape.
type ExecutionNode[TIntent, TMeta, TExecMeta any] = RequestExecutionNode[
	TIntent,
	NoRequestMeta,
	TMeta,
	TExecMeta,
]

// RequestExecutionBackend retrieves with access to typed execution metadata and
// can return side outputs through RetrievalResult.
type RequestExecutionBackend[TIntent, TRequestMeta, TMeta, TExecMeta any] interface {
	Retrieve(
		ctx context.Context,
		req Request[TIntent, TRequestMeta],
		exec TExecMeta,
	) (RetrievalResult[TMeta, TExecMeta], error)
}

// ExecutionBackend is the no-request-metadata execution backend shape.
type ExecutionBackend[TIntent, TMeta, TExecMeta any] = RequestExecutionBackend[
	TIntent,
	NoRequestMeta,
	TMeta,
	TExecMeta,
]

// RequestBackendNode wraps a ResultSet backend as an execution-aware node.
type RequestBackendNode[TIntent, TRequestMeta, TMeta, TExecMeta any] struct {
	Backend  RequestBackend[TIntent, TRequestMeta, TMeta]
	Resolver IdentityResolver[TMeta]
	Name     string
}

// BackendNode is the no-request-metadata backend node.
type BackendNode[TIntent, TMeta, TExecMeta any] = RequestBackendNode[
	TIntent,
	NoRequestMeta,
	TMeta,
	TExecMeta,
]

// Execute implements RequestExecutionNode.
func (n RequestBackendNode[TIntent, TRequestMeta, TMeta, TExecMeta]) Execute(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
	exec TExecMeta,
) (RetrievalResult[TMeta, TExecMeta], error) {
	resolver := n.Resolver
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	if n.Backend == nil {
		return emptyRetrievalResult(resolver, exec),
			fmt.Errorf("%w: backend node backend", ragy.ErrInvalidArgument)
	}
	if err := req.Options.Validate(); err != nil {
		return emptyRetrievalResult(resolver, exec), err
	}
	rs, err := n.Backend.Retrieve(ctx, req)
	result := RetrievalResult[TMeta, TExecMeta]{
		ResultSet:   ensureResultSet(rs, resolver),
		Executed:    exec,
		Diagnostics: nil,
		BranchTrace: []BranchStep{nodeBranchStep(n.Name, resultState(rs, err), err)},
	}
	if err != nil {
		result.ResultSet, _ = preserveResultOnError(result.ResultSet, err, resolver)
	}
	return result, err
}

func (n RequestBackendNode[TIntent, TRequestMeta, TMeta, TExecMeta]) validateExecutionNode() error {
	if n.Backend == nil {
		return fmt.Errorf("%w: backend node backend", ragy.ErrInvalidArgument)
	}
	return nil
}

//nolint:unused,unparam // injectExecutionNodeResolver discovers backend nodes through this internal hook.
func (n RequestBackendNode[TIntent, TRequestMeta, TMeta, TExecMeta]) withExecutionResolver(
	resolver IdentityResolver[TMeta],
) (RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta], error) {
	n.Resolver = resolver
	return n, nil
}

// RequestFallbackNode runs secondary when primary succeeds with an empty ResultSet.
type RequestFallbackNode[TIntent, TRequestMeta, TMeta, TExecMeta any] struct {
	Primary   RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta]
	Secondary RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta]
	Resolver  IdentityResolver[TMeta]
	Name      string
}

// FallbackNode is the no-request-metadata fallback node.
type FallbackNode[TIntent, TMeta, TExecMeta any] = RequestFallbackNode[
	TIntent,
	NoRequestMeta,
	TMeta,
	TExecMeta,
]

// Execute implements RequestExecutionNode.
func (n RequestFallbackNode[TIntent, TRequestMeta, TMeta, TExecMeta]) Execute(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
	exec TExecMeta,
) (RetrievalResult[TMeta, TExecMeta], error) {
	resolver := executionResolver(n.Resolver)
	if n.Primary == nil {
		return emptyRetrievalResult(resolver, exec),
			fmt.Errorf("%w: fallback primary node", ragy.ErrInvalidArgument)
	}
	primary, err := n.Primary.Execute(ctx, req, exec)
	primary.ResultSet = ensureResultSet(primary.ResultSet, resolver)
	if err != nil {
		if partialSuccessRS(primary.ResultSet, err) {
			primary.ResultSet, _ = preserveResultOnError(primary.ResultSet, err, resolver)
		}
		primary.BranchTrace = append(primary.BranchTrace, fallbackBranchStep(n.Name, BranchStateSkipped, err))
		return primary, err
	}
	if !primary.ResultSet.IsEmpty() || n.Secondary == nil {
		primary.BranchTrace = append(primary.BranchTrace, fallbackBranchStep(n.Name, BranchStateSkipped, nil))
		return primary, nil
	}
	secondary, secErr := n.Secondary.Execute(ctx, req, primary.Executed)
	return mergeExecutionBranchResult(
		primary,
		secondary,
		fallbackBranchStep(n.Name, BranchStateSelected, nil),
		resolver,
		secErr,
	)
}

func (n RequestFallbackNode[TIntent, TRequestMeta, TMeta, TExecMeta]) validateExecutionNode() error {
	if n.Primary == nil {
		return fmt.Errorf("%w: fallback primary node", ragy.ErrInvalidArgument)
	}
	if err := validateExecutionNodeTree[TIntent, TRequestMeta, TMeta, TExecMeta](n.Primary); err != nil {
		return err
	}
	if n.Secondary != nil {
		return validateExecutionNodeTree[TIntent, TRequestMeta, TMeta, TExecMeta](n.Secondary)
	}
	return nil
}

//nolint:unused // injectExecutionNodeResolver discovers fallback nodes through this internal hook.
func (n RequestFallbackNode[TIntent, TRequestMeta, TMeta, TExecMeta]) withExecutionResolver(
	resolver IdentityResolver[TMeta],
) (RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta], error) {
	var err error
	n.Resolver = resolver
	n.Primary, err = injectExecutionNodeResolver[TIntent, TRequestMeta, TMeta, TExecMeta](n.Primary, resolver)
	if err != nil {
		return nil, err
	}
	if n.Secondary != nil {
		n.Secondary, err = injectExecutionNodeResolver[TIntent, TRequestMeta, TMeta, TExecMeta](n.Secondary, resolver)
		if err != nil {
			return nil, err
		}
	}
	return n, nil
}

// RequestRescueNode runs secondary when primary errors without partial success.
type RequestRescueNode[TIntent, TRequestMeta, TMeta, TExecMeta any] struct {
	Primary   RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta]
	Secondary RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta]
	Resolver  IdentityResolver[TMeta]
	Name      string
}

// RescueNode is the no-request-metadata rescue node.
type RescueNode[TIntent, TMeta, TExecMeta any] = RequestRescueNode[
	TIntent,
	NoRequestMeta,
	TMeta,
	TExecMeta,
]

// Execute implements RequestExecutionNode.
func (n RequestRescueNode[TIntent, TRequestMeta, TMeta, TExecMeta]) Execute(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
	exec TExecMeta,
) (RetrievalResult[TMeta, TExecMeta], error) {
	resolver := executionResolver(n.Resolver)
	if n.Primary == nil {
		return emptyRetrievalResult(resolver, exec),
			fmt.Errorf("%w: rescue primary node", ragy.ErrInvalidArgument)
	}
	primary, err := n.Primary.Execute(ctx, req, exec)
	primary.ResultSet = ensureResultSet(primary.ResultSet, resolver)
	if err == nil {
		primary.BranchTrace = append(primary.BranchTrace, rescueBranchStep(n.Name, BranchStateSkipped, nil))
		return primary, nil
	}
	if partialSuccessRS(primary.ResultSet, err) {
		primary.ResultSet, _ = preserveResultOnError(primary.ResultSet, err, resolver)
		primary.BranchTrace = append(primary.BranchTrace, rescueBranchStep(n.Name, BranchStateSkipped, err))
		return primary, err
	}
	if n.Secondary == nil {
		primary.BranchTrace = append(primary.BranchTrace, rescueBranchStep(n.Name, BranchStateSkipped, err))
		return primary, err
	}
	secondary, secErr := n.Secondary.Execute(ctx, req, primary.Executed)
	merged, mergeErr := mergeExecutionBranchResult(
		primary,
		secondary,
		rescueBranchStep(n.Name, BranchStateSelected, nil),
		resolver,
		secErr,
	)
	if mergeErr != nil {
		return merged, fmt.Errorf("%w; rescue secondary: %w", err, mergeErr)
	}
	if merged.ResultSet.IsEmpty() {
		return merged, fmt.Errorf("%w: rescue secondary empty", err)
	}
	return merged, nil
}

func (n RequestRescueNode[TIntent, TRequestMeta, TMeta, TExecMeta]) validateExecutionNode() error {
	if n.Primary == nil {
		return fmt.Errorf("%w: rescue primary node", ragy.ErrInvalidArgument)
	}
	if err := validateExecutionNodeTree[TIntent, TRequestMeta, TMeta, TExecMeta](n.Primary); err != nil {
		return err
	}
	if n.Secondary != nil {
		return validateExecutionNodeTree[TIntent, TRequestMeta, TMeta, TExecMeta](n.Secondary)
	}
	return nil
}

//nolint:unused // injectExecutionNodeResolver discovers rescue nodes through this internal hook.
func (n RequestRescueNode[TIntent, TRequestMeta, TMeta, TExecMeta]) withExecutionResolver(
	resolver IdentityResolver[TMeta],
) (RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta], error) {
	var err error
	n.Resolver = resolver
	n.Primary, err = injectExecutionNodeResolver[TIntent, TRequestMeta, TMeta, TExecMeta](n.Primary, resolver)
	if err != nil {
		return nil, err
	}
	if n.Secondary != nil {
		n.Secondary, err = injectExecutionNodeResolver[TIntent, TRequestMeta, TMeta, TExecMeta](n.Secondary, resolver)
		if err != nil {
			return nil, err
		}
	}
	return n, nil
}

// RequestConditionalNode skips the child when predicate is false.
type RequestConditionalNode[TIntent, TRequestMeta, TMeta, TExecMeta any] struct {
	Predicate func(Request[TIntent, TRequestMeta]) bool
	Child     RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta]
	Resolver  IdentityResolver[TMeta]
	Name      string
}

// ConditionalNode is the no-request-metadata conditional node.
type ConditionalNode[TIntent, TMeta, TExecMeta any] = RequestConditionalNode[
	TIntent,
	NoRequestMeta,
	TMeta,
	TExecMeta,
]

// Execute implements RequestExecutionNode.
func (n RequestConditionalNode[TIntent, TRequestMeta, TMeta, TExecMeta]) Execute(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
	exec TExecMeta,
) (RetrievalResult[TMeta, TExecMeta], error) {
	resolver := executionResolver(n.Resolver)
	if n.Predicate != nil && !n.Predicate(req) {
		result := emptyRetrievalResult(resolver, exec)
		result.BranchTrace = []BranchStep{nodeBranchStep(n.Name, BranchStateSkipped, nil)}
		return result, nil
	}
	if n.Child == nil {
		err := fmt.Errorf("%w: conditional child node", ragy.ErrInvalidArgument)
		result := emptyRetrievalResult(resolver, exec)
		result.BranchTrace = []BranchStep{nodeBranchStep(n.Name, BranchStateErrored, err)}
		return result, err
	}
	result, err := n.Child.Execute(ctx, req, exec)
	result.BranchTrace = append(
		[]BranchStep{nodeBranchStep(n.Name, BranchStateSelected, err)},
		result.BranchTrace...,
	)
	return result, err
}

func (n RequestConditionalNode[TIntent, TRequestMeta, TMeta, TExecMeta]) validateExecutionNode() error {
	if n.Child == nil {
		return fmt.Errorf("%w: conditional child node", ragy.ErrInvalidArgument)
	}
	return validateExecutionNodeTree[TIntent, TRequestMeta, TMeta, TExecMeta](n.Child)
}

//nolint:unused // injectExecutionNodeResolver discovers conditional nodes through this internal hook.
func (n RequestConditionalNode[TIntent, TRequestMeta, TMeta, TExecMeta]) withExecutionResolver(
	resolver IdentityResolver[TMeta],
) (RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta], error) {
	n.Resolver = resolver
	rebound, err := injectExecutionNodeResolver[TIntent, TRequestMeta, TMeta, TExecMeta](n.Child, resolver)
	if err != nil {
		return nil, err
	}
	n.Child = rebound
	return n, nil
}

// RequestAggregateNode is the execution-aware aggregate node.
type RequestAggregateNode[TIntent, TRequestMeta, TMeta, TExecMeta any] = RequestExecutionAggregateNode[
	TIntent,
	TRequestMeta,
	TMeta,
	TExecMeta,
]

// AggregateNode is the no-request-metadata aggregate node.
type AggregateNode[TIntent, TMeta, TExecMeta any] = RequestAggregateNode[
	TIntent,
	NoRequestMeta,
	TMeta,
	TExecMeta,
]

type requestNodeExecutionAdapter[TIntent, TRequestMeta, TMeta, TExecMeta any] struct {
	Node     resultNode[TIntent, TRequestMeta, TMeta]
	Resolver IdentityResolver[TMeta]
	Name     string
}

func (n requestNodeExecutionAdapter[TIntent, TRequestMeta, TMeta, TExecMeta]) Execute(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
	exec TExecMeta,
) (RetrievalResult[TMeta, TExecMeta], error) {
	resolver := executionResolver(n.Resolver)
	if n.Node == nil {
		return emptyRetrievalResult(resolver, exec),
			fmt.Errorf("%w: result node adapter child", ragy.ErrInvalidArgument)
	}
	rs, err := n.Node.Retrieve(ctx, req)
	result := RetrievalResult[TMeta, TExecMeta]{
		ResultSet:   ensureResultSet(rs, resolver),
		Executed:    exec,
		Diagnostics: nil,
		BranchTrace: []BranchStep{nodeBranchStep(n.Name, resultState(rs, err), err)},
	}
	if err != nil {
		result.ResultSet, _ = preserveResultOnError(result.ResultSet, err, resolver)
	}
	return result, err
}

func (n requestNodeExecutionAdapter[TIntent, TRequestMeta, TMeta, TExecMeta]) validateExecutionNode() error {
	if n.Node == nil {
		return fmt.Errorf("%w: result node adapter child", ragy.ErrInvalidArgument)
	}
	return validateNodeTree[TIntent, TRequestMeta, TMeta](n.Node)
}

//nolint:unused // injectExecutionNodeResolver discovers internal test adapters through this hook.
func (n requestNodeExecutionAdapter[TIntent, TRequestMeta, TMeta, TExecMeta]) withExecutionResolver(
	resolver IdentityResolver[TMeta],
) (RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta], error) {
	n.Resolver = resolver
	child, err := injectNodeResolver[TIntent, TRequestMeta, TMeta](n.Node, resolver)
	if err != nil {
		return nil, err
	}
	n.Node = child
	return n, nil
}

// RequestExecutionRetrieverNode wraps a RequestExecutionBackend as an
// execution-aware orchestrator node.
type RequestExecutionRetrieverNode[TIntent, TRequestMeta, TMeta, TExecMeta any] struct {
	Backend  RequestExecutionBackend[TIntent, TRequestMeta, TMeta, TExecMeta]
	Resolver IdentityResolver[TMeta]
	Name     string
}

// ExecutionReducer combines caller-owned execution metadata from child branches.
type ExecutionReducer[TExecMeta any] func(current TExecMeta, child TExecMeta) TExecMeta

// RequestExecutionAggregateNode runs execution-aware child nodes and merges their
// ResultSets while preserving diagnostics, branch trace, and typed execution metadata.
type RequestExecutionAggregateNode[TIntent, TRequestMeta, TMeta, TExecMeta any] struct {
	Nodes          []RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta]
	Concurrency    int
	Resolver       IdentityResolver[TMeta]
	Merger         ResultMerger[TMeta]
	MergeExecution ExecutionReducer[TExecMeta]
	Name           string
}

type executionAggregateChildResult[TMeta, TExecMeta any] struct {
	result RetrievalResult[TMeta, TExecMeta]
	err    error
}

// Execute implements RequestExecutionNode.
func (n RequestExecutionAggregateNode[TIntent, TRequestMeta, TMeta, TExecMeta]) Execute(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
	exec TExecMeta,
) (RetrievalResult[TMeta, TExecMeta], error) {
	resolver := n.Resolver
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	if len(n.Nodes) == 0 {
		return emptyRetrievalResult(resolver, exec), nil
	}
	nodes := append([]RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta](nil), n.Nodes...)
	concurrency := n.Concurrency
	if concurrency <= 0 {
		concurrency = len(nodes)
	}
	children, err := parallel.MapOrdered(
		ctx,
		concurrency,
		nodes,
		func(ctx context.Context, node RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta]) (
			executionAggregateChildResult[TMeta, TExecMeta],
			error,
		) {
			return n.runExecutionAggregateChild(ctx, node, req, exec), nil
		},
	)
	if err != nil {
		return emptyRetrievalResult(resolver, exec), err
	}
	return n.finalizeExecutionAggregate(ctx, children, exec, resolver)
}

func (n RequestExecutionAggregateNode[TIntent, TRequestMeta, TMeta, TExecMeta]) runExecutionAggregateChild(
	ctx context.Context,
	node RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta],
	req Request[TIntent, TRequestMeta],
	exec TExecMeta,
) executionAggregateChildResult[TMeta, TExecMeta] {
	result, err := node.Execute(ctx, req, exec)
	resolver := n.Resolver
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	result.ResultSet = ensureResultSet(result.ResultSet, resolver)
	if err != nil {
		result.ResultSet, _ = preserveResultOnError(result.ResultSet, err, resolver)
	}
	return executionAggregateChildResult[TMeta, TExecMeta]{result: result, err: err}
}

func (n RequestExecutionAggregateNode[TIntent, TRequestMeta, TMeta, TExecMeta]) finalizeExecutionAggregate(
	ctx context.Context,
	children []executionAggregateChildResult[TMeta, TExecMeta],
	exec TExecMeta,
	resolver IdentityResolver[TMeta],
) (RetrievalResult[TMeta, TExecMeta], error) {
	sets := make([]aggregateChildResult[TMeta], 0, len(children))
	result := RetrievalResult[TMeta, TExecMeta]{
		ResultSet:   NewResultSet[TMeta](nil, resolver),
		Executed:    exec,
		Diagnostics: nil,
		BranchTrace: []BranchStep{nodeBranchStep(n.Name, BranchStateSelected, nil)},
	}
	for i, child := range children {
		sets = append(sets, aggregateChildResult[TMeta]{rs: child.result.ResultSet, err: child.err})
		result.Diagnostics = append(result.Diagnostics, child.result.Diagnostics...)
		result.BranchTrace = append(result.BranchTrace, aggregateChildTrace(i, child)...)
		if n.MergeExecution != nil {
			result.Executed = n.MergeExecution(result.Executed, child.result.Executed)
		}
	}
	merger, err := resolveAggregateMerger(n.Merger, resolver)
	if err != nil {
		result.ResultSet = NewResultSet[TMeta](nil, resolver)
		return result, err
	}
	merged, err := finalizeAggregateRetrieve(ctx, resolver, merger, sets)
	result.ResultSet = ensureResultSet(merged, resolver)
	if err != nil {
		result.ResultSet, _ = preserveResultOnError(result.ResultSet, err, resolver)
	}
	return result, err
}

func aggregateChildTrace[TMeta, TExecMeta any](
	index int,
	child executionAggregateChildResult[TMeta, TExecMeta],
) []BranchStep {
	state := resultState(child.result.ResultSet, child.err)
	step := nodeBranchStep(fmt.Sprintf("aggregate[%d]", index), state, child.err)
	trace := []BranchStep{step}
	trace = append(trace, child.result.BranchTrace...)
	return trace
}

func (n RequestExecutionAggregateNode[TIntent, TRequestMeta, TMeta, TExecMeta]) validateExecutionNode() error {
	for i, child := range n.Nodes {
		if child == nil {
			return fmt.Errorf("%w: execution aggregate child at index %d", ragy.ErrInvalidArgument, i)
		}
		if err := validateExecutionNodeTree[TIntent, TRequestMeta, TMeta, TExecMeta](child); err != nil {
			return err
		}
	}
	return nil
}

//nolint:unused // injectExecutionNodeResolver discovers aggregate nodes through this internal hook.
func (n RequestExecutionAggregateNode[TIntent, TRequestMeta, TMeta, TExecMeta]) withExecutionResolver(
	resolver IdentityResolver[TMeta],
) (RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta], error) {
	n.Resolver = resolver
	for i, child := range n.Nodes {
		rebound, err := injectExecutionNodeResolver[TIntent, TRequestMeta, TMeta, TExecMeta](child, resolver)
		if err != nil {
			return nil, err
		}
		n.Nodes[i] = rebound
	}
	reboundMerger, err := rebindAggregateMerger(n.Merger, resolver)
	if err != nil {
		return nil, err
	}
	n.Merger = reboundMerger
	return n, nil
}

// Execute implements RequestExecutionNode.
func (n RequestExecutionRetrieverNode[TIntent, TRequestMeta, TMeta, TExecMeta]) Execute(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
	exec TExecMeta,
) (RetrievalResult[TMeta, TExecMeta], error) {
	resolver := n.Resolver
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	if n.Backend == nil {
		return emptyRetrievalResult(resolver, exec),
			fmt.Errorf("%w: execution retriever backend", ragy.ErrInvalidArgument)
	}
	if err := req.Options.Validate(); err != nil {
		return emptyRetrievalResult(resolver, exec), err
	}
	result, err := n.Backend.Retrieve(ctx, req, exec)
	result.Executed = preserveExecutionMeta(exec, result.Executed)
	result.ResultSet = ensureResultSet(result.ResultSet, resolver)
	result.BranchTrace = append(
		[]BranchStep{nodeBranchStep(n.Name, resultState(result.ResultSet, err), err)},
		result.BranchTrace...,
	)
	if err != nil {
		result.ResultSet, _ = preserveResultOnError(result.ResultSet, err, resolver)
	}
	return result, err
}

func (n RequestExecutionRetrieverNode[TIntent, TRequestMeta, TMeta, TExecMeta]) validateExecutionNode() error {
	if n.Backend == nil {
		return fmt.Errorf("%w: execution retriever backend", ragy.ErrInvalidArgument)
	}
	return nil
}

//nolint:unused,unparam // injectExecutionNodeResolver discovers retriever nodes through this internal hook.
func (n RequestExecutionRetrieverNode[TIntent, TRequestMeta, TMeta, TExecMeta]) withExecutionResolver(
	resolver IdentityResolver[TMeta],
) (RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta], error) {
	n.Resolver = resolver
	return n, nil
}

func preserveExecutionMeta[TExecMeta any](incoming, returned TExecMeta) TExecMeta {
	var zero TExecMeta
	if reflect.DeepEqual(returned, zero) {
		return incoming
	}
	return returned
}

// RequestExecutionPipelineBuilder builds an execution-aware retrieval pipeline.
type RequestExecutionPipelineBuilder[TIntent, TRequestMeta, TMeta, TExecMeta any] struct {
	root      RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta]
	postChain *PostProcessorChain[TMeta]
	resolver  IdentityResolver[TMeta]
	planner   QueryPlanner[TIntent, TRequestMeta]
	binder    RequestPlanBinder[TIntent, TRequestMeta, TExecMeta]
	seed      func(Request[TIntent, TRequestMeta]) TExecMeta
}

// ExecutionPipelineBuilder is the no-request-metadata execution pipeline builder.
type ExecutionPipelineBuilder[TIntent, TMeta, TExecMeta any] = RequestExecutionPipelineBuilder[
	TIntent,
	NoRequestMeta,
	TMeta,
	TExecMeta,
]

// NewRequestExecutionPipelineBuilder starts execution-aware orchestrator construction.
func NewRequestExecutionPipelineBuilder[TIntent, TRequestMeta, TMeta, TExecMeta any]() *RequestExecutionPipelineBuilder[
	TIntent,
	TRequestMeta,
	TMeta,
	TExecMeta,
] {
	return &RequestExecutionPipelineBuilder[TIntent, TRequestMeta, TMeta, TExecMeta]{}
}

// NewExecutionPipelineBuilder starts execution-aware construction without request metadata.
func NewExecutionPipelineBuilder[TIntent, TMeta, TExecMeta any]() *ExecutionPipelineBuilder[TIntent, TMeta, TExecMeta] {
	return NewRequestExecutionPipelineBuilder[TIntent, NoRequestMeta, TMeta, TExecMeta]()
}

// WithRoot sets the execution-aware root retrieval node.
func (b *RequestExecutionPipelineBuilder[TIntent, TRequestMeta, TMeta, TExecMeta]) WithRoot(
	node RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta],
) *RequestExecutionPipelineBuilder[TIntent, TRequestMeta, TMeta, TExecMeta] {
	b.root = node
	return b
}

// WithAggregate configures parallel execution-aware aggregate routing.
func (b *RequestExecutionPipelineBuilder[TIntent, TRequestMeta, TMeta, TExecMeta]) WithAggregate(
	nodes []RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta],
	concurrency int,
	merger ResultMerger[TMeta],
	mergeExecution ExecutionReducer[TExecMeta],
) *RequestExecutionPipelineBuilder[TIntent, TRequestMeta, TMeta, TExecMeta] {
	b.root = RequestExecutionAggregateNode[TIntent, TRequestMeta, TMeta, TExecMeta]{
		Nodes:          nodes,
		Concurrency:    concurrency,
		Resolver:       nil,
		Merger:         merger,
		MergeExecution: mergeExecution,
		Name:           "",
	}
	return b
}

// WithPlanner runs planner before binder and retrieval graph.
func (b *RequestExecutionPipelineBuilder[TIntent, TRequestMeta, TMeta, TExecMeta]) WithPlanner(
	planner QueryPlanner[TIntent, TRequestMeta],
) *RequestExecutionPipelineBuilder[TIntent, TRequestMeta, TMeta, TExecMeta] {
	b.planner = planner
	return b
}

// WithPlanBinder runs a typed plan binding stage after planning and before retrieval execution.
func (b *RequestExecutionPipelineBuilder[TIntent, TRequestMeta, TMeta, TExecMeta]) WithPlanBinder(
	binder RequestPlanBinder[TIntent, TRequestMeta, TExecMeta],
) *RequestExecutionPipelineBuilder[TIntent, TRequestMeta, TMeta, TExecMeta] {
	b.binder = binder
	return b
}

// WithExecutionSeed derives initial execution metadata from the incoming request.
func (b *RequestExecutionPipelineBuilder[TIntent, TRequestMeta, TMeta, TExecMeta]) WithExecutionSeed(
	seed func(Request[TIntent, TRequestMeta]) TExecMeta,
) *RequestExecutionPipelineBuilder[TIntent, TRequestMeta, TMeta, TExecMeta] {
	b.seed = seed
	return b
}

// WithPostProcessors attaches post-processing after retrieval graph execution.
func (b *RequestExecutionPipelineBuilder[TIntent, TRequestMeta, TMeta, TExecMeta]) WithPostProcessors(
	processors ...PostProcessor[TMeta],
) *RequestExecutionPipelineBuilder[TIntent, TRequestMeta, TMeta, TExecMeta] {
	b.postChain = NewPostProcessorChain[TMeta](processors...)
	return b
}

// WithResolver sets the identity resolver for known execution nodes and post-processors.
func (b *RequestExecutionPipelineBuilder[TIntent, TRequestMeta, TMeta, TExecMeta]) WithResolver(
	resolver IdentityResolver[TMeta],
) *RequestExecutionPipelineBuilder[TIntent, TRequestMeta, TMeta, TExecMeta] {
	b.resolver = resolver
	return b
}

// Build returns the configured execution-aware pipeline.
func (b *RequestExecutionPipelineBuilder[TIntent, TRequestMeta, TMeta, TExecMeta]) Build() (
	*RequestExecutionPipeline[TIntent, TRequestMeta, TMeta, TExecMeta],
	error,
) {
	if b.root == nil {
		return nil, fmt.Errorf("%w: execution pipeline root node", ragy.ErrInvalidArgument)
	}
	if err := validateExecutionNodeTree[TIntent, TRequestMeta, TMeta, TExecMeta](b.root); err != nil {
		return nil, err
	}
	resolver := b.resolver
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	root, err := injectExecutionNodeResolver[TIntent, TRequestMeta, TMeta, TExecMeta](b.root, resolver)
	if err != nil {
		return nil, err
	}
	postChain := b.postChain
	if postChain != nil {
		postChain = postChain.withResolver(resolver)
	}
	return &RequestExecutionPipeline[TIntent, TRequestMeta, TMeta, TExecMeta]{
		root:      root,
		postChain: postChain,
		resolver:  resolver,
		planner:   b.planner,
		binder:    b.binder,
		seed:      b.seed,
	}, nil
}

// RequestExecutionPipeline is a declarative retrieval orchestrator that returns
// the full execution result envelope.
type RequestExecutionPipeline[TIntent, TRequestMeta, TMeta, TExecMeta any] struct {
	root      RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta]
	postChain *PostProcessorChain[TMeta]
	resolver  IdentityResolver[TMeta]
	planner   QueryPlanner[TIntent, TRequestMeta]
	binder    RequestPlanBinder[TIntent, TRequestMeta, TExecMeta]
	seed      func(Request[TIntent, TRequestMeta]) TExecMeta
}

// ExecutionPipeline is the no-request-metadata execution pipeline.
type ExecutionPipeline[TIntent, TMeta, TExecMeta any] = RequestExecutionPipeline[
	TIntent,
	NoRequestMeta,
	TMeta,
	TExecMeta,
]

// Execute runs planner, binder, retrieval graph, and post-processors.
func (p *RequestExecutionPipeline[TIntent, TRequestMeta, TMeta, TExecMeta]) Execute(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (RetrievalResult[TMeta, TExecMeta], error) {
	if p == nil || p.root == nil {
		var zero TExecMeta
		return emptyRetrievalResult(DocumentIDResolver[TMeta]{}, zero),
			fmt.Errorf("%w: execution pipeline root", ragy.ErrInvalidArgument)
	}
	exec := p.initialExecutionMeta(req)

	var diagnostics []ExecutionDiagnostic
	var err error
	req, diagnostics, err = p.planAndBind(ctx, req, exec)
	if err != nil {
		return RetrievalResult[TMeta, TExecMeta]{
			ResultSet:   NewResultSet[TMeta](nil, p.resolver),
			Executed:    exec,
			Diagnostics: diagnostics,
			BranchTrace: pipelineErrorTrace("planner", err),
		}, err
	}
	if p.binder != nil {
		bound, bindErr := p.binder.BindPlan(ctx, req, req.Plan, exec)
		diagnostics = append(diagnostics, bound.Diagnostics...)
		if bindErr != nil {
			return RetrievalResult[TMeta, TExecMeta]{
				ResultSet:   NewResultSet[TMeta](nil, p.resolver),
				Executed:    bound.Executed,
				Diagnostics: diagnostics,
				BranchTrace: pipelineErrorTrace("binder", bindErr),
			}, bindErr
		}
		req = bound.Request
		exec = bound.Executed
	}
	if err := req.Options.Validate(); err != nil {
		return RetrievalResult[TMeta, TExecMeta]{
			ResultSet:   NewResultSet[TMeta](nil, p.resolver),
			Executed:    exec,
			Diagnostics: diagnostics,
			BranchTrace: pipelineErrorTrace("options", err),
		}, err
	}

	result, retrieveErr := p.root.Execute(ctx, req, exec)
	result.ResultSet = ensureResultSet(result.ResultSet, p.resolver)
	result.Diagnostics = append(diagnostics, result.Diagnostics...)
	if retrieveErr != nil {
		result.ResultSet, _ = preserveResultOnError(result.ResultSet, retrieveErr, p.resolver)
	}
	if p.postChain != nil {
		var postErr error
		result.ResultSet, postErr = p.postChain.Process(ctx, req.Options, result.ResultSet)
		if postErr != nil {
			result.ResultSet, _ = preserveResultOnError(result.ResultSet, postErr, p.resolver)
			final := NewResultSet(result.ResultSet.Documents(), p.resolver)
			result.ResultSet = final
			if retrieveErr != nil {
				return result, errors.Join(syncPartialFailureResult(retrieveErr, final), postErr)
			}
			return result, postErr
		}
	} else {
		result.ResultSet = applyTerminalOptions(result.ResultSet, req.Options, p.resolver)
	}
	final := NewResultSet(result.ResultSet.Documents(), p.resolver)
	result.ResultSet = final
	if retrieveErr != nil {
		return result, syncPartialFailureResult(retrieveErr, final)
	}
	return result, nil
}

func (p *RequestExecutionPipeline[TIntent, TRequestMeta, TMeta, TExecMeta]) initialExecutionMeta(
	req Request[TIntent, TRequestMeta],
) TExecMeta {
	if p.seed != nil {
		return p.seed(req)
	}
	var zero TExecMeta
	return zero
}

func (p *RequestExecutionPipeline[TIntent, TRequestMeta, TMeta, TExecMeta]) planAndBind(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
	_ TExecMeta,
) (Request[TIntent, TRequestMeta], []ExecutionDiagnostic, error) {
	if req.Plan != nil {
		req = applyPlannedQuery(req)
		return req, plannerDiagnostics(req.Plan.Diagnostics), nil
	}
	if p.planner == nil {
		return req, nil, nil
	}
	plan, err := p.planner.Plan(ctx, req)
	if err != nil {
		return req, plannerDiagnostics(plan.Diagnostics), err
	}
	req = applyPlannedQuery(req.WithPlan(plan))
	return req, plannerDiagnostics(plan.Diagnostics), nil
}

func validateExecutionNodeTree[TIntent, TRequestMeta, TMeta, TExecMeta any](
	node RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta],
) error {
	if node == nil {
		return fmt.Errorf("%w: execution pipeline node", ragy.ErrInvalidArgument)
	}
	if n, ok := node.(interface{ validateExecutionNode() error }); ok {
		return n.validateExecutionNode()
	}
	return nil
}

func injectExecutionNodeResolver[TIntent, TRequestMeta, TMeta, TExecMeta any](
	node RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta],
	resolver IdentityResolver[TMeta],
) (RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta], error) {
	if node == nil {
		var zero RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta]
		return zero, nil
	}
	if n, ok := node.(interface {
		withExecutionResolver(IdentityResolver[TMeta]) (RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta], error)
	}); ok {
		return n.withExecutionResolver(resolver)
	}
	return node, nil
}

func emptyRetrievalResult[TMeta, TExecMeta any](
	resolver IdentityResolver[TMeta],
	exec TExecMeta,
) RetrievalResult[TMeta, TExecMeta] {
	return RetrievalResult[TMeta, TExecMeta]{
		ResultSet:   NewResultSet[TMeta](nil, resolver),
		Executed:    exec,
		Diagnostics: nil,
		BranchTrace: nil,
	}
}

func executionResolver[TMeta any](resolver IdentityResolver[TMeta]) IdentityResolver[TMeta] {
	if resolver == nil {
		return DocumentIDResolver[TMeta]{}
	}
	return resolver
}

func mergeExecutionBranchResult[TMeta, TExecMeta any](
	before RetrievalResult[TMeta, TExecMeta],
	after RetrievalResult[TMeta, TExecMeta],
	step BranchStep,
	resolver IdentityResolver[TMeta],
	err error,
) (RetrievalResult[TMeta, TExecMeta], error) {
	afterTrace := append([]BranchStep(nil), after.BranchTrace...)
	after.ResultSet = ensureResultSet(after.ResultSet, resolver)
	after.Diagnostics = append(before.Diagnostics, after.Diagnostics...)
	trace := append([]BranchStep(nil), before.BranchTrace...)
	trace = append(trace, step)
	trace = append(trace, afterTrace...)
	after.BranchTrace = trace
	if err != nil {
		after.ResultSet, _ = preserveResultOnError(after.ResultSet, err, resolver)
	}
	return after, err
}

func ensureResultSet[TMeta any](rs ResultSet[TMeta], resolver IdentityResolver[TMeta]) ResultSet[TMeta] {
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	if rs == nil {
		return NewResultSet[TMeta](nil, resolver)
	}
	return RewrapResultSet(rs, resolver)
}

func plannerDiagnostics(in []PlannerDiagnostic) []ExecutionDiagnostic {
	if len(in) == 0 {
		return nil
	}
	out := make([]ExecutionDiagnostic, 0, len(in))
	for _, diag := range in {
		out = append(out, ExecutionDiagnostic{
			Stage: "planner",
			Key:   diag.Key,
			Value: diag.Value,
		})
	}
	return out
}

func routeDiagnostics(in []PlannerDiagnostic) []ExecutionDiagnostic {
	if len(in) == 0 {
		return nil
	}
	out := make([]ExecutionDiagnostic, 0, len(in))
	for _, diag := range in {
		out = append(out, ExecutionDiagnostic{
			Stage: "route",
			Key:   diag.Key,
			Value: diag.Value,
		})
	}
	return out
}

func nodeBranchStep(name, state string, err error) BranchStep {
	step := BranchStep{
		Node:  name,
		Kind:  BranchKindNode,
		Route: "",
		State: state,
		Error: "",
	}
	if err != nil {
		step.Error = err.Error()
	}
	return step
}

func fallbackBranchStep(name, state string, err error) BranchStep {
	return executionBranchStep(name, BranchKindFallback, state, err)
}

func rescueBranchStep(name, state string, err error) BranchStep {
	return executionBranchStep(name, BranchKindRescue, state, err)
}

func executionBranchStep(name, kind, state string, err error) BranchStep {
	step := BranchStep{
		Node:  name,
		Kind:  kind,
		Route: "",
		State: state,
		Error: "",
	}
	if err != nil {
		step.Error = err.Error()
	}
	return step
}

func resultState[TMeta any](rs ResultSet[TMeta], err error) string {
	if err != nil {
		return BranchStateErrored
	}
	if rs == nil || rs.IsEmpty() {
		return BranchStateEmpty
	}
	return BranchStateReturned
}
