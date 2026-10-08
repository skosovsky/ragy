package retrieval

import (
	"context"
	"errors"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/internal/nilvalue"
	"github.com/skosovsky/ragy/observation"
)

// partialSuccessRS uses only the separately returned set as payload authority.
func nonRescuablePartial[TMeta any](rs ResultSet[TMeta], err error) bool {
	_, partial := AsPartialFailure[TMeta](err)
	return err != nil && (partial || partialSuccessRS(rs, err))
}

func partialSuccessRS[TMeta any](rs ResultSet[TMeta], err error) bool {
	return err != nil && !nilvalue.IsNil(rs) && !rs.IsEmpty()
}

const defaultAggregateRRFK = 60

// resultNode executes retrieval for a request and always returns a non-nil ResultSet.
type resultNode[TIntent, TRequestMeta, TMeta any] interface {
	Retrieve(ctx context.Context, req Request[TIntent, TRequestMeta]) (ResultSet[TMeta], error)
}

// resultNodeNoMeta is the no-request-metadata request node shape.
type resultNodeNoMeta[TIntent, TMeta any] = resultNode[TIntent, NoRequestMeta, TMeta]

// resultRetrieverNode wraps a RequestBackend as an orchestrator node.
type resultRetrieverNode[TIntent, TRequestMeta, TMeta any] struct {
	Backend  RequestBackend[TIntent, TRequestMeta, TMeta]
	Resolver IdentityResolver[TMeta]
}

// resultRetrieverNodeNoMeta is the no-request-metadata retriever node.
type resultRetrieverNodeNoMeta[TIntent, TMeta any] = resultRetrieverNode[TIntent, NoRequestMeta, TMeta]

// Retrieve implements resultNodeNoMeta.
func (n resultRetrieverNode[TIntent, TRequestMeta, TMeta]) Retrieve(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (ResultSet[TMeta], error) {
	result, err := resultExecutionNode[TIntent, TRequestMeta, TMeta](n).Execute(ctx, req, NoExecutionMeta{})
	return result.ResultSet, err
}

// resultFallbackNode runs secondary when primary succeeds (err == nil) and ResultSet is empty.
// On primary error with empty ResultSet, the error is propagated and secondary is not called.
// On partial success (error with non-empty docs), primary documents are preserved.
type resultFallbackNode[TIntent, TRequestMeta, TMeta any] struct {
	Primary   resultNode[TIntent, TRequestMeta, TMeta]
	Secondary resultNode[TIntent, TRequestMeta, TMeta]
	Resolver  IdentityResolver[TMeta]
}

// resultFallbackNodeNoMeta is the no-request-metadata fallback node.
type resultFallbackNodeNoMeta[TIntent, TMeta any] = resultFallbackNode[TIntent, NoRequestMeta, TMeta]

// Retrieve implements resultNodeNoMeta.
func (n resultFallbackNode[TIntent, TRequestMeta, TMeta]) Retrieve(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (ResultSet[TMeta], error) {
	result, err := resultExecutionNode[TIntent, TRequestMeta, TMeta](n).Execute(ctx, req, NoExecutionMeta{})
	return result.ResultSet, err
}

// resultRescueNode runs secondary when primary returns an error and ResultSet is empty.
// On primary success with empty ResultSet, returns empty without calling secondary.
// On partial success, preserves primary documents.
// Rescue with non-empty secondary returns nil error; empty secondary propagates primary error.
type resultRescueNode[TIntent, TRequestMeta, TMeta any] struct {
	Primary   resultNode[TIntent, TRequestMeta, TMeta]
	Secondary resultNode[TIntent, TRequestMeta, TMeta]
	Resolver  IdentityResolver[TMeta]
}

// resultRescueNodeNoMeta is the no-request-metadata rescue node.
type resultRescueNodeNoMeta[TIntent, TMeta any] = resultRescueNode[TIntent, NoRequestMeta, TMeta]

// Retrieve implements resultNodeNoMeta.
func (n resultRescueNode[TIntent, TRequestMeta, TMeta]) Retrieve(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (ResultSet[TMeta], error) {
	result, err := resultExecutionNode[TIntent, TRequestMeta, TMeta](n).Execute(ctx, req, NoExecutionMeta{})
	return result.ResultSet, err
}

// resultAggregateNode runs child nodes in parallel and merges their ResultSets.
// When Merger is nil, ReciprocalRankFusion is used (recommended for heterogeneous sources).
// For homogeneous score scales, set Merger to NewScoreMerger explicitly.
// Fusion errors preserve observations without an implicit merger. Hosts select
// degradation explicitly with DegradingMerger; score scales remain host-attested.
type resultAggregateNode[TIntent, TRequestMeta, TMeta any] struct {
	Nodes       []resultNode[TIntent, TRequestMeta, TMeta]
	Concurrency int
	Resolver    IdentityResolver[TMeta]
	Merger      ResultMerger[TMeta]
}

// resultAggregateNodeNoMeta is the no-request-metadata aggregate node.
type resultAggregateNodeNoMeta[TIntent, TMeta any] = resultAggregateNode[TIntent, NoRequestMeta, TMeta]

// aggregateChildResult captures one aggregate branch outcome.
type aggregateChildResult[TMeta any] struct {
	rs  ResultSet[TMeta]
	err error
}

// Retrieve implements resultNodeNoMeta.
func (n resultAggregateNode[TIntent, TRequestMeta, TMeta]) Retrieve(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (ResultSet[TMeta], error) {
	result, err := resultExecutionNode[TIntent, TRequestMeta, TMeta](n).Execute(ctx, req, NoExecutionMeta{})
	return result.ResultSet, err
}

func resolveAggregateMerger[TMeta any](
	merger ResultMerger[TMeta],
	resolver IdentityResolver[TMeta],
) (ResultMerger[TMeta], error) {
	if merger != nil {
		if nilvalue.IsNil(merger) {
			return nil, fmt.Errorf("%w: typed-nil aggregate merger", ragy.ErrInvalidArgument)
		}
		switch m := merger.(type) {
		case DegradingMerger[TMeta]:
			if err := m.validate(); err != nil {
				return nil, err
			}
		case *DegradingMerger[TMeta]:
			if err := m.validate(); err != nil {
				return nil, err
			}
		}
		return merger, nil
	}
	return NewReciprocalRankFusion(defaultAggregateRRFK, resolver)
}

func finalizeAggregateRetrieve[TMeta any](
	ctx context.Context,
	resolver IdentityResolver[TMeta],
	merger ResultMerger[TMeta],
	sets []aggregateChildResult[TMeta],
) (ResultSet[TMeta], error) {
	successSets := make([]ResultSet[TMeta], 0, len(sets))
	childErrors := make([]error, 0, len(sets))
	for _, result := range sets {
		if result.err != nil {
			childErrors = append(childErrors, result.err)
		}
		if !nilvalue.IsNil(result.rs) && !result.rs.IsEmpty() {
			successSets = append(successSets, result.rs)
		}
	}
	if err := errors.Join(childErrors...); access.IsProtectionFailure(err) {
		return NewResultSet[TMeta](nil, resolver), &access.ProtectionError{Cause: err}
	}
	if err := ctx.Err(); err != nil {
		return NewResultSet[TMeta](nil, resolver), errors.Join(append(childErrors, err)...)
	}
	fusionCtx, fusionSpan := observation.Begin(ctx, observation.StageFusion)
	merged, mergeErr := merger.Merge(fusionCtx, successSets...)
	merged = ensureResultSet(merged, resolver)
	if fusionSpan != nil {
		fusionSpan.End(observationCompletion(mergeErr, merged))
	}
	if err := ctx.Err(); err != nil {
		return NewResultSet[TMeta](nil, resolver), errors.Join(append(childErrors, mergeErr, err)...)
	}
	if stopsDegradation(mergeErr) {
		return NewResultSet[TMeta](
			nil,
			resolver,
		), &access.ProtectionError{
			Cause: errors.Join(append(childErrors, mergeErr)...),
		}
	}
	if mergeErr != nil {
		observations := make([]ResultSet[TMeta], 0, len(successSets))
		for _, set := range successSets {
			observations = append(observations, NewResultSet(set.Documents(), ResolverFor(set)))
		}
		failure := &FusionFailureError[TMeta]{
			Cause:        errors.Join(append(childErrors, mergeErr)...),
			observations: observations,
		}
		if !merged.IsEmpty() {
			return merged, &PartialFailureError[TMeta]{Errors: []error{failure}, Result: merged}
		}
		return merged, failure
	}
	if len(childErrors) > 0 {
		if len(successSets) == 0 {
			return merged, errors.Join(childErrors...)
		}
		return merged, &PartialFailureError[TMeta]{Errors: childErrors, Result: merged}
	}
	return merged, nil
}

// resultConditionalNode skips the child when predicate is false.
type resultConditionalNode[TIntent, TRequestMeta, TMeta any] struct {
	Predicate func(Request[TIntent, TRequestMeta]) bool
	Child     resultNode[TIntent, TRequestMeta, TMeta]
	Resolver  IdentityResolver[TMeta]
}

// resultConditionalNodeNoMeta is the no-request-metadata conditional node.
type resultConditionalNodeNoMeta[TIntent, TMeta any] = resultConditionalNode[TIntent, NoRequestMeta, TMeta]

// Retrieve implements resultNodeNoMeta.
func (n resultConditionalNode[TIntent, TRequestMeta, TMeta]) Retrieve(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (ResultSet[TMeta], error) {
	result, err := resultExecutionNode[TIntent, TRequestMeta, TMeta](n).Execute(ctx, req, NoExecutionMeta{})
	return result.ResultSet, err
}

type resultPipelineBuilder[TIntent, TRequestMeta, TMeta any] struct {
	root      resultNode[TIntent, TRequestMeta, TMeta]
	postChain *PostProcessorChain[TMeta]
	resolver  IdentityResolver[TMeta]
	planner   QueryPlanner[TIntent, TRequestMeta]
	binder    RequestPlanBinder[TIntent, TRequestMeta, NoExecutionMeta]
}

// newResultPipelineBuilder starts request-metadata-aware orchestrator construction.
func newResultPipelineBuilder[TIntent, TRequestMeta, TMeta any]() *resultPipelineBuilder[TIntent, TRequestMeta, TMeta] {
	return &resultPipelineBuilder[TIntent, TRequestMeta, TMeta]{}
}

// newResultPipelineBuilderNoMeta starts no-request-metadata orchestrator construction.
// Execute still returns the typed RetrievalResult envelope.
func newResultPipelineBuilderNoMeta[TIntent, TMeta any]() *resultPipelineBuilder[TIntent, NoRequestMeta, TMeta] {
	return newResultPipelineBuilder[TIntent, NoRequestMeta, TMeta]()
}

// WithRoot sets the root retrieval node.
func (b *resultPipelineBuilder[TIntent, TRequestMeta, TMeta]) WithRoot(
	node resultNode[TIntent, TRequestMeta, TMeta],
) *resultPipelineBuilder[TIntent, TRequestMeta, TMeta] {
	b.root = node
	return b
}

// WithFallback configures primary/secondary fallback routing.
// Shorthand methods (WithFallback, WithRescue, WithAggregate, WithConditional) replace the
// current root node. Compose complex graphs via WithRoot explicitly.
func (b *resultPipelineBuilder[TIntent, TRequestMeta, TMeta]) WithFallback(
	primary, secondary resultNode[TIntent, TRequestMeta, TMeta],
) *resultPipelineBuilder[TIntent, TRequestMeta, TMeta] {
	b.root = resultFallbackNode[TIntent, TRequestMeta, TMeta]{
		Primary:   primary,
		Secondary: secondary,
		Resolver:  nil,
	}
	return b
}

// WithRescue configures primary/secondary rescue routing on primary errors.
func (b *resultPipelineBuilder[TIntent, TRequestMeta, TMeta]) WithRescue(
	primary, secondary resultNode[TIntent, TRequestMeta, TMeta],
) *resultPipelineBuilder[TIntent, TRequestMeta, TMeta] {
	b.root = resultRescueNode[TIntent, TRequestMeta, TMeta]{ //nolint:exhaustruct_v5 // Resolver injected in Build()
		Primary:   primary,
		Secondary: secondary,
	}
	return b
}

// WithAggregate configures parallel aggregate routing.
// Pass nil merger to use ReciprocalRankFusion (recommended for heterogeneous sources).
func (b *resultPipelineBuilder[TIntent, TRequestMeta, TMeta]) WithAggregate(
	nodes []resultNode[TIntent, TRequestMeta, TMeta],
	concurrency int,
	merger ResultMerger[TMeta],
) *resultPipelineBuilder[TIntent, TRequestMeta, TMeta] {
	b.root = resultAggregateNode[TIntent, TRequestMeta, TMeta]{
		Nodes:       nodes,
		Concurrency: concurrency,
		Merger:      merger,
		Resolver:    nil,
	}
	return b
}

// WithConditional wraps a node behind a predicate.
func (b *resultPipelineBuilder[TIntent, TRequestMeta, TMeta]) WithConditional(
	predicate func(Request[TIntent, TRequestMeta]) bool,
	child resultNode[TIntent, TRequestMeta, TMeta],
) *resultPipelineBuilder[TIntent, TRequestMeta, TMeta] {
	b.root = resultConditionalNode[TIntent, TRequestMeta, TMeta]{
		Predicate: predicate,
		Child:     child,
		Resolver:  nil,
	}
	return b
}

// WithPostProcessors attaches a post-processing chain after retrieval.
// Replaces any previously configured post-processor chain. Shorthand root methods do not clear postChain.
func (b *resultPipelineBuilder[TIntent, TRequestMeta, TMeta]) WithPostProcessors(
	processors ...PostProcessor[TMeta],
) *resultPipelineBuilder[TIntent, TRequestMeta, TMeta] {
	b.postChain = NewPostProcessorChain[TMeta](processors...)
	return b
}

// WithPlanner runs planner before the retrieval graph and attaches its output to Request.Plan.
func (b *resultPipelineBuilder[TIntent, TRequestMeta, TMeta]) WithPlanner(
	planner QueryPlanner[TIntent, TRequestMeta],
) *resultPipelineBuilder[TIntent, TRequestMeta, TMeta] {
	b.planner = planner
	return b
}

// WithPlanBinder runs a typed binding stage after planning and before retrieval execution.
func (b *resultPipelineBuilder[TIntent, TRequestMeta, TMeta]) WithPlanBinder(
	binder RequestPlanBinder[TIntent, TRequestMeta, NoExecutionMeta],
) *resultPipelineBuilder[TIntent, TRequestMeta, TMeta] {
	b.binder = binder
	return b
}

// WithResolver sets the identity resolver for known node types and post-processors.
// Custom resultNodeNoMeta implementations (types not handled by injectNodeResolver) are not
// modified; set Resolver on those nodes explicitly before Build.
func (b *resultPipelineBuilder[TIntent, TRequestMeta, TMeta]) WithResolver(
	resolver IdentityResolver[TMeta],
) *resultPipelineBuilder[TIntent, TRequestMeta, TMeta] {
	b.resolver = resolver
	return b
}

// Build returns the configured orchestrator pipeline.
func (b *resultPipelineBuilder[TIntent, TRequestMeta, TMeta]) Build() (*resultPipeline[TIntent, TRequestMeta, TMeta], error) {
	engine, err := (&RequestExecutionPipelineBuilder[TIntent, TRequestMeta, TMeta, NoExecutionMeta]{
		root: resultExecutionNode[TIntent, TRequestMeta, TMeta](b.root), postChain: b.postChain, resolver: b.resolver, planner: b.planner, binder: b.binder, seed: nil,
	}).Build()
	if err != nil {
		return nil, err
	}
	return &resultPipeline[TIntent, TRequestMeta, TMeta]{engine: engine}, nil
}

func validateNodeTree[TIntent, TRequestMeta, TMeta any](
	node resultNode[TIntent, TRequestMeta, TMeta],
) error {
	if nilvalue.IsNil(node) {
		return fmt.Errorf("%w: pipeline node", ragy.ErrInvalidArgument)
	}
	switch n := node.(type) {
	case resultFallbackNode[TIntent, TRequestMeta, TMeta]:
		return validateBinaryNodeTree(n.Primary, "fallback primary node", n.Secondary)
	case resultRescueNode[TIntent, TRequestMeta, TMeta]:
		return validateBinaryNodeTree(n.Primary, "rescue primary node", n.Secondary)
	case resultAggregateNode[TIntent, TRequestMeta, TMeta]:
		return validateAggregateNodeTree(n.Nodes)
	case resultConditionalNode[TIntent, TRequestMeta, TMeta]:
		if n.Child == nil {
			return fmt.Errorf("%w: conditional child node", ragy.ErrInvalidArgument)
		}
		return validateNodeTree(n.Child)
	case resultRetrieverNode[TIntent, TRequestMeta, TMeta]:
		if n.Backend == nil {
			return fmt.Errorf("%w: retriever node backend", ragy.ErrInvalidArgument)
		}
	default:
		if n, ok := node.(interface{ validateNode() error }); ok {
			return n.validateNode()
		}
	}
	return nil
}

func validateBinaryNodeTree[TIntent, TRequestMeta, TMeta any](
	primary resultNode[TIntent, TRequestMeta, TMeta],
	primaryLabel string,
	secondary resultNode[TIntent, TRequestMeta, TMeta],
) error {
	if primary == nil {
		return fmt.Errorf("%w: %s", ragy.ErrInvalidArgument, primaryLabel)
	}
	if err := validateNodeTree(primary); err != nil {
		return err
	}
	if secondary != nil {
		return validateNodeTree(secondary)
	}
	return nil
}

func validateAggregateNodeTree[TIntent, TRequestMeta, TMeta any](
	nodes []resultNode[TIntent, TRequestMeta, TMeta],
) error {
	for i, child := range nodes {
		if child == nil {
			return fmt.Errorf("%w: aggregate node child at index %d", ragy.ErrInvalidArgument, i)
		}
		if err := validateNodeTree(child); err != nil {
			return err
		}
	}
	return nil
}

func injectNodeResolver[TIntent, TRequestMeta, TMeta any](
	node resultNode[TIntent, TRequestMeta, TMeta],
	resolver IdentityResolver[TMeta],
) (resultNode[TIntent, TRequestMeta, TMeta], error) {
	if nilvalue.IsNil(node) {
		var zero resultNode[TIntent, TRequestMeta, TMeta]
		return zero, nil // unreachable after validateNodeTree; kept as defense-in-depth
	}
	switch n := node.(type) {
	case resultFallbackNode[TIntent, TRequestMeta, TMeta]:
		return injectFallbackResolver(n, resolver)
	case resultRescueNode[TIntent, TRequestMeta, TMeta]:
		return injectRescueResolver(n, resolver)
	case resultAggregateNode[TIntent, TRequestMeta, TMeta]:
		return injectAggregateResolver(n, resolver)
	case resultConditionalNode[TIntent, TRequestMeta, TMeta]:
		return injectConditionalResolver(n, resolver)
	case resultRetrieverNode[TIntent, TRequestMeta, TMeta]:
		n.Resolver = resolver
		return n, nil
	default:
		if n, ok := node.(interface {
			withResolver(IdentityResolver[TMeta]) (resultNode[TIntent, TRequestMeta, TMeta], error)
		}); ok {
			return n.withResolver(resolver)
		}
		return node, nil
	}
}

func injectFallbackResolver[TIntent, TRequestMeta, TMeta any](
	n resultFallbackNode[TIntent, TRequestMeta, TMeta],
	resolver IdentityResolver[TMeta],
) (resultNode[TIntent, TRequestMeta, TMeta], error) {
	n.Resolver = resolver
	var err error
	n.Primary, err = injectNodeResolver(n.Primary, resolver)
	if err != nil {
		return nil, err
	}
	n.Secondary, err = injectNodeResolver(n.Secondary, resolver)
	if err != nil {
		return nil, err
	}
	return n, nil
}

func injectRescueResolver[TIntent, TRequestMeta, TMeta any](
	n resultRescueNode[TIntent, TRequestMeta, TMeta],
	resolver IdentityResolver[TMeta],
) (resultNode[TIntent, TRequestMeta, TMeta], error) {
	n.Resolver = resolver
	var err error
	n.Primary, err = injectNodeResolver(n.Primary, resolver)
	if err != nil {
		return nil, err
	}
	n.Secondary, err = injectNodeResolver(n.Secondary, resolver)
	if err != nil {
		return nil, err
	}
	return n, nil
}

func injectAggregateResolver[TIntent, TRequestMeta, TMeta any](
	n resultAggregateNode[TIntent, TRequestMeta, TMeta],
	resolver IdentityResolver[TMeta],
) (resultNode[TIntent, TRequestMeta, TMeta], error) {
	n.Resolver = resolver
	for i, child := range n.Nodes {
		rebound, err := injectNodeResolver(child, resolver)
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

func injectConditionalResolver[TIntent, TRequestMeta, TMeta any](
	n resultConditionalNode[TIntent, TRequestMeta, TMeta],
	resolver IdentityResolver[TMeta],
) (resultNode[TIntent, TRequestMeta, TMeta], error) {
	n.Resolver = resolver
	rebound, err := injectNodeResolver(n.Child, resolver)
	if err != nil {
		return nil, err
	}
	n.Child = rebound
	return n, nil
}

func rebindAggregateMerger[TMeta any](
	merger ResultMerger[TMeta],
	resolver IdentityResolver[TMeta],
) (ResultMerger[TMeta], error) {
	if merger != nil && nilvalue.IsNil(merger) {
		return nil, fmt.Errorf("%w: typed-nil aggregate merger", ragy.ErrInvalidArgument)
	}
	switch m := merger.(type) {
	case *DegradingMerger[TMeta]:
		return rebindAggregateMerger(*m, resolver)
	case DegradingMerger[TMeta]:
		if err := m.validate(); err != nil {
			return nil, err
		}
		primary, err := rebindAggregateMerger(m.Primary, resolver)
		if err != nil {
			return nil, err
		}
		fallback, err := rebindAggregateMerger(m.Fallback, resolver)
		if err != nil {
			return nil, err
		}
		return DegradingMerger[TMeta]{Primary: primary, Fallback: fallback}, nil
	case *ScoreMerger[TMeta]:
		return NewScoreMerger(resolver), nil
	case *ReciprocalRankFusion[TMeta]:
		rrf, err := NewReciprocalRankFusion(m.k, resolver)
		if err != nil {
			return nil, fmt.Errorf("rebind aggregate RRF merger: %w", err)
		}
		return rrf, nil
	default:
		// Custom ResultMerger implementations are not rebound; capture resolver in constructor.
		return merger, nil
	}
}

// resultPipeline is a result-shaped adapter over the common execution engine.
type resultPipeline[TIntent, TRequestMeta, TMeta any] struct {
	engine *RequestExecutionPipeline[TIntent, TRequestMeta, TMeta, NoExecutionMeta]
}

func (p *resultPipeline[TIntent, TRequestMeta, TMeta]) Execute(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (RetrievalResult[TMeta, NoExecutionMeta], error) {
	var engine *RequestExecutionPipeline[TIntent, TRequestMeta, TMeta, NoExecutionMeta]
	if p != nil {
		engine = p.engine
	}
	return engine.Execute(ctx, req)
}

// resultExecutionNode translates declarative result syntax, never dispatch policy.
func resultExecutionNode[TIntent, TRequestMeta, TMeta any](
	node resultNode[TIntent, TRequestMeta, TMeta],
) RequestExecutionNode[TIntent, TRequestMeta, TMeta, NoExecutionMeta] {
	if nilvalue.IsNil(node) {
		return nil
	}
	switch n := node.(type) {
	case resultRetrieverNode[TIntent, TRequestMeta, TMeta]:
		return RequestBackendNode[TIntent, TRequestMeta, TMeta, NoExecutionMeta]{
			Backend:  n.Backend,
			Resolver: n.Resolver, Name: "",
		}
	case resultFallbackNode[TIntent, TRequestMeta, TMeta]:
		return RequestFallbackNode[TIntent, TRequestMeta, TMeta, NoExecutionMeta]{
			Primary:   resultExecutionNode(n.Primary),
			Secondary: resultExecutionNode(n.Secondary),
			Resolver:  n.Resolver, Name: "",
		}
	case resultRescueNode[TIntent, TRequestMeta, TMeta]:
		return RequestRescueNode[TIntent, TRequestMeta, TMeta, NoExecutionMeta]{
			Primary:   resultExecutionNode(n.Primary),
			Secondary: resultExecutionNode(n.Secondary),
			Resolver:  n.Resolver, Name: "",
		}
	case resultConditionalNode[TIntent, TRequestMeta, TMeta]:
		return RequestConditionalNode[TIntent, TRequestMeta, TMeta, NoExecutionMeta]{
			Predicate: n.Predicate,
			Child:     resultExecutionNode(n.Child),
			Resolver:  n.Resolver, Name: "",
		}
	case resultAggregateNode[TIntent, TRequestMeta, TMeta]:
		children := make([]RequestExecutionNode[TIntent, TRequestMeta, TMeta, NoExecutionMeta], len(n.Nodes))
		for i, child := range n.Nodes {
			children[i] = resultExecutionNode(child)
		}
		return RequestExecutionAggregateNode[TIntent, TRequestMeta, TMeta, NoExecutionMeta]{
			Nodes:       children,
			Concurrency: n.Concurrency,
			Resolver:    n.Resolver,
			Merger:      n.Merger, MergeExecution: nil, Name: "",
		}
	default:
		return requestNodeExecutionAdapter[TIntent, TRequestMeta, TMeta, NoExecutionMeta]{
			Node:     node,
			Resolver: nil,
			Name:     "",
		}
	}
}

func applyPlannedQuery[TIntent, TRequestMeta any](
	req Request[TIntent, TRequestMeta],
) Request[TIntent, TRequestMeta] {
	if req.Plan == nil {
		return req
	}
	if !filter.IsEmpty(req.Plan.Filters.IR()) {
		req.Options.Filters = req.Plan.Filters
	}
	return req
}
