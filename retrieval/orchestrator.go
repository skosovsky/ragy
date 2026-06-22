package retrieval

import (
	"context"
	"errors"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/internal/parallel"
)

// partialSuccessRS reports whether err should preserve a non-empty primary ResultSet
// rather than invoke Fallback secondary or Rescue secondary.
// True when err wraps PartialFailureError with documents, or when rs is non-empty.
// Also true for any non-empty rs with error (e.g. adapter PreserveResultOnError),
// which blocks Fallback/Rescue secondary per spec "empty ResultSet only".
func partialSuccessRS[TMeta any](rs ResultSet[TMeta], err error) bool {
	if err == nil {
		return false
	}
	if partial, ok := AsPartialFailure[TMeta](err); ok && partial != nil && !partial.Result.IsEmpty() {
		return true
	}
	return rs != nil && !rs.IsEmpty()
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
	resolver := n.Resolver
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	if n.Backend == nil {
		return NewResultSet[TMeta](nil, resolver),
			fmt.Errorf("%w: retriever node backend", ragy.ErrInvalidArgument)
	}
	if err := req.Options.Validate(); err != nil {
		return NewResultSet[TMeta](nil, resolver), err
	}
	rs, err := n.Backend.Retrieve(ctx, req)
	if err != nil {
		return preserveResultOnError(rs, err, resolver)
	}
	if rs == nil {
		return NewResultSet[TMeta](nil, resolver), nil
	}
	return RewrapResultSet(rs, resolver), nil
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
	resolver := n.Resolver
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	if n.Primary == nil {
		return NewResultSet[TMeta](nil, resolver),
			fmt.Errorf("%w: fallback primary node", ragy.ErrInvalidArgument)
	}

	primary, err := n.Primary.Retrieve(ctx, req)
	if err != nil {
		if partialSuccessRS(primary, err) {
			return preserveResultOnError(primary, err, resolver)
		}
		return NewResultSet[TMeta](nil, resolver), err
	}
	if primary != nil && !primary.IsEmpty() {
		return RewrapResultSet(primary, resolver), nil
	}
	if n.Secondary == nil {
		return NewResultSet[TMeta](nil, resolver), nil
	}
	rs, err := n.Secondary.Retrieve(ctx, req)
	if err != nil {
		return preserveResultOnError(rs, err, resolver)
	}
	return RewrapResultSet(rs, resolver), nil
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
	resolver := n.Resolver
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	if n.Primary == nil {
		return NewResultSet[TMeta](nil, resolver),
			fmt.Errorf("%w: rescue primary node", ragy.ErrInvalidArgument)
	}

	primary, err := n.Primary.Retrieve(ctx, req)
	if err != nil {
		if partialSuccessRS(primary, err) {
			return preserveResultOnError(primary, err, resolver)
		}
		if n.Secondary == nil {
			return NewResultSet[TMeta](nil, resolver), err
		}
		secondary, secErr := n.Secondary.Retrieve(ctx, req)
		if secErr != nil {
			wrapped := fmt.Errorf("%w; rescue secondary: %w", err, secErr)
			return preserveResultOnError(secondary, wrapped, resolver)
		}
		if secondary.IsEmpty() {
			return NewResultSet[TMeta](nil, resolver), fmt.Errorf("%w: rescue secondary empty", err)
		}
		return RewrapResultSet(secondary, resolver), nil
	}
	if primary != nil && !primary.IsEmpty() {
		return RewrapResultSet(primary, resolver), nil
	}
	return NewResultSet[TMeta](nil, resolver), nil
}

// resultAggregateNode runs child nodes in parallel and merges their ResultSets.
// When Merger is nil, ReciprocalRankFusion is used (recommended for heterogeneous sources).
// For homogeneous score scales, set Merger to NewScoreMerger explicitly.
// When merger.Merge fails, degraded fallback uses sequential ResultSet.Merge (score-by-MergeKey),
// not RRF — ordering may differ from the success-path merger.
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
	resolver := n.Resolver
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	if len(n.Nodes) == 0 {
		return NewResultSet[TMeta](nil, resolver), nil
	}

	nodes := make([]resultNode[TIntent, TRequestMeta, TMeta], 0, len(n.Nodes))
	for i, node := range n.Nodes {
		if node == nil {
			return NewResultSet[TMeta](nil, resolver),
				fmt.Errorf("%w: aggregate node child at index %d", ragy.ErrInvalidArgument, i)
		}
		nodes = append(nodes, node)
	}

	concurrency := n.Concurrency
	if concurrency <= 0 {
		concurrency = len(nodes)
	}

	sets, err := parallel.MapOrdered(
		ctx,
		concurrency,
		nodes,
		func(ctx context.Context, node resultNode[TIntent, TRequestMeta, TMeta]) (aggregateChildResult[TMeta], error) {
			return runAggregateChild(ctx, node, req, resolver), nil
		},
	)
	if err != nil {
		merger, mergeResolveErr := resolveAggregateMerger(n.Merger, resolver)
		if mergeResolveErr != nil {
			return NewResultSet[TMeta](nil, resolver), mergeResolveErr
		}
		if partialSets, hasPartial := partialAggregateChildResults(sets); hasPartial {
			rs, finalizeErr := finalizeAggregateRetrieve(ctx, resolver, merger, partialSets)
			return preserveResultOnError(rs, errors.Join(err, finalizeErr), resolver)
		}
		return NewResultSet[TMeta](nil, resolver), err
	}

	merger, err := resolveAggregateMerger(n.Merger, resolver)
	if err != nil {
		return NewResultSet[TMeta](nil, resolver), err
	}
	return finalizeAggregateRetrieve(ctx, resolver, merger, sets)
}

func resolveAggregateMerger[TMeta any](
	merger ResultMerger[TMeta],
	resolver IdentityResolver[TMeta],
) (ResultMerger[TMeta], error) {
	if merger != nil {
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
			if result.rs != nil && !result.rs.IsEmpty() {
				successSets = append(successSets, result.rs)
			}
			continue
		}
		if result.rs != nil && !result.rs.IsEmpty() {
			successSets = append(successSets, result.rs)
		}
	}

	merged, mergeErr := merger.Merge(ctx, successSets...)
	if mergeErr != nil {
		return aggregateMergeFailureResult(resolver, successSets, childErrors, mergeErr)
	}
	if !merged.IsEmpty() {
		if len(childErrors) > 0 {
			return merged, &PartialFailureError[TMeta]{Errors: childErrors, Result: merged}
		}
		return merged, nil
	}
	if len(childErrors) > 0 {
		return aggregatePartialWithChildErrors(resolver, successSets, childErrors)
	}
	return merged, nil
}

func aggregatePartialWithChildErrors[TMeta any](
	resolver IdentityResolver[TMeta],
	successSets []ResultSet[TMeta],
	childErrors []error,
) (ResultSet[TMeta], error) {
	fallback, fbErr := tryAggregateFallback(successSets, resolver)
	if fallback != nil && !fallback.IsEmpty() {
		errs := append([]error{}, childErrors...)
		if fbErr != nil {
			errs = append(errs, fbErr)
		}
		return fallback, &PartialFailureError[TMeta]{Errors: errs, Result: fallback}
	}
	return NewResultSet[TMeta](nil, resolver), errors.Join(childErrors...)
}

func aggregateMergeFailureResult[TMeta any](
	resolver IdentityResolver[TMeta],
	successSets []ResultSet[TMeta],
	childErrors []error,
	mergeErr error,
) (ResultSet[TMeta], error) {
	fallback, fbErr := tryAggregateFallback(successSets, resolver)
	if fallback != nil && !fallback.IsEmpty() {
		errs := append(append([]error{}, childErrors...), mergeErr)
		if fbErr != nil {
			errs = append(errs, fbErr)
		}
		return fallback, &PartialFailureError[TMeta]{Errors: errs, Result: fallback}
	}
	errs := append(append([]error{}, childErrors...), mergeErr)
	if fbErr != nil {
		errs = append(errs, fbErr)
	}
	return NewResultSet[TMeta](nil, resolver), errors.Join(errs...)
}

func tryAggregateFallback[TMeta any](
	successSets []ResultSet[TMeta],
	resolver IdentityResolver[TMeta],
) (ResultSet[TMeta], error) {
	// Score-merge fallback: sequential ResultSet.Merge, not RRF.
	fallback, fbErr := fallbackUnmergedSets(successSets, resolver)
	if fbErr != nil {
		fallback, _ = preserveResultOnError(fallback, fbErr, resolver)
	}
	return fallback, fbErr
}

func runAggregateChild[TIntent, TRequestMeta, TMeta any](
	ctx context.Context,
	node resultNode[TIntent, TRequestMeta, TMeta],
	req Request[TIntent, TRequestMeta],
	resolver IdentityResolver[TMeta],
) aggregateChildResult[TMeta] {
	rs, retrieveErr := node.Retrieve(ctx, req)
	if retrieveErr != nil {
		rs, _ = preserveResultOnError(rs, retrieveErr, resolver)
		return aggregateChildResult[TMeta]{
			rs:  rs,
			err: retrieveErr,
		}
	}
	if rs == nil {
		rs = NewResultSet[TMeta](nil, resolver)
	}
	return aggregateChildResult[TMeta]{rs: RewrapResultSet(rs, resolver), err: nil}
}

func partialAggregateChildResults[TMeta any](
	sets []aggregateChildResult[TMeta],
) ([]aggregateChildResult[TMeta], bool) {
	if len(sets) == 0 {
		return nil, false
	}
	out := make([]aggregateChildResult[TMeta], 0, len(sets))
	for _, result := range sets {
		if result.err != nil || (result.rs != nil && !result.rs.IsEmpty()) {
			out = append(out, result)
		}
	}
	return out, len(out) > 0
}

func fallbackUnmergedSets[TMeta any](
	sets []ResultSet[TMeta],
	resolver IdentityResolver[TMeta],
) (ResultSet[TMeta], error) {
	if len(sets) == 0 {
		return NewResultSet[TMeta](nil, resolver), nil
	}
	merged := NewResultSet[TMeta](nil, resolver)
	for _, set := range sets {
		var err error
		merged, err = merged.Merge(set)
		if err != nil {
			return merged, err
		}
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
	resolver := n.Resolver
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	// Nil Predicate is treated as always true (child always runs).
	if n.Predicate != nil && !n.Predicate(req) {
		return NewResultSet[TMeta](nil, resolver), nil
	}
	if n.Child == nil {
		return NewResultSet[TMeta](nil, resolver),
			fmt.Errorf("%w: conditional child node", ragy.ErrInvalidArgument)
	}
	rs, err := n.Child.Retrieve(ctx, req)
	if err != nil {
		return preserveResultOnError(rs, err, resolver)
	}
	if rs == nil {
		return NewResultSet[TMeta](nil, resolver), nil
	}
	return RewrapResultSet(rs, resolver), nil
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
	b.root = resultRescueNode[TIntent, TRequestMeta, TMeta]{ //nolint:exhaustruct // Resolver injected in Build()
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
	if b.root == nil {
		return nil, fmt.Errorf("%w: pipeline root node", ragy.ErrInvalidArgument)
	}
	if err := validateNodeTree(b.root); err != nil {
		return nil, err
	}
	resolver := b.resolver
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	root, err := injectNodeResolver(b.root, resolver)
	if err != nil {
		return nil, err
	}
	postChain := b.postChain
	if postChain != nil {
		postChain = postChain.withResolver(resolver)
	}
	return &resultPipeline[TIntent, TRequestMeta, TMeta]{
		root:      root,
		postChain: postChain,
		resolver:  resolver,
		planner:   b.planner,
		binder:    b.binder,
	}, nil
}

func validateNodeTree[TIntent, TRequestMeta, TMeta any](
	node resultNode[TIntent, TRequestMeta, TMeta],
) error {
	if node == nil {
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
	if node == nil {
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
	switch m := merger.(type) {
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

// resultPipeline is a declarative retrieval orchestrator.
type resultPipeline[TIntent, TRequestMeta, TMeta any] struct {
	root      resultNode[TIntent, TRequestMeta, TMeta]
	postChain *PostProcessorChain[TMeta]
	resolver  IdentityResolver[TMeta]
	planner   QueryPlanner[TIntent, TRequestMeta]
	binder    RequestPlanBinder[TIntent, TRequestMeta, NoExecutionMeta]
}

// Execute runs planner, binder, retrieval graph, and optional post-processors.
func (p *resultPipeline[TIntent, TRequestMeta, TMeta]) Execute(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (RetrievalResult[TMeta, NoExecutionMeta], error) {
	exec := NoExecutionMeta{}
	if p == nil || p.root == nil {
		return emptyRetrievalResult(DocumentIDResolver[TMeta]{}, exec),
			fmt.Errorf("%w: pipeline root", ragy.ErrInvalidArgument)
	}
	var diagnostics []ExecutionDiagnostic
	var planErr error
	req, diagnostics, planErr = p.planQuery(ctx, req)
	if planErr != nil {
		return RetrievalResult[TMeta, NoExecutionMeta]{
			ResultSet:   NewResultSet[TMeta](nil, p.resolver),
			Executed:    exec,
			Diagnostics: diagnostics,
			BranchTrace: pipelineErrorTrace("planner", planErr),
		}, planErr
	}
	if p.binder != nil {
		bound, bindErr := p.binder.BindPlan(ctx, req, req.Plan, exec)
		diagnostics = append(diagnostics, bound.Diagnostics...)
		if bindErr != nil {
			return RetrievalResult[TMeta, NoExecutionMeta]{
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
		return RetrievalResult[TMeta, NoExecutionMeta]{
			ResultSet:   NewResultSet[TMeta](nil, p.resolver),
			Executed:    exec,
			Diagnostics: diagnostics,
			BranchTrace: pipelineErrorTrace("options", err),
		}, err
	}

	rs, err := p.root.Retrieve(ctx, req)
	partialErr := err
	if partialErr != nil {
		rs, _ = preserveResultOnError(rs, partialErr, p.resolver)
	} else if rs == nil {
		rs = NewResultSet[TMeta](nil, p.resolver)
	}
	result := RetrievalResult[TMeta, NoExecutionMeta]{
		ResultSet:   ensureResultSet(rs, p.resolver),
		Executed:    exec,
		Diagnostics: diagnostics,
		BranchTrace: []BranchStep{nodeBranchStep("root", resultState(rs, partialErr), partialErr)},
	}
	if p.postChain != nil {
		var postErr error
		result.ResultSet, postErr = p.postChain.Process(ctx, req.Options, result.ResultSet)
		if postErr != nil {
			result.ResultSet, _ = preserveResultOnError(result.ResultSet, postErr, p.resolver)
			final := NewResultSet(result.ResultSet.Documents(), p.resolver)
			result.ResultSet = final
			if partialErr != nil {
				return result, errors.Join(syncPartialFailureResult(partialErr, final), postErr)
			}
			return result, postErr
		}
	} else {
		result.ResultSet = applyTerminalOptions(result.ResultSet, req.Options, p.resolver)
	}
	final := NewResultSet(result.ResultSet.Documents(), p.resolver)
	result.ResultSet = final
	if partialErr != nil {
		return result, syncPartialFailureResult(partialErr, final)
	}
	return result, nil
}

func (p *resultPipeline[TIntent, TRequestMeta, TMeta]) planQuery(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
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
