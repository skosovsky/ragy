package retrieval

import (
	"context"
	"errors"
	"fmt"

	ragy "github.com/skosovsky/ragy"
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

// Node executes retrieval for a query and always returns a non-nil ResultSet.
type Node[TIntent, TMeta any] interface {
	Retrieve(ctx context.Context, query Query[TIntent]) (ResultSet[TMeta], error)
}

// RetrieverNode wraps a Backend as an orchestrator node.
type RetrieverNode[TIntent, TMeta any] struct {
	Backend  Backend[TMeta]
	Resolver IdentityResolver[TMeta]
}

// Retrieve implements Node.
func (n RetrieverNode[TIntent, TMeta]) Retrieve(
	ctx context.Context,
	query Query[TIntent],
) (ResultSet[TMeta], error) {
	resolver := n.Resolver
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	if n.Backend == nil {
		return NewResultSet[TMeta](nil, resolver),
			fmt.Errorf("%w: retriever node backend", ragy.ErrInvalidArgument)
	}
	if err := query.Options.Validate(); err != nil {
		return NewResultSet[TMeta](nil, resolver), err
	}
	rs, err := n.Backend.Retrieve(ctx, query.Text, query.Options)
	if err != nil {
		return preserveResultOnError(rs, err, resolver)
	}
	if rs == nil {
		return NewResultSet[TMeta](nil, resolver), nil
	}
	return RewrapResultSet(rs, resolver), nil
}

// FallbackNode runs secondary when primary succeeds (err == nil) and ResultSet is empty.
// On primary error with empty ResultSet, the error is propagated and secondary is not called.
// On partial success (error with non-empty docs), primary documents are preserved.
type FallbackNode[TIntent, TMeta any] struct {
	Primary   Node[TIntent, TMeta]
	Secondary Node[TIntent, TMeta]
	Resolver  IdentityResolver[TMeta]
}

// Retrieve implements Node.
func (n FallbackNode[TIntent, TMeta]) Retrieve(
	ctx context.Context,
	query Query[TIntent],
) (ResultSet[TMeta], error) {
	resolver := n.Resolver
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	if n.Primary == nil {
		return NewResultSet[TMeta](nil, resolver),
			fmt.Errorf("%w: fallback primary node", ragy.ErrInvalidArgument)
	}

	primary, err := n.Primary.Retrieve(ctx, query)
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
	rs, err := n.Secondary.Retrieve(ctx, query)
	if err != nil {
		return preserveResultOnError(rs, err, resolver)
	}
	return RewrapResultSet(rs, resolver), nil
}

// RescueNode runs secondary when primary returns an error and ResultSet is empty.
// On primary success with empty ResultSet, returns empty without calling secondary.
// On partial success, preserves primary documents.
// Rescue with non-empty secondary returns nil error; empty secondary propagates primary error.
type RescueNode[TIntent, TMeta any] struct {
	Primary   Node[TIntent, TMeta]
	Secondary Node[TIntent, TMeta]
	Resolver  IdentityResolver[TMeta]
}

// Retrieve implements Node.
func (n RescueNode[TIntent, TMeta]) Retrieve(
	ctx context.Context,
	query Query[TIntent],
) (ResultSet[TMeta], error) {
	resolver := n.Resolver
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	if n.Primary == nil {
		return NewResultSet[TMeta](nil, resolver),
			fmt.Errorf("%w: rescue primary node", ragy.ErrInvalidArgument)
	}

	primary, err := n.Primary.Retrieve(ctx, query)
	if err != nil {
		if partialSuccessRS(primary, err) {
			return preserveResultOnError(primary, err, resolver)
		}
		if n.Secondary == nil {
			return NewResultSet[TMeta](nil, resolver), err
		}
		secondary, secErr := n.Secondary.Retrieve(ctx, query)
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

// AggregateNode runs child nodes in parallel and merges their ResultSets.
// When Merger is nil, ReciprocalRankFusion is used (recommended for heterogeneous sources).
// For homogeneous score scales, set Merger to NewScoreMerger explicitly.
// When merger.Merge fails, degraded fallback uses sequential ResultSet.Merge (score-by-MergeKey),
// not RRF — ordering may differ from the success-path merger.
type AggregateNode[TIntent, TMeta any] struct {
	Nodes       []Node[TIntent, TMeta]
	Concurrency int
	Resolver    IdentityResolver[TMeta]
	Merger      ResultMerger[TMeta]
}

// aggregateChildResult captures one aggregate branch outcome.
type aggregateChildResult[TMeta any] struct {
	rs  ResultSet[TMeta]
	err error
}

// Retrieve implements Node.
func (n AggregateNode[TIntent, TMeta]) Retrieve(
	ctx context.Context,
	query Query[TIntent],
) (ResultSet[TMeta], error) {
	resolver := n.Resolver
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	if len(n.Nodes) == 0 {
		return NewResultSet[TMeta](nil, resolver), nil
	}

	nodes := make([]Node[TIntent, TMeta], 0, len(n.Nodes))
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
		func(ctx context.Context, node Node[TIntent, TMeta]) (aggregateChildResult[TMeta], error) {
			return runAggregateChild(ctx, node, query, resolver), nil
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

func runAggregateChild[TIntent, TMeta any](
	ctx context.Context,
	node Node[TIntent, TMeta],
	query Query[TIntent],
	resolver IdentityResolver[TMeta],
) aggregateChildResult[TMeta] {
	rs, retrieveErr := node.Retrieve(ctx, query)
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

// ConditionalNode skips the child when predicate is false.
type ConditionalNode[TIntent, TMeta any] struct {
	Predicate func(Query[TIntent]) bool
	Child     Node[TIntent, TMeta]
	Resolver  IdentityResolver[TMeta]
}

// Retrieve implements Node.
func (n ConditionalNode[TIntent, TMeta]) Retrieve(
	ctx context.Context,
	query Query[TIntent],
) (ResultSet[TMeta], error) {
	resolver := n.Resolver
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	// Nil Predicate is treated as always true (child always runs).
	if n.Predicate != nil && !n.Predicate(query) {
		return NewResultSet[TMeta](nil, resolver), nil
	}
	if n.Child == nil {
		return NewResultSet[TMeta](nil, resolver),
			fmt.Errorf("%w: conditional child node", ragy.ErrInvalidArgument)
	}
	rs, err := n.Child.Retrieve(ctx, query)
	if err != nil {
		return preserveResultOnError(rs, err, resolver)
	}
	if rs == nil {
		return NewResultSet[TMeta](nil, resolver), nil
	}
	return RewrapResultSet(rs, resolver), nil
}

type PipelineBuilder[TIntent, TMeta any] struct {
	root      Node[TIntent, TMeta]
	postChain *PostProcessorChain[TMeta]
	resolver  IdentityResolver[TMeta]
}

// NewPipelineBuilder starts orchestrator construction.
func NewPipelineBuilder[TIntent, TMeta any]() *PipelineBuilder[TIntent, TMeta] {
	return &PipelineBuilder[TIntent, TMeta]{}
}

// WithRoot sets the root retrieval node.
func (b *PipelineBuilder[TIntent, TMeta]) WithRoot(node Node[TIntent, TMeta]) *PipelineBuilder[TIntent, TMeta] {
	b.root = node
	return b
}

// WithFallback configures primary/secondary fallback routing.
// Shorthand methods (WithFallback, WithRescue, WithAggregate, WithConditional) replace the
// current root node. Compose complex graphs via WithRoot explicitly.
func (b *PipelineBuilder[TIntent, TMeta]) WithFallback(
	primary, secondary Node[TIntent, TMeta],
) *PipelineBuilder[TIntent, TMeta] {
	b.root = FallbackNode[TIntent, TMeta]{Primary: primary, Secondary: secondary}
	return b
}

// WithRescue configures primary/secondary rescue routing on primary errors.
func (b *PipelineBuilder[TIntent, TMeta]) WithRescue(
	primary, secondary Node[TIntent, TMeta],
) *PipelineBuilder[TIntent, TMeta] {
	b.root = RescueNode[TIntent, TMeta]{ //nolint:exhaustruct // Resolver injected in Build()
		Primary:   primary,
		Secondary: secondary,
	}
	return b
}

// WithAggregate configures parallel aggregate routing.
// Pass nil merger to use ReciprocalRankFusion (recommended for heterogeneous sources).
func (b *PipelineBuilder[TIntent, TMeta]) WithAggregate(
	nodes []Node[TIntent, TMeta],
	concurrency int,
	merger ResultMerger[TMeta],
) *PipelineBuilder[TIntent, TMeta] {
	b.root = AggregateNode[TIntent, TMeta]{
		Nodes:       nodes,
		Concurrency: concurrency,
		Merger:      merger,
	}
	return b
}

// WithConditional wraps a node behind a predicate.
func (b *PipelineBuilder[TIntent, TMeta]) WithConditional(
	predicate func(Query[TIntent]) bool,
	child Node[TIntent, TMeta],
) *PipelineBuilder[TIntent, TMeta] {
	b.root = ConditionalNode[TIntent, TMeta]{Predicate: predicate, Child: child}
	return b
}

// WithPostProcessors attaches a post-processing chain after retrieval.
// Replaces any previously configured post-processor chain. Shorthand root methods do not clear postChain.
func (b *PipelineBuilder[TIntent, TMeta]) WithPostProcessors(
	processors ...PostProcessor[TMeta],
) *PipelineBuilder[TIntent, TMeta] {
	b.postChain = NewPostProcessorChain[TMeta](processors...)
	return b
}

// WithResolver sets the identity resolver for known node types and post-processors.
// Custom Node implementations (types not handled by injectNodeResolver) are not
// modified; set Resolver on those nodes explicitly before Build.
func (b *PipelineBuilder[TIntent, TMeta]) WithResolver(
	resolver IdentityResolver[TMeta],
) *PipelineBuilder[TIntent, TMeta] {
	b.resolver = resolver
	return b
}

// Build returns the configured orchestrator pipeline.
func (b *PipelineBuilder[TIntent, TMeta]) Build() (*Pipeline[TIntent, TMeta], error) {
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
	return &Pipeline[TIntent, TMeta]{
		root:      root,
		postChain: postChain,
		resolver:  resolver,
	}, nil
}

func validateNodeTree[TIntent, TMeta any](node Node[TIntent, TMeta]) error {
	if node == nil {
		return fmt.Errorf("%w: pipeline node", ragy.ErrInvalidArgument)
	}
	switch n := node.(type) {
	case FallbackNode[TIntent, TMeta]:
		return validateBinaryNodeTree(n.Primary, "fallback primary node", n.Secondary)
	case RescueNode[TIntent, TMeta]:
		return validateBinaryNodeTree(n.Primary, "rescue primary node", n.Secondary)
	case AggregateNode[TIntent, TMeta]:
		return validateAggregateNodeTree(n.Nodes)
	case ConditionalNode[TIntent, TMeta]:
		if n.Child == nil {
			return fmt.Errorf("%w: conditional child node", ragy.ErrInvalidArgument)
		}
		return validateNodeTree(n.Child)
	case RetrieverNode[TIntent, TMeta]:
		if n.Backend == nil {
			return fmt.Errorf("%w: retriever node backend", ragy.ErrInvalidArgument)
		}
	}
	return nil
}

func validateBinaryNodeTree[TIntent, TMeta any](
	primary Node[TIntent, TMeta],
	primaryLabel string,
	secondary Node[TIntent, TMeta],
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

func validateAggregateNodeTree[TIntent, TMeta any](nodes []Node[TIntent, TMeta]) error {
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

func injectNodeResolver[TIntent, TMeta any](
	node Node[TIntent, TMeta],
	resolver IdentityResolver[TMeta],
) (Node[TIntent, TMeta], error) {
	if node == nil {
		var zero Node[TIntent, TMeta]
		return zero, nil // unreachable after validateNodeTree; kept as defense-in-depth
	}
	switch n := node.(type) {
	case FallbackNode[TIntent, TMeta]:
		return injectFallbackResolver(n, resolver)
	case RescueNode[TIntent, TMeta]:
		return injectRescueResolver(n, resolver)
	case AggregateNode[TIntent, TMeta]:
		return injectAggregateResolver(n, resolver)
	case ConditionalNode[TIntent, TMeta]:
		return injectConditionalResolver(n, resolver)
	case RetrieverNode[TIntent, TMeta]:
		n.Resolver = resolver
		return n, nil
	default:
		return node, nil
	}
}

func injectFallbackResolver[TIntent, TMeta any](
	n FallbackNode[TIntent, TMeta],
	resolver IdentityResolver[TMeta],
) (Node[TIntent, TMeta], error) {
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

func injectRescueResolver[TIntent, TMeta any](
	n RescueNode[TIntent, TMeta],
	resolver IdentityResolver[TMeta],
) (Node[TIntent, TMeta], error) {
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

func injectAggregateResolver[TIntent, TMeta any](
	n AggregateNode[TIntent, TMeta],
	resolver IdentityResolver[TMeta],
) (Node[TIntent, TMeta], error) {
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

func injectConditionalResolver[TIntent, TMeta any](
	n ConditionalNode[TIntent, TMeta],
	resolver IdentityResolver[TMeta],
) (Node[TIntent, TMeta], error) {
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

// Pipeline is a declarative retrieval orchestrator.
type Pipeline[TIntent, TMeta any] struct {
	root      Node[TIntent, TMeta]
	postChain *PostProcessorChain[TMeta]
	resolver  IdentityResolver[TMeta]
}

// Retrieve executes the configured graph and optional post-processors.
func (p *Pipeline[TIntent, TMeta]) Retrieve(
	ctx context.Context,
	query Query[TIntent],
) (ResultSet[TMeta], error) {
	if p == nil || p.root == nil {
		return NewResultSet[TMeta](nil, DocumentIDResolver[TMeta]{}),
			fmt.Errorf("%w: pipeline root", ragy.ErrInvalidArgument)
	}
	if err := query.Options.Validate(); err != nil {
		return NewResultSet[TMeta](nil, p.resolver), err
	}

	rs, err := p.root.Retrieve(ctx, query)
	partialErr := err
	if partialErr != nil {
		rs, _ = preserveResultOnError(rs, partialErr, p.resolver)
	} else if rs == nil {
		rs = NewResultSet[TMeta](nil, p.resolver)
	}
	if p.postChain != nil {
		var postErr error
		rs, postErr = p.postChain.Process(ctx, query.Text, query.Options, rs)
		if postErr != nil {
			rs, _ = preserveResultOnError(rs, postErr, p.resolver)
			final := NewResultSet(rs.Documents(), p.resolver)
			if partialErr != nil {
				return final, errors.Join(syncPartialFailureResult(partialErr, final), postErr)
			}
			return final, postErr
		}
	} else {
		rs = applyTerminalOptions(rs, query.Options, p.resolver)
	}
	final := NewResultSet(rs.Documents(), p.resolver)
	if partialErr != nil {
		return final, syncPartialFailureResult(partialErr, final)
	}
	return final, nil
}
