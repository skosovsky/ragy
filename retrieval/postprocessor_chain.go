package retrieval

import (
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/internal/nilvalue"

	"context"
	"fmt"
	"sort"

	ragy "github.com/skosovsky/ragy"
)

// PostProcessorChain applies post-processors to retrieval results.
type PostProcessorChain[TMeta any] struct {
	processors []PostProcessor[TMeta]
	resolver   IdentityResolver[TMeta]
}

// NewPostProcessorChain constructs a post-processor-only chain without a backend.
func NewPostProcessorChain[TMeta any](processors ...PostProcessor[TMeta]) *PostProcessorChain[TMeta] {
	return NewPostProcessorChainWithResolver[TMeta](DocumentIDResolver[TMeta]{}, processors...)
}

// NewPostProcessorChainWithResolver constructs a post-processor chain with identity resolver.
// Built-in processors (GroupBy, TopPerGroup, Rerank) receive resolver via bindProcessorResolver.
// Custom PostProcessor implementations must capture resolver in their constructor.
func NewPostProcessorChainWithResolver[TMeta any](
	resolver IdentityResolver[TMeta],
	processors ...PostProcessor[TMeta],
) *PostProcessorChain[TMeta] {
	if nilvalue.IsNil(resolver) {
		resolver = DocumentIDResolver[TMeta]{}
	}
	return &PostProcessorChain[TMeta]{
		processors: bindProcessorsResolver(processors, resolver),
		resolver:   resolver,
	}
}

func (p *PostProcessorChain[TMeta]) withResolver(resolver IdentityResolver[TMeta]) *PostProcessorChain[TMeta] {
	if p == nil {
		return nil
	}
	if nilvalue.IsNil(resolver) {
		resolver = DocumentIDResolver[TMeta]{}
	}
	clone := *p
	clone.resolver = resolver
	clone.processors = bindProcessorsResolver(p.processors, resolver)
	return &clone
}

func bindProcessorsResolver[TMeta any](
	processors []PostProcessor[TMeta],
	resolver IdentityResolver[TMeta],
) []PostProcessor[TMeta] {
	if len(processors) == 0 {
		return nil
	}
	out := make([]PostProcessor[TMeta], len(processors))
	for i, processor := range processors {
		if nilvalue.IsNil(processor) {
			processor = invalidPostProcessorFor[TMeta](fmt.Errorf("%w: nil postprocessor", ragy.ErrInvalidArgument))
		}
		out[i] = bindProcessorResolver(processor, resolver)
	}
	return out
}

func bindProcessorResolver[TMeta any](
	processor PostProcessor[TMeta],
	resolver IdentityResolver[TMeta],
) PostProcessor[TMeta] {
	// Custom PostProcessor types must capture resolver in their constructor.
	switch proc := processor.(type) {
	case groupByProcessor[TMeta]:
		proc.resolver = resolver
		return proc
	case topPerGroupProcessor[TMeta]:
		proc.resolver = resolver
		return proc
	case rerankProcessor[TMeta]:
		proc.resolver = resolver
		return proc
	case invalidPostProcessor[TMeta]:
		proc.resolver = resolver
		return proc
	default:
		return processor
	}
}

// Process enforces freshness before each processor and at every delivery path.
func (p *PostProcessorChain[TMeta]) Process(
	ctx context.Context,
	read access.Binding,
	opts RetrieveOptions,
	rs ResultSet[TMeta],
) (ResultSet[TMeta], error) {
	out, err := p.process(ctx, read, opts, rs)
	var resolver IdentityResolver[TMeta]
	if p != nil {
		resolver = p.resolver
	}
	return DeliverRead(ctx, read, out, err, resolver)
}

func (p *PostProcessorChain[TMeta]) process(
	ctx context.Context,
	read access.Binding, opts RetrieveOptions,
	rs ResultSet[TMeta],
) (ResultSet[TMeta], error) {
	if p == nil {
		return rs, nil
	}
	if err := read.Check(ctx); err != nil {
		return NewResultSet[TMeta](nil, p.resolver), err
	}
	if err := opts.Validate(); err != nil {
		return preserveResultOnError(rs, err, p.resolver)
	}
	for _, processor := range p.processors {
		if invalid, ok := processor.(invalidPostProcessor[TMeta]); ok {
			return preserveResultOnError(rs, invalid.err, p.resolver)
		}
	}
	if nilvalue.IsNil(rs) {
		rs = NewResultSet[TMeta](nil, p.resolver)
	}
	if err := validateResultSet(rs); err != nil {
		return preserveResultOnError(rs, err, p.resolver)
	}

	var thresholdErr error
	rs, thresholdErr = applyScoreThreshold(rs, opts.Threshold, p.resolver)
	if thresholdErr != nil {
		return preserveResultOnError(rs, thresholdErr, p.resolver)
	}

	for _, processor := range p.processors {
		if err := read.Check(ctx); err != nil {
			return NewResultSet[TMeta](nil, p.resolver), err
		}
		var err error
		rs, err = processor.Process(ctx, read, rs)
		rs, err = DeliverRead(ctx, read, rs, err, p.resolver)
		if err != nil {
			return preserveResultOnError(rs, err, p.resolver)
		}
		if err := validateResultSet(rs); err != nil {
			return preserveResultOnError(rs, err, p.resolver)
		}
	}

	if err := read.Check(ctx); err != nil {
		return NewResultSet[TMeta](nil, p.resolver), err
	}
	if err := validateComparable(rs.Documents()); err != nil {
		return preserveResultOnError(rs, err, p.resolver)
	}
	rs = applyTopK(rs, opts.TopK, p.resolver)
	return rs, nil
}

// applyTerminalOptions applies score threshold and TopK after all post-processors or orchestrator root.
func applyTerminalOptions[TMeta any](
	rs ResultSet[TMeta],
	opts RetrieveOptions,
	resolver IdentityResolver[TMeta],
) (ResultSet[TMeta], error) {
	if err := opts.Validate(); err != nil {
		return preserveResultOnError(rs, err, resolver)
	}
	if rs != nil {
		if err := validateComparable(rs.Documents()); err != nil {
			return preserveResultOnError(rs, err, resolver)
		}
	}
	filtered, err := applyScoreThreshold(rs, opts.Threshold, resolver)
	if err != nil {
		return preserveResultOnError(filtered, err, resolver)
	}
	return applyTopK(filtered, opts.TopK, resolver), nil
}

func validateResultSet[TMeta any](rs ResultSet[TMeta]) error {
	if nilvalue.IsNil(rs) {
		return nil
	}
	for _, doc := range rs.Documents() {
		if err := ValidateDocument(doc); err != nil {
			return ragy.WrapProjectionError(err, "postprocessor validate")
		}
	}
	return nil
}

func applyScoreThreshold[TMeta any](
	rs ResultSet[TMeta],
	threshold *ScoreThreshold,
	resolver IdentityResolver[TMeta],
) (ResultSet[TMeta], error) {
	if threshold == nil || nilvalue.IsNil(rs) || rs.IsEmpty() {
		return RewrapResultSet(rs, resolver), nil
	}
	if err := threshold.Validate(); err != nil {
		return rs, err
	}
	docs := rs.Documents()
	for _, doc := range docs {
		if !doc.ScoreState.IsScored() || doc.ScoreState != threshold.State ||
			doc.ScoreSemantics != threshold.Semantics {
			return rs, fmt.Errorf("%w: threshold cannot compare score scale", ragy.ErrInvalidArgument)
		}
	}
	out := make([]Document[TMeta], 0, len(docs))
	for _, doc := range docs {
		if doc.Score >= threshold.Value {
			out = append(out, doc)
		}
	}
	return NewResultSet(out, resolver), nil
}

func applyTopK[TMeta any](rs ResultSet[TMeta], topK int, resolver IdentityResolver[TMeta]) ResultSet[TMeta] {
	if topK <= 0 || nilvalue.IsNil(rs) {
		return RewrapResultSet(rs, resolver)
	}
	docs := rs.Documents()
	if len(docs) == 0 {
		return NewResultSet(docs, resolver)
	}
	if len(docs) > 1 {
		sort.SliceStable(docs, func(i, j int) bool {
			return rankedDocumentLess(docs[i], docs[j])
		})
	}
	if topK > 0 && len(docs) > topK {
		docs = docs[:topK]
	}
	return NewResultSet(docs, resolver)
}
