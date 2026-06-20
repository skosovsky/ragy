package retrieval

import (
	"context"
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
	if resolver == nil {
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
	if resolver == nil {
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

// Process runs configured post-processors on an existing result set.
// ctx is accepted for call-site symmetry with retrieval pipelines.
func (p *PostProcessorChain[TMeta]) Process(
	ctx context.Context,
	opts RetrieveOptions,
	rs ResultSet[TMeta],
) (ResultSet[TMeta], error) {
	_ = ctx
	if p == nil {
		return rs, nil
	}
	if err := opts.Validate(); err != nil {
		return preserveResultOnError(rs, err, p.resolver)
	}
	if rs == nil {
		rs = NewResultSet[TMeta](nil, p.resolver)
	}
	if err := validateResultSet(rs); err != nil {
		return preserveResultOnError(rs, err, p.resolver)
	}

	rs = applyMinSimilarity(rs, opts.MinSimilarity, p.resolver)

	for _, processor := range p.processors {
		var err error
		rs, err = processor.Process(rs)
		if err != nil {
			return preserveResultOnError(rs, err, p.resolver)
		}
		if err := validateResultSet(rs); err != nil {
			return preserveResultOnError(rs, err, p.resolver)
		}
	}

	rs = applyTopK(rs, opts.TopK, p.resolver)
	return rs, nil
}

// applyTerminalOptions applies MinSimilarity and TopK after all post-processors or orchestrator root.
func applyTerminalOptions[TMeta any](
	rs ResultSet[TMeta],
	opts RetrieveOptions,
	resolver IdentityResolver[TMeta],
) ResultSet[TMeta] {
	rs = applyMinSimilarity(rs, opts.MinSimilarity, resolver)
	return applyTopK(rs, opts.TopK, resolver)
}

func validateResultSet[TMeta any](rs ResultSet[TMeta]) error {
	if rs == nil {
		return nil
	}
	for _, doc := range rs.Documents() {
		if err := ValidateDocument(doc); err != nil {
			return ragy.WrapProjectionError(err, "postprocessor validate")
		}
	}
	return nil
}

func applyMinSimilarity[TMeta any](
	rs ResultSet[TMeta],
	minSimilarity float64,
	resolver IdentityResolver[TMeta],
) ResultSet[TMeta] {
	if minSimilarity <= 0 || rs == nil || rs.IsEmpty() {
		return RewrapResultSet(rs, resolver)
	}
	docs := rs.Documents()
	out := make([]Document[TMeta], 0, len(docs))
	for _, doc := range docs {
		if doc.Score >= minSimilarity {
			out = append(out, doc)
		}
	}
	return NewResultSet(out, resolver)
}

func applyTopK[TMeta any](rs ResultSet[TMeta], topK int, resolver IdentityResolver[TMeta]) ResultSet[TMeta] {
	if topK <= 0 || rs == nil {
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
