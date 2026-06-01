package retrieval

import (
	"context"
	"fmt"

	ragy "github.com/skosovsky/ragy"
)

// Pipeline applies post-processors after a backend retrieve call.
type Pipeline[TMeta any] struct {
	backend    Backend[TMeta]
	processors []PostProcessor[TMeta]
}

// NewPipeline constructs a retriever from a backend and optional processors.
func NewPipeline[TMeta any](backend Backend[TMeta], processors ...PostProcessor[TMeta]) *Pipeline[TMeta] {
	return &Pipeline[TMeta]{
		backend:    backend,
		processors: append([]PostProcessor[TMeta](nil), processors...),
	}
}

// Retrieve implements Retriever.
func (p *Pipeline[TMeta]) Retrieve(ctx context.Context, query string, opts RetrieveOptions) ([]Document[TMeta], error) {
	if p == nil || p.backend == nil {
		return nil, fmt.Errorf("%w: retrieval pipeline backend", ragy.ErrInvalidArgument)
	}
	if err := opts.Validate(); err != nil {
		return nil, err
	}

	docs, err := p.backend.Retrieve(ctx, query, opts)
	if err != nil {
		return nil, err
	}

	if validateErr := validateDocuments(docs); validateErr != nil {
		return nil, validateErr
	}

	docs = applyMinSimilarity(docs, opts.MinSimilarity)
	docs = applyTopK(docs, opts.TopK)

	for _, processor := range p.processors {
		docs, err = processor.Process(docs)
		if err != nil {
			return nil, err
		}
		if validateErr := validateDocuments(docs); validateErr != nil {
			return nil, validateErr
		}
	}

	return docs, nil
}

func validateDocuments[TMeta any](docs []Document[TMeta]) error {
	for _, doc := range docs {
		if err := ValidateDocument(doc); err != nil {
			return err
		}
	}
	return nil
}

func applyMinSimilarity[TMeta any](docs []Document[TMeta], minSimilarity float64) []Document[TMeta] {
	if minSimilarity <= 0 || len(docs) == 0 {
		return docs
	}
	out := make([]Document[TMeta], 0, len(docs))
	for _, doc := range docs {
		if doc.Score >= minSimilarity {
			out = append(out, doc)
		}
	}
	return out
}

func applyTopK[TMeta any](docs []Document[TMeta], topK int) []Document[TMeta] {
	if topK <= 0 || len(docs) <= topK {
		return docs
	}
	return docs[:topK]
}
