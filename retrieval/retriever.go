package retrieval

import "context"

// Backend executes retrieval against a concrete store without post-processing.
type Backend[TMeta any] interface {
	Retrieve(ctx context.Context, query string, opts RetrieveOptions) ([]Document[TMeta], error)
}

// Retriever returns post-processed, typed documents for a query.
type Retriever[TMeta any] interface {
	Retrieve(ctx context.Context, query string, opts RetrieveOptions) ([]Document[TMeta], error)
}

// PostProcessor transforms a ranked document batch.
type PostProcessor[TMeta any] interface {
	Process(docs []Document[TMeta]) ([]Document[TMeta], error)
}
