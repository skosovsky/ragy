package retrieval

import "context"

// Backend executes retrieval against a concrete store without post-processing.
// Prefer retrieval.Pipeline for orchestration; this path is for direct backend access.
type Backend[TMeta any] interface {
	Retrieve(ctx context.Context, query string, opts RetrieveOptions) (ResultSet[TMeta], error)
}

// PostProcessor transforms a ranked result set.
type PostProcessor[TMeta any] interface {
	Process(rs ResultSet[TMeta]) (ResultSet[TMeta], error)
}
