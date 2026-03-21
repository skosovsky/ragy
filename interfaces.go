package ragy

import (
	"context"
	"iter"

	"github.com/skosovsky/ragy/filter"
)

// DenseEmbedder produces one dense vector per input text.
// Used by BaseVectorRetriever and HyDERetriever.
type DenseEmbedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// TensorEmbedder produces per-token tensor representations (e.g. for ColBERT late interaction).
// Returns [][][]float32: for each text, a slice of token vectors.
type TensorEmbedder interface {
	EmbedTensors(ctx context.Context, texts []string) ([][][]float32, error)
}

// MultimodalEmbedder produces one dense vector per (text, media) pair.
// len(texts) must equal len(media); empty media[i] means text-only embedding for that index.
type MultimodalEmbedder interface {
	EmbedMultimodal(ctx context.Context, texts []string, media [][]Media) ([][]float32, error)
}

// VectorStore performs similarity search and batch upsert.
// Search reads DenseVector, TensorVector, or SparseVector from SearchRequest (type-safe).
// Upsert accepts slices of any size; adapter implementations MUST micro-batch internally
// (e.g. 500 docs per network call) to avoid gRPC/HTTP payload limits and timeouts.
type VectorStore interface {
	Search(ctx context.Context, req SearchRequest) ([]Document, error)
	Stream(ctx context.Context, req SearchRequest) iter.Seq2[Document, error]
	Upsert(ctx context.Context, docs []Document) error
	DeleteByFilter(ctx context.Context, f filter.Expr) error
}

// GraphStore provides graph traversal and upsert for GraphRAG.
type GraphStore interface {
	SearchGraph(ctx context.Context, entities []string, depth int, req SearchRequest) ([]Node, []Edge, error)
	UpsertGraph(ctx context.Context, nodes []Node, edges []Edge) error
}

// Retriever is the unified contract for all retrieval strategies.
type Retriever interface {
	Retrieve(ctx context.Context, req SearchRequest) ([]Document, error)
	Stream(ctx context.Context, req SearchRequest) iter.Seq2[Document, error]
}

// HierarchyRetriever is optional: fetch parent documents for child chunk IDs (parent-child chunking).
type HierarchyRetriever interface {
	FetchParents(ctx context.Context, childIDs []string) ([]Document, error)
}

// QueryTransformer prepares the query (e.g. multi-query expansion).
// Returns one or more transformed query strings.
type QueryTransformer interface {
	Transform(ctx context.Context, query string) ([]string, error)
}

// Reranker re-scores and truncates a document list (e.g. RRF merge or cross-encoder).
type Reranker interface {
	Rerank(ctx context.Context, query string, docs []Document, topK int) ([]Document, error)
}

// EntityExtractor is a legacy callback that extracts nodes and edges from raw text (application / LLM side).
// Prefer ChunkGraphProvider for GraphExtractor: core does not invoke text-based extraction.
type EntityExtractor func(ctx context.Context, text string) ([]Node, []Edge, error)

// ChunkGraphProvider supplies prepared graph nodes and edges for a chunk after splitting.
// Implementations may read chunk.Metadata or call out-of-band pipelines; ragy core does not call LLMs.
type ChunkGraphProvider func(ctx context.Context, chunk Document) ([]Node, []Edge, error)

// Contextualizer generates enriching context for a chunk based on the full document.
// The returned string (1-2 sentences) is prepended to the chunk content.
type Contextualizer interface {
	GenerateContext(ctx context.Context, fullContent string, chunkContent string) (string, error)
}
