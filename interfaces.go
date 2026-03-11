package ragy

import (
	"context"

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
// Search reads DenseVector or TensorVector from SearchRequest (type-safe; no embedding any).
// Upsert accepts slices of any size; adapter implementations MUST micro-batch internally
// (e.g. 500 docs per network call) to avoid gRPC/HTTP payload limits and timeouts.
type VectorStore interface {
	Search(ctx context.Context, req SearchRequest) ([]Document, error)
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
	Retrieve(ctx context.Context, req SearchRequest) (RetrievalResult, error)
}

// QueryTransformer prepares the query (e.g. multi-query expansion).
// Returns one or more transformed query strings.
type QueryTransformer interface {
	Transform(ctx context.Context, query string) ([]string, error)
}

// QueryParser translates a natural-language query into a structured ParsedQuery
// (semantic part + filter.Expr). Typically implemented via LLM on the application side.
type QueryParser interface {
	Parse(ctx context.Context, naturalQuery string) (ParsedQuery, error)
}

// Reranker re-scores and truncates a document list (e.g. RRF merge or cross-encoder).
type Reranker interface {
	Rerank(ctx context.Context, query string, docs []Document, topK int) ([]Document, error)
}

// EntityExtractor is a callback that extracts nodes and edges from text (typically via LLM).
// Used by GraphExtractor and optionally by GraphRetriever.
type EntityExtractor func(ctx context.Context, text string) ([]Node, []Edge, error)

// Contextualizer generates enriching context for a chunk based on the full document.
// The returned string (1-2 sentences) is prepended to the chunk content.
type Contextualizer interface {
	GenerateContext(ctx context.Context, fullContent string, chunkContent string) (string, error)
}
