package ragy

import "github.com/skosovsky/ragy/filter"

// Media represents a binary attachment (e.g. image, PDF) for multimodal documents and search.
type Media struct {
	MimeType string // e.g. "image/jpeg", "application/pdf"
	Data     []byte
}

// EmbeddingMetadataKey is the conventional Document.Metadata key for the dense vector ([]float32).
// VectorStore adapters that support dense search and ragy/cache expect this key for similarity.
const EmbeddingMetadataKey = "embedding"

// ParentDocumentIDKey is optional metadata on child chunks: business ID of the parent document for HierarchyRetriever.FetchParents.
// Value must be a string ID that exists as Document.ID for the parent row/point.
const ParentDocumentIDKey = "_parent_doc_id"

// Document is the basic unit of knowledge: a chunk of text with metadata and optional scores.
type Document struct {
	ID         string
	Content    string
	Media      []Media        // For multimodal documents (images, etc.)
	Metadata   map[string]any // e.g. TenantID, ParentID, Author, Source, CreatedAt; use EmbeddingMetadataKey for dense vector
	Score      float32        // Raw relevance score (e.g. after reranking or adapter-specific)
	Confidence float64        // Normalized relevance in [0,1]; adapters must set when returning from Search
}

// Node is a primitive for GraphRAG: a vertex in the knowledge graph.
type Node struct {
	ID         string
	Label      string
	Properties map[string]any
}

// Edge is a primitive for GraphRAG: a directed edge between two nodes.
type Edge struct {
	SourceID   string
	TargetID   string
	Relation   string
	Properties map[string]any
}

// SearchRequest carries the query, optional pre-computed vectors, pagination, and filter.
// DenseVector is filled by BaseVectorRetriever/HyDERetriever; TensorVector by ColBERTRetriever.
// SparseVector is for hybrid sparse retrieval when supported by the store.
// ParsedQuery is set by the application for SelfQueryRetriever (no LLM inside ragy).
// GraphSeedEntityIDs seeds GraphRetriever when graph traversal starts from known entity IDs.
// Adapters read vectors from here (type-safe). Offset is for pagination; adapters for ANN
// typically request Limit+Offset and slice in Go.
type SearchRequest struct {
	Query              string
	Media              []Media     // Image(s) for similarity search (e.g. photo of symptoms)
	DenseVector        []float32   // Pre-computed by BaseVectorRetriever / HyDERetriever
	TensorVector       [][]float32 // Pre-computed by ColBERTRetriever (per-token vectors for one query)
	SparseVector       map[uint32]float32
	Limit              int
	Offset             int         // Pagination for aggregating retrievers
	Filter             filter.Expr // AST-based filter; nil means no filter
	ParsedQuery        *ParsedQuery
	GraphSeedEntityIDs []string
}

// ParsedQuery is the result of parsing a natural-language query outside ragy (e.g. via LLM in the app).
type ParsedQuery struct {
	SemanticQuery string      // Cleaned query for vector search
	Filter        filter.Expr // AST filter tree extracted from the query
	Limit         int         // Extracted limit (e.g. "find 5 best...")
}
