package ragy

import "github.com/skosovsky/ragy/filter"

// Document is the basic unit of knowledge: a chunk of text with metadata and optional score.
type Document struct {
	ID       string
	Content  string
	Metadata map[string]any // e.g. TenantID, ParentID, Author, Source, CreatedAt
	Score    float32        // Final relevance score (e.g. after reranking)
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

// RetrievalResult is the enriched response for observability and eval pipelines.
// EvalData holds intermediate data: raw scores, multi-query results, etc.
type RetrievalResult struct {
	Documents []Document
	EvalData  map[string]any
}

// SearchRequest carries the query, optional pre-computed vectors, pagination, and filter.
// DenseVector is filled by BaseVectorRetriever/HyDERetriever; TensorVector by ColBERTRetriever.
// Adapters read vectors from here (type-safe). Offset is for pagination; adapters for ANN
// typically request Limit+Offset and slice in Go.
type SearchRequest struct {
	Query        string
	DenseVector  []float32   // Pre-computed by BaseVectorRetriever / HyDERetriever
	TensorVector [][]float32 // Pre-computed by ColBERTRetriever (per-token vectors for one query)
	Limit        int
	Offset       int         // Pagination for aggregating retrievers
	Filter       filter.Expr // AST-based filter; nil means no filter
}
