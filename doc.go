// Package ragy provides a stateless Go engine for semantic memory (RAG pipeline).
//
// Design principles:
//   - No LLM dependency: logic that needs LLM (e.g. query rewriting, entity extraction for GraphRAG)
//     is injected via interfaces/callbacks on the application side.
//   - Observability: use module github.com/skosovsky/ragy/adapters/observability/otel to wrap Retriever/VectorStore with OpenTelemetry (not in core).
//   - Streaming: Retriever and VectorStore expose Stream(ctx, req) [iter.Seq2][Document, error] for TTFT-friendly consumption.
//
// Core types (Document, Node, Edge, SearchRequest, ParsedQuery) and interfaces
// (DenseEmbedder, TensorEmbedder, VectorStore, GraphStore, Retriever, Reranker, QueryTransformer,
// Contextualizer, HierarchyRetriever, ChunkGraphProvider) are in this package. Implementations live in sub-packages:
//
//   - filter: AST-based type-safe filters (Eq, Neq, And, Or, Not, etc.)
//   - splitters: ingestion pipeline (RecursiveSplitter, MarkdownSplitter, SemanticSplitter, GraphExtractor, ContextualSplitter)
//   - retrievers: retrieval pipeline (BaseVectorRetriever, ColBERTRetriever, GraphRetriever, RouterRetriever, SelfQueryRetriever, EnsembleRetriever, MultiQueryRetriever, HyDERetriever)
//   - rerankers: RRF and CrossEncoder rerankers
//   - cache: semantic cache (SemanticCache, VectorCache)
//   - testutil: in-memory stores and mock embedders for tests
//
// Official adapters live in separate modules under adapters/ (e.g. adapters/openai, adapters/gemini, adapters/pgvector).
// For local development, use go.work in the repo root so adapters resolve the local ragy module.
//
// Adapter contract (for VectorStore/GraphStore implementations):
//   - Search: vectors are read from req.DenseVector (dense search) or req.TensorVector (ColBERT), not a separate embedding parameter.
//   - For dense search, documents must have Metadata[EmbeddingMetadataKey] = []float32 so the store can score by similarity; ragy/cache and retriever pipelines rely on this.
//   - Return Document.Confidence in [0,1] when possible (normalized relevance).
//   - Upsert(ctx, docs) accepts slices of any size; adapters MUST micro-batch internally (e.g. 500 per call).
//   - Search with Offset: ANN indexes often do not support OFFSET; request Limit+Offset and slice in Go.
//   - Filter: use req.Filter (filter.Expr); traverse via type switch to build native queries (SQL WHERE, Qdrant filter JSON, etc.).
package ragy
