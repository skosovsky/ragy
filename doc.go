// Package ragy provides a stateless Go engine for semantic memory (RAG pipeline).
//
// Design principles:
//   - No LLM dependency: logic that needs LLM (e.g. query rewriting, entity extraction for GraphRAG)
//     is injected via interfaces/callbacks.
//   - Observability first: use sub-package obs/ to wrap retrievers and stores with OpenTelemetry.
//   - Eval-ready: RetrievalResult.EvalData holds intermediate state for RAGAs/TrueLens pipelines.
//
// Core types (Document, Node, Edge, SearchRequest, RetrievalResult) and interfaces
// (DenseEmbedder, TensorEmbedder, VectorStore, GraphStore, Retriever, Reranker, QueryTransformer)
// are in this package. Implementations live in sub-packages:
//
//   - filter: AST-based type-safe filters (Eq, Neq, And, Or, Not, etc.)
//   - splitters: ingestion pipeline (RecursiveSplitter, MarkdownSplitter, SemanticSplitter, GraphExtractor)
//   - retrievers: retrieval pipeline (BaseVectorRetriever, ColBERTRetriever, GraphRetriever, RouterRetriever, EnsembleRetriever, MultiQueryRetriever, HyDERetriever)
//   - rerankers: RRF and CrossEncoder rerankers
//   - obs: OpenTelemetry decorators for tracing
//   - testutil: in-memory stores and mock embedders for tests
//
// Adapter contract (for VectorStore/GraphStore implementations):
//   - Search: vectors are read from req.DenseVector (dense search) or req.TensorVector (ColBERT), not a separate embedding parameter.
//   - Upsert(ctx, docs) accepts slices of any size; adapters MUST micro-batch internally (e.g. 500 per call).
//   - Search with Offset: ANN indexes often do not support OFFSET; request Limit+Offset and slice in Go.
//   - Filter: use req.Filter (filter.Expr); traverse via type switch to build native queries (SQL WHERE, Qdrant filter JSON, etc.).
package ragy
