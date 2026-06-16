// Package testutil provides capability-specific fakes for tests.
package testutil

import (
	"context"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/chunking"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/documents"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/internal/contracttest"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/tensor"
)

// DenseEmbedder is a fake dense embedder.
type DenseEmbedder struct {
	Vectors  [][]float32
	Err      error
	Requests [][]string
}

// Embed implements dense.Embedder.
func (e *DenseEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	e.Requests = append(e.Requests, append([]string(nil), texts...))
	if e.Err != nil {
		return nil, e.Err
	}

	out := make([][]float32, len(e.Vectors))
	for i := range e.Vectors {
		out[i] = append([]float32(nil), e.Vectors[i]...)
	}
	return out, nil
}

// RetrievalBackend is an alias for StructRetrievalBackend.
type RetrievalBackend = StructRetrievalBackend

// DocumentStore is an alias for StructDocumentStore.
type DocumentStore = StructDocumentStore

// DenseIndex is a fake dense index.
type DenseIndex struct {
	Records      [][]dense.Record[contracttest.StructMeta]
	Err          error
	FilterSchema filter.Schema
}

// Upsert implements dense.Index.
func (i *DenseIndex) Upsert(_ context.Context, records []dense.Record[contracttest.StructMeta]) error {
	if i.Err != nil {
		return i.Err
	}
	if !i.FilterSchema.IsFinalized() {
		return fmt.Errorf("%w: dense index schema", ragy.ErrInvalidArgument)
	}

	copied := make([]dense.Record[contracttest.StructMeta], len(records))
	for index, record := range records {
		if err := record.Validate(); err != nil {
			return err
		}

		codec := retrieval.NewJSONCodec[contracttest.StructMeta](i.FilterSchema)
		attrs, err := codec.Encode(record.Meta)
		if err != nil {
			return err
		}
		meta, err := codec.Decode(attrs)
		if err != nil {
			return err
		}

		copied[index] = dense.Record[contracttest.StructMeta]{
			ID:      record.ID,
			Content: record.Content,
			Meta:    meta,
			Vector:  append([]float32(nil), record.Vector...),
		}
	}
	i.Records = append(i.Records, copied)
	return nil
}

// Schema returns the configured filter schema used by the fake index.
func (i *DenseIndex) Schema() filter.Schema {
	return i.FilterSchema
}

// TensorIndex is a fake tensor index.
type TensorIndex struct {
	Records      [][]tensor.Record[contracttest.StructMeta]
	Err          error
	FilterSchema filter.Schema
}

// Upsert implements tensor.Index.
func (i *TensorIndex) Upsert(_ context.Context, records []tensor.Record[contracttest.StructMeta]) error {
	if i.Err != nil {
		return i.Err
	}
	if !i.FilterSchema.IsFinalized() {
		return fmt.Errorf("%w: tensor index schema", ragy.ErrInvalidArgument)
	}

	copied := make([]tensor.Record[contracttest.StructMeta], len(records))
	for index, record := range records {
		if err := record.Validate(); err != nil {
			return err
		}

		codec := retrieval.NewJSONCodec[contracttest.StructMeta](i.FilterSchema)
		attrs, err := codec.Encode(record.Meta)
		if err != nil {
			return err
		}
		meta, err := codec.Decode(attrs)
		if err != nil {
			return err
		}

		copied[index] = tensor.Record[contracttest.StructMeta]{
			ID:      record.ID,
			Content: record.Content,
			Meta:    meta,
			Tensor:  cloneTensor(record.Tensor),
		}
	}
	i.Records = append(i.Records, copied)
	return nil
}

// Schema returns the configured filter schema used by the fake index.
func (i *TensorIndex) Schema() filter.Schema {
	return i.FilterSchema
}

func deleteByFilter[TMeta any](
	docs []retrieval.Document[TMeta],
	cond filter.Condition,
	schema filter.Schema,
	match func(retrieval.Document[TMeta], filter.Condition) (bool, error),
	clone func(retrieval.Document[TMeta]) retrieval.Document[TMeta],
	storeErr error,
	assign func([]retrieval.Document[TMeta]),
) (documents.DeleteResult, error) {
	if storeErr != nil {
		return documents.DeleteResult{}, storeErr
	}

	expr := cond.IR()
	if filter.IsEmpty(expr) {
		return documents.DeleteResult{}, fmt.Errorf("%w: delete filter", ragy.ErrInvalidArgument)
	}
	if err := schema.ValidateSchemaIR(expr); err != nil {
		return documents.DeleteResult{}, err
	}

	deleted := 0
	kept := make([]retrieval.Document[TMeta], 0, len(docs))
	for _, doc := range docs {
		matched, err := match(doc, cond)
		if err != nil {
			return documents.DeleteResult{}, err
		}
		if matched {
			deleted++
			continue
		}
		kept = append(kept, clone(doc))
	}

	assign(kept)
	return documents.DeleteResult{Deleted: deleted}, nil
}

// GraphStore is a memory-backed graph.Store fake.
type GraphStore struct {
	Snapshot    graph.Snapshot[contracttest.StructMeta]
	GraphSchema graph.Schema
	Err         error
	Requests    []graph.TraversalRequest
}

// Traverse implements graph.Store.
func (s *GraphStore) Traverse(
	_ context.Context,
	req graph.TraversalRequest,
) (graph.Snapshot[contracttest.StructMeta], error) {
	s.Requests = append(s.Requests, cloneTraversalRequest(req))
	if s.Err != nil {
		return graph.Snapshot[contracttest.StructMeta]{}, s.Err
	}
	if err := s.GraphSchema.ValidateTraversal(req); err != nil {
		return graph.Snapshot[contracttest.StructMeta]{}, err
	}
	snapshot, err := graph.NormalizeSnapshot(s.GraphSchema, s.Snapshot)
	if err != nil {
		return graph.Snapshot[contracttest.StructMeta]{}, err
	}

	out, err := traverseSnapshot(snapshot, req, s.GraphSchema.NodeAttributes, s.GraphSchema.EdgeAttributes)
	if err != nil {
		return graph.Snapshot[contracttest.StructMeta]{}, err
	}

	return graph.NormalizeSnapshot(s.GraphSchema, out)
}

// Upsert implements graph.Store.
func (s *GraphStore) Upsert(_ context.Context, snapshot graph.Snapshot[contracttest.StructMeta]) error {
	if s.Err != nil {
		return s.Err
	}
	normalized, err := graph.NormalizeSnapshot(s.GraphSchema, snapshot)
	if err != nil {
		return err
	}

	s.Snapshot = mergeSnapshot(s.Snapshot, normalized)
	return nil
}

// Schema returns the configured graph schema used by the fake store.
func (s *GraphStore) Schema() graph.Schema {
	return s.GraphSchema
}

// ContextGenerator is a fake chunk context generator.
type ContextGenerator struct {
	Value string
	Err   error
}

// Context implements chunking.ContextGenerator.
func (g *ContextGenerator) Context(
	_ context.Context,
	_ retrieval.Document[contracttest.StructMeta],
	_ chunking.Chunk[contracttest.StructMeta],
) (string, error) {
	return g.Value, g.Err
}

// GraphProvider is a fake graph extraction provider.
type GraphProvider struct {
	Snapshot graph.Snapshot[contracttest.StructMeta]
	Err      error
}

// Extract extracts a graph snapshot from chunks.
func (p *GraphProvider) Extract(
	_ context.Context,
	_ []chunking.Chunk[contracttest.StructMeta],
) (graph.Snapshot[contracttest.StructMeta], error) {
	return cloneSnapshot(p.Snapshot), p.Err
}

func cloneTensor(in tensor.Tensor) tensor.Tensor {
	if len(in) == 0 {
		return nil
	}

	out := make(tensor.Tensor, len(in))
	for i := range in {
		out[i] = append([]float32(nil), in[i]...)
	}

	return out
}

func cloneNode(in graph.Node[contracttest.StructMeta]) graph.Node[contracttest.StructMeta] {
	return graph.Node[contracttest.StructMeta]{
		ID:      in.ID,
		Labels:  append([]string(nil), in.Labels...),
		Content: in.Content,
		Meta:    in.Meta,
	}
}

func cloneEdge(in graph.Edge[contracttest.StructMeta]) graph.Edge[contracttest.StructMeta] {
	return graph.Edge[contracttest.StructMeta]{
		ID:       in.ID,
		SourceID: in.SourceID,
		TargetID: in.TargetID,
		Type:     in.Type,
		Meta:     in.Meta,
	}
}

func cloneSnapshot(in graph.Snapshot[contracttest.StructMeta]) graph.Snapshot[contracttest.StructMeta] {
	out := graph.Snapshot[contracttest.StructMeta]{
		Nodes: make([]graph.Node[contracttest.StructMeta], len(in.Nodes)),
		Edges: make([]graph.Edge[contracttest.StructMeta], len(in.Edges)),
	}
	for i := range in.Nodes {
		out.Nodes[i] = cloneNode(in.Nodes[i])
	}
	for i := range in.Edges {
		out.Edges[i] = cloneEdge(in.Edges[i])
	}
	return out
}

func mergeSnapshot(base, incoming graph.Snapshot[contracttest.StructMeta]) graph.Snapshot[contracttest.StructMeta] {
	out := cloneSnapshot(base)

	nodeIndex := make(map[string]int, len(out.Nodes))
	for i, node := range out.Nodes {
		nodeIndex[node.ID] = i
	}
	for _, node := range incoming.Nodes {
		cloned := cloneNode(node)
		if index, ok := nodeIndex[cloned.ID]; ok {
			out.Nodes[index] = cloned
			continue
		}
		nodeIndex[cloned.ID] = len(out.Nodes)
		out.Nodes = append(out.Nodes, cloned)
	}

	edgeIndex := make(map[string]int, len(out.Edges))
	for i, edge := range out.Edges {
		edgeIndex[edge.ID] = i
	}
	for _, edge := range incoming.Edges {
		cloned := cloneEdge(edge)
		if index, ok := edgeIndex[cloned.ID]; ok {
			out.Edges[index] = cloned
			continue
		}
		edgeIndex[cloned.ID] = len(out.Edges)
		out.Edges = append(out.Edges, cloned)
	}

	return out
}

func cloneTraversalRequest(in graph.TraversalRequest) graph.TraversalRequest {
	var page *ragy.Page
	if in.Page != nil {
		page = &ragy.Page{Limit: in.Page.Limit, Offset: in.Page.Offset}
	}

	return graph.TraversalRequest{
		Seeds:      append([]string(nil), in.Seeds...),
		Direction:  in.Direction,
		Depth:      in.Depth,
		NodeFilter: in.NodeFilter,
		EdgeFilter: in.EdgeFilter,
		Page:       page,
	}
}

func traverseSnapshot(
	snapshot graph.Snapshot[contracttest.StructMeta],
	req graph.TraversalRequest,
	nodeSchema filter.Schema,
	edgeSchema filter.Schema,
) (graph.Snapshot[contracttest.StructMeta], error) {
	nodesByID := indexNodes(snapshot.Nodes)
	visitedNodes, frontier := seedFrontier(nodesByID, req.Seeds)
	visitedEdges, err := expandTraversal(snapshot.Edges, nodesByID, visitedNodes, frontier, req, edgeSchema)
	if err != nil {
		return graph.Snapshot[contracttest.StructMeta]{}, err
	}

	nodes, allowedNodes, err := projectNodes(snapshot.Nodes, visitedNodes, req.NodeFilter, nodeSchema)
	if err != nil {
		return graph.Snapshot[contracttest.StructMeta]{}, err
	}
	if req.Page != nil {
		nodes, allowedNodes = pageNodes(nodes, req.Page)
	}

	edges, err := projectEdges(snapshot.Edges, visitedEdges, allowedNodes, req.EdgeFilter, edgeSchema)
	if err != nil {
		return graph.Snapshot[contracttest.StructMeta]{}, err
	}

	return graph.Snapshot[contracttest.StructMeta]{Nodes: nodes, Edges: edges}, nil
}

func traversesEdge(edge graph.Edge[contracttest.StructMeta], current string, direction graph.Direction) (bool, string) {
	switch direction {
	case graph.DirectionOutbound:
		return edge.SourceID == current, edge.TargetID
	case graph.DirectionInbound:
		return edge.TargetID == current, edge.SourceID
	case graph.DirectionUndirected:
		switch {
		case edge.SourceID == current:
			return true, edge.TargetID
		case edge.TargetID == current:
			return true, edge.SourceID
		default:
			return false, ""
		}
	default:
		return false, ""
	}
}

func indexNodes(nodes []graph.Node[contracttest.StructMeta]) map[string]graph.Node[contracttest.StructMeta] {
	out := make(map[string]graph.Node[contracttest.StructMeta], len(nodes))
	for _, node := range nodes {
		out[node.ID] = node
	}
	return out
}

func seedFrontier(
	nodesByID map[string]graph.Node[contracttest.StructMeta],
	seeds []string,
) (map[string]struct{}, []string) {
	visited := make(map[string]struct{}, len(seeds))
	frontier := make([]string, 0, len(seeds))
	for _, seed := range seeds {
		if _, ok := nodesByID[seed]; !ok {
			continue
		}
		if _, ok := visited[seed]; ok {
			continue
		}
		visited[seed] = struct{}{}
		frontier = append(frontier, seed)
	}
	return visited, frontier
}

func expandTraversal(
	edges []graph.Edge[contracttest.StructMeta],
	nodesByID map[string]graph.Node[contracttest.StructMeta],
	visitedNodes map[string]struct{},
	frontier []string,
	req graph.TraversalRequest,
	edgeSchema filter.Schema,
) (map[string]struct{}, error) {
	visitedEdges := make(map[string]struct{})
	edgeCodec := retrieval.NewJSONCodec[contracttest.StructMeta](edgeSchema)
	for level := 0; level < req.Depth && len(frontier) > 0; level++ {
		var err error
		frontier, err = expandLevel(edges, nodesByID, visitedNodes, visitedEdges, frontier, req, edgeCodec)
		if err != nil {
			return nil, err
		}
	}
	return visitedEdges, nil
}

func expandLevel(
	edges []graph.Edge[contracttest.StructMeta],
	nodesByID map[string]graph.Node[contracttest.StructMeta],
	visitedNodes map[string]struct{},
	visitedEdges map[string]struct{},
	frontier []string,
	req graph.TraversalRequest,
	edgeCodec retrieval.MetadataCodec[contracttest.StructMeta],
) ([]string, error) {
	nextFrontier := make([]string, 0)
	nextSeen := make(map[string]struct{})
	for _, current := range frontier {
		for _, edge := range edges {
			traverses, neighbor := traversesEdge(edge, current, req.Direction)
			if !traverses {
				continue
			}

			matched, err := matchGraphMeta(edge.Meta, req.EdgeFilter, edgeCodec)
			if err != nil {
				return nil, err
			}
			if !matched {
				continue
			}
			if _, ok := nodesByID[neighbor]; !ok {
				continue
			}

			visitedEdges[edge.ID] = struct{}{}
			visitedNodes[neighbor] = struct{}{}
			if _, ok := nextSeen[neighbor]; ok {
				continue
			}
			nextSeen[neighbor] = struct{}{}
			nextFrontier = append(nextFrontier, neighbor)
		}
	}
	return nextFrontier, nil
}

func projectNodes(
	nodes []graph.Node[contracttest.StructMeta],
	visited map[string]struct{},
	cond filter.Condition,
	schema filter.Schema,
) ([]graph.Node[contracttest.StructMeta], map[string]struct{}, error) {
	out := make([]graph.Node[contracttest.StructMeta], 0, len(visited))
	allowed := make(map[string]struct{}, len(visited))
	codec := retrieval.NewJSONCodec[contracttest.StructMeta](schema)
	for _, node := range nodes {
		if _, ok := visited[node.ID]; !ok {
			continue
		}

		matched, err := matchGraphMeta(node.Meta, cond, codec)
		if err != nil {
			return nil, nil, err
		}
		if !matched {
			continue
		}

		cloned := cloneNode(node)
		out = append(out, cloned)
		allowed[cloned.ID] = struct{}{}
	}
	return out, allowed, nil
}

func pageNodes(
	nodes []graph.Node[contracttest.StructMeta],
	page *ragy.Page,
) ([]graph.Node[contracttest.StructMeta], map[string]struct{}) {
	start := minInt(page.Offset, len(nodes))
	end := minInt(start+page.Limit, len(nodes))
	paged := append([]graph.Node[contracttest.StructMeta](nil), nodes[start:end]...)
	allowed := make(map[string]struct{}, len(paged))
	for _, node := range paged {
		allowed[node.ID] = struct{}{}
	}
	return paged, allowed
}

func projectEdges(
	edges []graph.Edge[contracttest.StructMeta],
	visited map[string]struct{},
	allowedNodes map[string]struct{},
	cond filter.Condition,
	schema filter.Schema,
) ([]graph.Edge[contracttest.StructMeta], error) {
	out := make([]graph.Edge[contracttest.StructMeta], 0, len(visited))
	codec := retrieval.NewJSONCodec[contracttest.StructMeta](schema)
	for _, edge := range edges {
		if _, ok := visited[edge.ID]; !ok {
			continue
		}
		if _, ok := allowedNodes[edge.SourceID]; !ok {
			continue
		}
		if _, ok := allowedNodes[edge.TargetID]; !ok {
			continue
		}

		matched, err := matchGraphMeta(edge.Meta, cond, codec)
		if err != nil {
			return nil, err
		}
		if !matched {
			continue
		}
		out = append(out, cloneEdge(edge))
	}
	return out, nil
}

func matchGraphMeta(
	meta contracttest.StructMeta,
	cond filter.Condition,
	codec retrieval.MetadataCodec[contracttest.StructMeta],
) (bool, error) {
	return retrieval.MatchDocument(codec, retrieval.Document[contracttest.StructMeta]{Meta: meta}, cond)
}

func minInt(left, right int) int {
	if left < right {
		return left
	}
	return right
}

var (
	_ dense.Embedder                                     = (*DenseEmbedder)(nil)
	_ dense.Index[contracttest.StructMeta]               = (*DenseIndex)(nil)
	_ retrieval.Backend[contracttest.StructMeta]         = (*StructRetrievalBackend)(nil)
	_ tensor.Index[contracttest.StructMeta]              = (*TensorIndex)(nil)
	_ documents.Store[contracttest.StructMeta]           = (*StructDocumentStore)(nil)
	_ graph.Store[contracttest.StructMeta]               = (*GraphStore)(nil)
	_ chunking.ContextGenerator[contracttest.StructMeta] = (*ContextGenerator)(nil)
)
