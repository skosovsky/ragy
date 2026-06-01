// Package testutil provides capability-specific fakes for tests.
package testutil

import (
	"context"
	"fmt"
	"strings"

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

// RetrievalBackend is a fake retrieval.Backend.
type RetrievalBackend struct {
	Docs           []retrieval.Document[contracttest.Meta]
	Err            error
	Requests       []retrieval.RetrieveOptions
	FilterSchema   filter.Schema
	VectorRequired bool
}

// Retrieve implements retrieval.Backend.
func (b *RetrievalBackend) Retrieve(
	_ context.Context,
	query string,
	opts retrieval.RetrieveOptions,
) ([]retrieval.Document[contracttest.Meta], error) {
	b.Requests = append(b.Requests, opts)
	if b.Err != nil {
		return nil, b.Err
	}
	if err := opts.Validate(); err != nil {
		return nil, err
	}
	if b.VectorRequired {
		if len(opts.Vector) == 0 {
			return nil, fmt.Errorf("%w: retrieve vector", ragy.ErrEmptyVector)
		}
	} else if strings.TrimSpace(query) == "" {
		return nil, fmt.Errorf("%w: retrieve query", ragy.ErrEmptyText)
	}
	if err := b.Schema().ValidateSchemaIR(opts.Filters.IR()); err != nil {
		return nil, err
	}

	return validateRetrievalDocuments(b.Docs)
}

// Schema returns the configured filter schema used by the fake backend.
func (b *RetrievalBackend) Schema() filter.Schema {
	return b.FilterSchema
}

// DenseIndex is a fake dense index.
type DenseIndex struct {
	Records      [][]dense.Record[contracttest.Meta]
	Err          error
	FilterSchema filter.Schema
}

// Upsert implements dense.Index.
func (i *DenseIndex) Upsert(_ context.Context, records []dense.Record[contracttest.Meta]) error {
	if i.Err != nil {
		return i.Err
	}
	if !i.FilterSchema.IsFinalized() {
		return fmt.Errorf("%w: dense index schema", ragy.ErrInvalidArgument)
	}

	copied := make([]dense.Record[contracttest.Meta], len(records))
	for index, record := range records {
		if err := record.Validate(); err != nil {
			return err
		}

		attrs, err := i.FilterSchema.NormalizeAttributes(filter.RawAttributes(record.Meta))
		if err != nil {
			return err
		}

		copied[index] = dense.Record[contracttest.Meta]{
			ID:      record.ID,
			Content: record.Content,
			Meta:    contracttest.Meta(filter.CloneRawAttributes(attrs)),
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
	Records      [][]tensor.Record[contracttest.Meta]
	Err          error
	FilterSchema filter.Schema
}

// Upsert implements tensor.Index.
func (i *TensorIndex) Upsert(_ context.Context, records []tensor.Record[contracttest.Meta]) error {
	if i.Err != nil {
		return i.Err
	}
	if !i.FilterSchema.IsFinalized() {
		return fmt.Errorf("%w: tensor index schema", ragy.ErrInvalidArgument)
	}

	copied := make([]tensor.Record[contracttest.Meta], len(records))
	for index, record := range records {
		if err := record.Validate(); err != nil {
			return err
		}

		attrs, err := i.FilterSchema.NormalizeAttributes(filter.RawAttributes(record.Meta))
		if err != nil {
			return err
		}

		copied[index] = tensor.Record[contracttest.Meta]{
			ID:      record.ID,
			Content: record.Content,
			Meta:    contracttest.Meta(filter.CloneRawAttributes(attrs)),
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

// DocumentStore is a memory-backed documents.Store fake.
type DocumentStore struct {
	Docs         []retrieval.Document[contracttest.Meta]
	Err          error
	FindCalls    [][]string
	FilterSchema filter.Schema
}

// FindByIDs implements documents.Store.
func (s *DocumentStore) FindByIDs(_ context.Context, ids []string) ([]retrieval.Document[contracttest.Meta], error) {
	s.FindCalls = append(s.FindCalls, append([]string(nil), ids...))
	if s.Err != nil {
		return nil, s.Err
	}

	if len(ids) == 0 {
		return nil, nil
	}

	byID := make(map[string]retrieval.Document[contracttest.Meta], len(s.Docs))
	for _, doc := range s.Docs {
		byID[doc.ID] = cloneRetrievalDocument(doc)
	}

	out := make([]retrieval.Document[contracttest.Meta], 0, len(ids))
	for _, id := range ids {
		doc, ok := byID[id]
		if !ok {
			continue
		}
		out = append(out, doc)
	}

	if len(out) == 0 {
		return nil, nil
	}

	return validateRetrievalDocuments(out)
}

// DeleteByIDs implements documents.Store.
func (s *DocumentStore) DeleteByIDs(_ context.Context, ids []string) (documents.DeleteResult, error) {
	if s.Err != nil {
		return documents.DeleteResult{}, s.Err
	}

	if len(ids) == 0 {
		return documents.DeleteResult{}, nil
	}

	remove := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		remove[id] = struct{}{}
	}

	deleted := 0
	kept := make([]retrieval.Document[contracttest.Meta], 0, len(s.Docs))
	for _, doc := range s.Docs {
		if _, ok := remove[doc.ID]; ok {
			deleted++
			continue
		}
		kept = append(kept, cloneRetrievalDocument(doc))
	}

	s.Docs = kept
	return documents.DeleteResult{Deleted: deleted}, nil
}

// DeleteByFilter implements documents.Store.
func (s *DocumentStore) DeleteByFilter(_ context.Context, cond filter.Condition) (documents.DeleteResult, error) {
	return deleteByFilter(
		s.Docs,
		cond,
		s.Schema(),
		matchDocument,
		cloneRetrievalDocument,
		s.Err,
		func(docs []retrieval.Document[contracttest.Meta]) {
			s.Docs = docs
		},
	)
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

// Schema returns the configured filter schema used by the fake store.
func (s *DocumentStore) Schema() filter.Schema {
	return s.FilterSchema
}

// GraphStore is a memory-backed graph.Store fake.
type GraphStore struct {
	Snapshot    graph.Snapshot[contracttest.Meta]
	GraphSchema graph.Schema
	Err         error
	Requests    []graph.TraversalRequest
}

// Traverse implements graph.Store.
func (s *GraphStore) Traverse(
	_ context.Context,
	req graph.TraversalRequest,
) (graph.Snapshot[contracttest.Meta], error) {
	s.Requests = append(s.Requests, cloneTraversalRequest(req))
	if s.Err != nil {
		return graph.Snapshot[contracttest.Meta]{}, s.Err
	}
	if err := s.GraphSchema.ValidateTraversal(req); err != nil {
		return graph.Snapshot[contracttest.Meta]{}, err
	}
	snapshot, err := graph.NormalizeSnapshot(s.GraphSchema, s.Snapshot)
	if err != nil {
		return graph.Snapshot[contracttest.Meta]{}, err
	}

	out, err := traverseSnapshot(snapshot, req)
	if err != nil {
		return graph.Snapshot[contracttest.Meta]{}, err
	}

	return graph.NormalizeSnapshot(s.GraphSchema, out)
}

// Upsert implements graph.Store.
func (s *GraphStore) Upsert(_ context.Context, snapshot graph.Snapshot[contracttest.Meta]) error {
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
	_ retrieval.Document[contracttest.Meta],
	_ chunking.Chunk[contracttest.Meta],
) (string, error) {
	return g.Value, g.Err
}

// GraphProvider is a fake graph extraction provider.
type GraphProvider struct {
	Snapshot graph.Snapshot[contracttest.Meta]
	Err      error
}

// Extract extracts a graph snapshot from chunks.
func (p *GraphProvider) Extract(
	_ context.Context,
	_ []chunking.Chunk[contracttest.Meta],
) (graph.Snapshot[contracttest.Meta], error) {
	return cloneSnapshot(p.Snapshot), p.Err
}

func validateRetrievalDocuments(
	in []retrieval.Document[contracttest.Meta],
) ([]retrieval.Document[contracttest.Meta], error) {
	if len(in) == 0 {
		return nil, nil
	}

	out := make([]retrieval.Document[contracttest.Meta], len(in))
	for i, doc := range in {
		if err := retrieval.ValidateDocument(doc); err != nil {
			return nil, err
		}
		out[i] = cloneRetrievalDocument(doc)
	}

	return out, nil
}

func cloneRetrievalDocument(in retrieval.Document[contracttest.Meta]) retrieval.Document[contracttest.Meta] {
	var meta contracttest.Meta
	if len(in.Meta) > 0 {
		meta = contracttest.Meta(filter.CloneRawAttributes(filter.RawAttributes(in.Meta)))
	}

	return retrieval.Document[contracttest.Meta]{
		ID:      in.ID,
		Content: in.Content,
		Score:   in.Score,
		Meta:    meta,
	}
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

func cloneNode(in graph.Node[contracttest.Meta]) graph.Node[contracttest.Meta] {
	var meta contracttest.Meta
	if len(in.Meta) > 0 {
		meta = contracttest.Meta(filter.CloneRawAttributes(filter.RawAttributes(in.Meta)))
	}
	return graph.Node[contracttest.Meta]{
		ID:      in.ID,
		Labels:  append([]string(nil), in.Labels...),
		Content: in.Content,
		Meta:    meta,
	}
}

func cloneEdge(in graph.Edge[contracttest.Meta]) graph.Edge[contracttest.Meta] {
	var meta contracttest.Meta
	if len(in.Meta) > 0 {
		meta = contracttest.Meta(filter.CloneRawAttributes(filter.RawAttributes(in.Meta)))
	}
	return graph.Edge[contracttest.Meta]{
		ID:       in.ID,
		SourceID: in.SourceID,
		TargetID: in.TargetID,
		Type:     in.Type,
		Meta:     meta,
	}
}

func cloneSnapshot(in graph.Snapshot[contracttest.Meta]) graph.Snapshot[contracttest.Meta] {
	out := graph.Snapshot[contracttest.Meta]{
		Nodes: make([]graph.Node[contracttest.Meta], len(in.Nodes)),
		Edges: make([]graph.Edge[contracttest.Meta], len(in.Edges)),
	}
	for i := range in.Nodes {
		out.Nodes[i] = cloneNode(in.Nodes[i])
	}
	for i := range in.Edges {
		out.Edges[i] = cloneEdge(in.Edges[i])
	}
	return out
}

func mergeSnapshot(base, incoming graph.Snapshot[contracttest.Meta]) graph.Snapshot[contracttest.Meta] {
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
	snapshot graph.Snapshot[contracttest.Meta],
	req graph.TraversalRequest,
) (graph.Snapshot[contracttest.Meta], error) {
	nodesByID := indexNodes(snapshot.Nodes)
	visitedNodes, frontier := seedFrontier(nodesByID, req.Seeds)
	visitedEdges, err := expandTraversal(snapshot.Edges, nodesByID, visitedNodes, frontier, req)
	if err != nil {
		return graph.Snapshot[contracttest.Meta]{}, err
	}

	nodes, allowedNodes, err := projectNodes(snapshot.Nodes, visitedNodes, req.NodeFilter)
	if err != nil {
		return graph.Snapshot[contracttest.Meta]{}, err
	}
	if req.Page != nil {
		nodes, allowedNodes = pageNodes(nodes, req.Page)
	}

	edges, err := projectEdges(snapshot.Edges, visitedEdges, allowedNodes, req.EdgeFilter)
	if err != nil {
		return graph.Snapshot[contracttest.Meta]{}, err
	}

	return graph.Snapshot[contracttest.Meta]{Nodes: nodes, Edges: edges}, nil
}

func traversesEdge(edge graph.Edge[contracttest.Meta], current string, direction graph.Direction) (bool, string) {
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

func indexNodes(nodes []graph.Node[contracttest.Meta]) map[string]graph.Node[contracttest.Meta] {
	out := make(map[string]graph.Node[contracttest.Meta], len(nodes))
	for _, node := range nodes {
		out[node.ID] = node
	}
	return out
}

func seedFrontier(nodesByID map[string]graph.Node[contracttest.Meta], seeds []string) (map[string]struct{}, []string) {
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
	edges []graph.Edge[contracttest.Meta],
	nodesByID map[string]graph.Node[contracttest.Meta],
	visitedNodes map[string]struct{},
	frontier []string,
	req graph.TraversalRequest,
) (map[string]struct{}, error) {
	visitedEdges := make(map[string]struct{})
	for level := 0; level < req.Depth && len(frontier) > 0; level++ {
		var err error
		frontier, err = expandLevel(edges, nodesByID, visitedNodes, visitedEdges, frontier, req)
		if err != nil {
			return nil, err
		}
	}
	return visitedEdges, nil
}

func expandLevel(
	edges []graph.Edge[contracttest.Meta],
	nodesByID map[string]graph.Node[contracttest.Meta],
	visitedNodes map[string]struct{},
	visitedEdges map[string]struct{},
	frontier []string,
	req graph.TraversalRequest,
) ([]string, error) {
	nextFrontier := make([]string, 0)
	nextSeen := make(map[string]struct{})
	for _, current := range frontier {
		for _, edge := range edges {
			traverses, neighbor := traversesEdge(edge, current, req.Direction)
			if !traverses {
				continue
			}

			matched, err := matchEdge(edge, req.EdgeFilter)
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
	nodes []graph.Node[contracttest.Meta],
	visited map[string]struct{},
	cond filter.Condition,
) ([]graph.Node[contracttest.Meta], map[string]struct{}, error) {
	out := make([]graph.Node[contracttest.Meta], 0, len(visited))
	allowed := make(map[string]struct{}, len(visited))
	for _, node := range nodes {
		if _, ok := visited[node.ID]; !ok {
			continue
		}

		matched, err := matchNode(node, cond)
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
	nodes []graph.Node[contracttest.Meta],
	page *ragy.Page,
) ([]graph.Node[contracttest.Meta], map[string]struct{}) {
	start := minInt(page.Offset, len(nodes))
	end := minInt(start+page.Limit, len(nodes))
	paged := append([]graph.Node[contracttest.Meta](nil), nodes[start:end]...)
	allowed := make(map[string]struct{}, len(paged))
	for _, node := range paged {
		allowed[node.ID] = struct{}{}
	}
	return paged, allowed
}

func projectEdges(
	edges []graph.Edge[contracttest.Meta],
	visited map[string]struct{},
	allowedNodes map[string]struct{},
	cond filter.Condition,
) ([]graph.Edge[contracttest.Meta], error) {
	out := make([]graph.Edge[contracttest.Meta], 0, len(visited))
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

		matched, err := matchEdge(edge, cond)
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

func matchDocument(doc retrieval.Document[contracttest.Meta], cond filter.Condition) (bool, error) {
	return matchFilter(cond.IR(), func(field string) (any, bool) {
		value, ok := doc.Meta[field]
		return value, ok
	})
}

func matchNode(node graph.Node[contracttest.Meta], cond filter.Condition) (bool, error) {
	return matchFilter(cond.IR(), func(field string) (any, bool) {
		value, ok := node.Meta[field]
		return value, ok
	})
}

func matchEdge(edge graph.Edge[contracttest.Meta], cond filter.Condition) (bool, error) {
	return matchFilter(cond.IR(), func(field string) (any, bool) {
		value, ok := edge.Meta[field]
		return value, ok
	})
}

func matchFilter(expr filter.IR, lookup func(field string) (any, bool)) (bool, error) {
	matcher := &filterMatcher{
		lookup: lookup,
		stack:  nil,
		result: false,
	}
	if err := filter.Walk(expr, matcher); err != nil {
		return false, err
	}

	return matcher.result, nil
}

type matchFrame struct {
	op     string
	values []bool
}

type filterMatcher struct {
	lookup func(string) (any, bool)
	stack  []matchFrame
	result bool
}

func (m *filterMatcher) OnEmpty() error {
	return m.push(true)
}

func (m *filterMatcher) OnEq(field string, value filter.Value) error {
	matched, err := compareEqual(m.lookup, field, value)
	if err != nil {
		return err
	}

	return m.push(matched)
}

func (m *filterMatcher) OnNeq(field string, value filter.Value) error {
	matched, err := compareEqual(m.lookup, field, value)
	if err != nil {
		return err
	}

	return m.push(!matched)
}

func (m *filterMatcher) OnGt(field string, value filter.Value) error {
	return m.pushOrdered(field, value, func(cmp int) bool { return cmp > 0 })
}

func (m *filterMatcher) OnGte(field string, value filter.Value) error {
	return m.pushOrdered(field, value, func(cmp int) bool { return cmp >= 0 })
}

func (m *filterMatcher) OnLt(field string, value filter.Value) error {
	return m.pushOrdered(field, value, func(cmp int) bool { return cmp < 0 })
}

func (m *filterMatcher) OnLte(field string, value filter.Value) error {
	return m.pushOrdered(field, value, func(cmp int) bool { return cmp <= 0 })
}

func (m *filterMatcher) OnIn(field string, values []filter.Value) error {
	for _, value := range values {
		matched, err := compareEqual(m.lookup, field, value)
		if err != nil {
			return err
		}
		if matched {
			return m.push(true)
		}
	}

	return m.push(false)
}

func (m *filterMatcher) EnterAnd(_ int) error {
	m.stack = append(m.stack, matchFrame{op: "and", values: nil})
	return nil
}

func (m *filterMatcher) LeaveAnd() error {
	frame, err := m.popFrame("and")
	if err != nil {
		return err
	}

	result := true
	for _, value := range frame.values {
		if !value {
			result = false
			break
		}
	}

	return m.push(result)
}

func (m *filterMatcher) EnterOr(_ int) error {
	m.stack = append(m.stack, matchFrame{op: "or", values: nil})
	return nil
}

func (m *filterMatcher) LeaveOr() error {
	frame, err := m.popFrame("or")
	if err != nil {
		return err
	}

	result := false
	for _, value := range frame.values {
		if value {
			result = true
			break
		}
	}

	return m.push(result)
}

func (m *filterMatcher) EnterNot() error {
	m.stack = append(m.stack, matchFrame{op: "not", values: nil})
	return nil
}

func (m *filterMatcher) LeaveNot() error {
	frame, err := m.popFrame("not")
	if err != nil {
		return err
	}
	if len(frame.values) != 1 {
		return fmt.Errorf("%w: NOT matcher arity", ragy.ErrUnsupported)
	}

	return m.push(!frame.values[0])
}

func (m *filterMatcher) pushOrdered(field string, value filter.Value, check func(int) bool) error {
	matched, err := compareOrdered(m.lookup, field, value, check)
	if err != nil {
		return err
	}

	return m.push(matched)
}

func (m *filterMatcher) push(value bool) error {
	if len(m.stack) == 0 {
		m.result = value
		return nil
	}

	last := len(m.stack) - 1
	m.stack[last].values = append(m.stack[last].values, value)
	return nil
}

func (m *filterMatcher) popFrame(op string) (matchFrame, error) {
	if len(m.stack) == 0 {
		return matchFrame{}, fmt.Errorf("%w: unmatched filter group", ragy.ErrUnsupported)
	}

	last := len(m.stack) - 1
	frame := m.stack[last]
	m.stack = m.stack[:last]
	if frame.op != op {
		return matchFrame{}, fmt.Errorf("%w: unexpected filter group %q", ragy.ErrUnsupported, frame.op)
	}

	return frame, nil
}

func compareEqual(lookup func(string) (any, bool), field string, expected filter.Value) (bool, error) {
	actual, ok := lookup(field)
	if !ok {
		return false, nil
	}

	switch expected.Kind() {
	case filter.KindString:
		value, ok := actual.(string)
		expectedValue, expectedOK := expected.Raw().(string)
		if !expectedOK {
			return false, fmt.Errorf("%w: invalid string filter value", ragy.ErrUnsupported)
		}
		return ok && value == expectedValue, nil
	case filter.KindBool:
		value, ok := actual.(bool)
		expectedValue, expectedOK := expected.Raw().(bool)
		if !expectedOK {
			return false, fmt.Errorf("%w: invalid bool filter value", ragy.ErrUnsupported)
		}
		return ok && value == expectedValue, nil
	case filter.KindInt:
		value, ok := toInt64(actual)
		expectedValue, expectedOK := expected.Raw().(int64)
		if !expectedOK {
			return false, fmt.Errorf("%w: invalid int filter value", ragy.ErrUnsupported)
		}
		return ok && value == expectedValue, nil
	case filter.KindFloat:
		value, ok := toFloat64(actual)
		expectedValue, expectedOK := expected.Raw().(float64)
		if !expectedOK {
			return false, fmt.Errorf("%w: invalid float filter value", ragy.ErrUnsupported)
		}
		return ok && value == expectedValue, nil
	default:
		return false, fmt.Errorf("%w: unsupported filter kind %q", ragy.ErrUnsupported, expected.Kind())
	}
}

func compareOrdered(
	lookup func(string) (any, bool),
	field string,
	expected filter.Value,
	check func(int) bool,
) (bool, error) {
	actual, ok := lookup(field)
	if !ok {
		return false, nil
	}

	switch expected.Kind() {
	case filter.KindString:
		return false, fmt.Errorf("%w: unsupported ordered filter kind %q", ragy.ErrUnsupported, expected.Kind())
	case filter.KindInt:
		value, ok := toInt64(actual)
		if !ok {
			return false, nil
		}
		expectedValue, expectedOK := expected.Raw().(int64)
		if !expectedOK {
			return false, fmt.Errorf("%w: invalid int filter value", ragy.ErrUnsupported)
		}
		return check(compareInts(value, expectedValue)), nil
	case filter.KindFloat:
		value, ok := toFloat64(actual)
		if !ok {
			return false, nil
		}
		expectedValue, expectedOK := expected.Raw().(float64)
		if !expectedOK {
			return false, fmt.Errorf("%w: invalid float filter value", ragy.ErrUnsupported)
		}
		return check(compareFloats(value, expectedValue)), nil
	case filter.KindBool:
		return false, fmt.Errorf("%w: unsupported ordered filter kind %q", ragy.ErrUnsupported, expected.Kind())
	default:
		return false, fmt.Errorf("%w: unsupported ordered filter kind %q", ragy.ErrUnsupported, expected.Kind())
	}
}

func toInt64(value any) (int64, bool) {
	switch v := value.(type) {
	case int:
		return int64(v), true
	case int8:
		return int64(v), true
	case int16:
		return int64(v), true
	case int32:
		return int64(v), true
	case int64:
		return v, true
	default:
		return 0, false
	}
}

func toFloat64(value any) (float64, bool) {
	switch v := value.(type) {
	case int:
		return float64(v), true
	case int8:
		return float64(v), true
	case int16:
		return float64(v), true
	case int32:
		return float64(v), true
	case int64:
		return float64(v), true
	case float32:
		return float64(v), true
	case float64:
		return v, true
	default:
		return 0, false
	}
}

func compareInts(left, right int64) int {
	switch {
	case left < right:
		return -1
	case left > right:
		return 1
	default:
		return 0
	}
}

func compareFloats(left, right float64) int {
	switch {
	case left < right:
		return -1
	case left > right:
		return 1
	default:
		return 0
	}
}

func minInt(left, right int) int {
	if left < right {
		return left
	}
	return right
}

var (
	_ dense.Embedder                               = (*DenseEmbedder)(nil)
	_ dense.Index[contracttest.Meta]               = (*DenseIndex)(nil)
	_ retrieval.Backend[contracttest.Meta]         = (*RetrievalBackend)(nil)
	_ tensor.Index[contracttest.Meta]              = (*TensorIndex)(nil)
	_ documents.Store[contracttest.Meta]           = (*DocumentStore)(nil)
	_ graph.Store[contracttest.Meta]               = (*GraphStore)(nil)
	_ chunking.ContextGenerator[contracttest.Meta] = (*ContextGenerator)(nil)
)
