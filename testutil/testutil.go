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
	"github.com/skosovsky/ragy/lexical"
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

// DenseSearcher is a fake dense searcher.
type DenseSearcher struct {
	Docs         []ragy.Document
	Err          error
	Requests     []dense.Request
	FilterSchema filter.Schema
}

// Search implements dense.Searcher.
func (s *DenseSearcher) Search(_ context.Context, req dense.Request) ([]ragy.Document, error) {
	s.Requests = append(s.Requests, req)
	if s.Err != nil {
		return nil, s.Err
	}
	if err := req.Validate(); err != nil {
		return nil, err
	}
	if err := s.Schema().ValidateSchemaIR(req.Filter); err != nil {
		return nil, err
	}

	return validateDocuments(s.Docs)
}

// Schema returns the configured filter schema used by the fake searcher.
func (s *DenseSearcher) Schema() filter.Schema {
	return s.FilterSchema
}

// DenseIndex is a fake dense index.
type DenseIndex struct {
	Records      [][]dense.Record
	Err          error
	FilterSchema filter.Schema
}

// Upsert implements dense.Index.
func (i *DenseIndex) Upsert(_ context.Context, records []dense.Record) error {
	if i.Err != nil {
		return i.Err
	}
	if !i.FilterSchema.IsFinalized() {
		return fmt.Errorf("%w: dense index schema", ragy.ErrInvalidArgument)
	}

	copied := make([]dense.Record, len(records))
	for index, record := range records {
		if err := record.Validate(); err != nil {
			return err
		}

		attrs, err := i.FilterSchema.NormalizeAttributes(record.Attributes)
		if err != nil {
			return err
		}

		copied[index] = dense.Record{
			ID:         record.ID,
			Content:    record.Content,
			Attributes: ragy.CloneAttributes(attrs),
			Vector:     append([]float32(nil), record.Vector...),
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
	Records      [][]tensor.Record
	Err          error
	FilterSchema filter.Schema
}

// Upsert implements tensor.Index.
func (i *TensorIndex) Upsert(_ context.Context, records []tensor.Record) error {
	if i.Err != nil {
		return i.Err
	}
	if !i.FilterSchema.IsFinalized() {
		return fmt.Errorf("%w: tensor index schema", ragy.ErrInvalidArgument)
	}

	copied := make([]tensor.Record, len(records))
	for index, record := range records {
		if err := record.Validate(); err != nil {
			return err
		}

		attrs, err := i.FilterSchema.NormalizeAttributes(record.Attributes)
		if err != nil {
			return err
		}

		copied[index] = tensor.Record{
			ID:         record.ID,
			Content:    record.Content,
			Attributes: ragy.CloneAttributes(attrs),
			Tensor:     cloneTensor(record.Tensor),
		}
	}
	i.Records = append(i.Records, copied)
	return nil
}

// Schema returns the configured filter schema used by the fake index.
func (i *TensorIndex) Schema() filter.Schema {
	return i.FilterSchema
}

// LexicalSearcher is a fake lexical searcher.
type LexicalSearcher struct {
	Docs         []ragy.Document
	Err          error
	Requests     []lexical.Request
	FilterSchema filter.Schema
}

// Search implements lexical.Searcher.
func (s *LexicalSearcher) Search(_ context.Context, req lexical.Request) ([]ragy.Document, error) {
	s.Requests = append(s.Requests, req)
	if s.Err != nil {
		return nil, s.Err
	}
	if err := req.Validate(); err != nil {
		return nil, err
	}
	if err := s.Schema().ValidateSchemaIR(req.Filter); err != nil {
		return nil, err
	}

	return validateDocuments(s.Docs)
}

// Schema returns the configured filter schema used by the fake searcher.
func (s *LexicalSearcher) Schema() filter.Schema {
	return s.FilterSchema
}

// TensorSearcher is a fake tensor searcher.
type TensorSearcher struct {
	Docs         []ragy.Document
	Err          error
	Requests     []tensor.Request
	FilterSchema filter.Schema
}

// Search implements tensor.Searcher.
func (s *TensorSearcher) Search(_ context.Context, req tensor.Request) ([]ragy.Document, error) {
	s.Requests = append(s.Requests, req)
	if s.Err != nil {
		return nil, s.Err
	}
	if err := req.Validate(); err != nil {
		return nil, err
	}
	if err := s.Schema().ValidateSchemaIR(req.Filter); err != nil {
		return nil, err
	}

	return validateDocuments(s.Docs)
}

// Schema returns the configured filter schema used by the fake searcher.
func (s *TensorSearcher) Schema() filter.Schema {
	return s.FilterSchema
}

// DocumentStore is a memory-backed documents.Store fake.
type DocumentStore struct {
	Docs         []ragy.Document
	Err          error
	FindCalls    [][]string
	FilterSchema filter.Schema
}

// FindByIDs implements documents.Store.
func (s *DocumentStore) FindByIDs(_ context.Context, ids []string) ([]ragy.Document, error) {
	s.FindCalls = append(s.FindCalls, append([]string(nil), ids...))
	if s.Err != nil {
		return nil, s.Err
	}

	if len(ids) == 0 {
		return nil, nil
	}

	byID := make(map[string]ragy.Document, len(s.Docs))
	for _, doc := range s.Docs {
		byID[doc.ID] = cloneDocument(doc)
	}

	out := make([]ragy.Document, 0, len(ids))
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

	return validateDocuments(out)
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
	kept := make([]ragy.Document, 0, len(s.Docs))
	for _, doc := range s.Docs {
		if _, ok := remove[doc.ID]; ok {
			deleted++
			continue
		}
		kept = append(kept, cloneDocument(doc))
	}

	s.Docs = kept
	return documents.DeleteResult{Deleted: deleted}, nil
}

// DeleteByFilter implements documents.Store.
func (s *DocumentStore) DeleteByFilter(_ context.Context, expr filter.IR) (documents.DeleteResult, error) {
	if s.Err != nil {
		return documents.DeleteResult{}, s.Err
	}

	if expr == nil {
		return documents.DeleteResult{}, fmt.Errorf("%w: delete filter", ragy.ErrInvalidArgument)
	}
	if filter.IsEmpty(expr) {
		return documents.DeleteResult{}, fmt.Errorf("%w: delete filter", ragy.ErrInvalidArgument)
	}
	if err := s.Schema().ValidateSchemaIR(expr); err != nil {
		return documents.DeleteResult{}, err
	}

	deleted := 0
	kept := make([]ragy.Document, 0, len(s.Docs))
	for _, doc := range s.Docs {
		matched, err := matchDocument(doc, expr)
		if err != nil {
			return documents.DeleteResult{}, err
		}
		if matched {
			deleted++
			continue
		}
		kept = append(kept, cloneDocument(doc))
	}

	s.Docs = kept
	return documents.DeleteResult{Deleted: deleted}, nil
}

// Schema returns the configured filter schema used by the fake store.
func (s *DocumentStore) Schema() filter.Schema {
	return s.FilterSchema
}

// GraphStore is a memory-backed graph.Store fake.
type GraphStore struct {
	Snapshot    graph.Snapshot
	GraphSchema graph.Schema
	Err         error
	Requests    []graph.TraversalRequest
}

// Traverse implements graph.Store.
func (s *GraphStore) Traverse(_ context.Context, req graph.TraversalRequest) (graph.Snapshot, error) {
	s.Requests = append(s.Requests, cloneTraversalRequest(req))
	if s.Err != nil {
		return graph.Snapshot{}, s.Err
	}
	if err := s.GraphSchema.ValidateTraversal(req); err != nil {
		return graph.Snapshot{}, err
	}
	snapshot, err := s.GraphSchema.NormalizeSnapshot(s.Snapshot)
	if err != nil {
		return graph.Snapshot{}, err
	}

	out, err := traverseSnapshot(snapshot, req)
	if err != nil {
		return graph.Snapshot{}, err
	}

	return s.GraphSchema.NormalizeSnapshot(out)
}

// Upsert implements graph.Store.
func (s *GraphStore) Upsert(_ context.Context, snapshot graph.Snapshot) error {
	if s.Err != nil {
		return s.Err
	}
	normalized, err := s.GraphSchema.NormalizeSnapshot(snapshot)
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
func (g *ContextGenerator) Context(_ context.Context, _ ragy.Document, _ ragy.Chunk) (string, error) {
	return g.Value, g.Err
}

// GraphProvider is a fake graph extraction provider.
type GraphProvider struct {
	Snapshot graph.Snapshot
	Err      error
}

// Extract extracts a graph snapshot from chunks.
func (p *GraphProvider) Extract(_ context.Context, _ []ragy.Chunk) (graph.Snapshot, error) {
	return cloneSnapshot(p.Snapshot), p.Err
}

func validateDocuments(in []ragy.Document) ([]ragy.Document, error) {
	if len(in) == 0 {
		return nil, nil
	}

	out := make([]ragy.Document, len(in))
	for i, doc := range in {
		normalized, err := ragy.NormalizeDocument(doc)
		if err != nil {
			return nil, err
		}
		out[i] = normalized
	}

	return out, nil
}

func cloneDocument(in ragy.Document) ragy.Document {
	return ragy.Document{
		ID:         in.ID,
		Content:    in.Content,
		Attributes: ragy.CloneAttributes(in.Attributes),
		Relevance:  in.Relevance,
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

func cloneNode(in graph.Node) graph.Node {
	return graph.Node{
		ID:         in.ID,
		Labels:     append([]string(nil), in.Labels...),
		Content:    in.Content,
		Attributes: ragy.CloneAttributes(in.Attributes),
	}
}

func cloneEdge(in graph.Edge) graph.Edge {
	return graph.Edge{
		ID:         in.ID,
		SourceID:   in.SourceID,
		TargetID:   in.TargetID,
		Type:       in.Type,
		Attributes: ragy.CloneAttributes(in.Attributes),
	}
}

func cloneSnapshot(in graph.Snapshot) graph.Snapshot {
	out := graph.Snapshot{
		Nodes: make([]graph.Node, len(in.Nodes)),
		Edges: make([]graph.Edge, len(in.Edges)),
	}
	for i := range in.Nodes {
		out.Nodes[i] = cloneNode(in.Nodes[i])
	}
	for i := range in.Edges {
		out.Edges[i] = cloneEdge(in.Edges[i])
	}
	return out
}

func mergeSnapshot(base, incoming graph.Snapshot) graph.Snapshot {
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

func traverseSnapshot(snapshot graph.Snapshot, req graph.TraversalRequest) (graph.Snapshot, error) {
	nodesByID := indexNodes(snapshot.Nodes)
	visitedNodes, frontier := seedFrontier(nodesByID, req.Seeds)
	visitedEdges, err := expandTraversal(snapshot.Edges, nodesByID, visitedNodes, frontier, req)
	if err != nil {
		return graph.Snapshot{}, err
	}

	nodes, allowedNodes, err := projectNodes(snapshot.Nodes, visitedNodes, req.NodeFilter)
	if err != nil {
		return graph.Snapshot{}, err
	}
	if req.Page != nil {
		nodes, allowedNodes = pageNodes(nodes, req.Page)
	}

	edges, err := projectEdges(snapshot.Edges, visitedEdges, allowedNodes, req.EdgeFilter)
	if err != nil {
		return graph.Snapshot{}, err
	}

	return graph.Snapshot{Nodes: nodes, Edges: edges}, nil
}

func traversesEdge(edge graph.Edge, current string, direction graph.Direction) (bool, string) {
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

func indexNodes(nodes []graph.Node) map[string]graph.Node {
	out := make(map[string]graph.Node, len(nodes))
	for _, node := range nodes {
		out[node.ID] = node
	}
	return out
}

func seedFrontier(nodesByID map[string]graph.Node, seeds []string) (map[string]struct{}, []string) {
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
	edges []graph.Edge,
	nodesByID map[string]graph.Node,
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
	edges []graph.Edge,
	nodesByID map[string]graph.Node,
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
	nodes []graph.Node,
	visited map[string]struct{},
	expr filter.IR,
) ([]graph.Node, map[string]struct{}, error) {
	out := make([]graph.Node, 0, len(visited))
	allowed := make(map[string]struct{}, len(visited))
	for _, node := range nodes {
		if _, ok := visited[node.ID]; !ok {
			continue
		}

		matched, err := matchNode(node, expr)
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

func pageNodes(nodes []graph.Node, page *ragy.Page) ([]graph.Node, map[string]struct{}) {
	start := minInt(page.Offset, len(nodes))
	end := minInt(start+page.Limit, len(nodes))
	paged := append([]graph.Node(nil), nodes[start:end]...)
	allowed := make(map[string]struct{}, len(paged))
	for _, node := range paged {
		allowed[node.ID] = struct{}{}
	}
	return paged, allowed
}

func projectEdges(
	edges []graph.Edge,
	visited map[string]struct{},
	allowedNodes map[string]struct{},
	expr filter.IR,
) ([]graph.Edge, error) {
	out := make([]graph.Edge, 0, len(visited))
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

		matched, err := matchEdge(edge, expr)
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

func matchDocument(doc ragy.Document, expr filter.IR) (bool, error) {
	return matchFilter(expr, func(field string) (any, bool) {
		value, ok := doc.Attributes[field]
		return value, ok
	})
}

func matchNode(node graph.Node, expr filter.IR) (bool, error) {
	return matchFilter(expr, func(field string) (any, bool) {
		value, ok := node.Attributes[field]
		return value, ok
	})
}

func matchEdge(edge graph.Edge, expr filter.IR) (bool, error) {
	return matchFilter(expr, func(field string) (any, bool) {
		value, ok := edge.Attributes[field]
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
	_ dense.Embedder            = (*DenseEmbedder)(nil)
	_ dense.Searcher            = (*DenseSearcher)(nil)
	_ dense.Index               = (*DenseIndex)(nil)
	_ lexical.Searcher          = (*LexicalSearcher)(nil)
	_ tensor.Searcher           = (*TensorSearcher)(nil)
	_ tensor.Index              = (*TensorIndex)(nil)
	_ documents.Store           = (*DocumentStore)(nil)
	_ graph.Store               = (*GraphStore)(nil)
	_ chunking.ContextGenerator = (*ContextGenerator)(nil)
)
