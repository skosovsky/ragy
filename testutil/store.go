package testutil

import (
	"context"
	"iter"
	"reflect"
	"sort"
	"sync"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/internal/mathutil"
)

// EmbeddingKey is the Metadata key for dense embedding in tests. Same as ragy.EmbeddingMetadataKey (type []float32).
const EmbeddingKey = ragy.EmbeddingMetadataKey

// TensorKey is the Metadata key for tensor (per-token) embedding. Type: [][]float32.
const TensorKey = "tensor"

// InMemoryVectorStore implements ragy.VectorStore with brute-force cosine similarity.
// For Upsert: store documents; optionally set Metadata[EmbeddingKey] to []float32 for dense search,
// or Metadata[TensorKey] to [][]float32 for tensor (ColBERT-style) search.
// Search uses req.DenseVector or req.TensorVector; applies Limit and Offset.
type InMemoryVectorStore struct {
	mu   sync.RWMutex
	docs map[string]ragy.Document
}

// NewInMemoryVectorStore returns a new in-memory vector store.
func NewInMemoryVectorStore() *InMemoryVectorStore {
	return &InMemoryVectorStore{
		docs: make(map[string]ragy.Document),
	}
}

// Search implements ragy.VectorStore.
func (s *InMemoryVectorStore) Search(_ context.Context, req ragy.SearchRequest) ([]ragy.Document, error) {
	s.mu.RLock()
	list := make([]ragy.Document, 0, len(s.docs))
	for _, d := range s.docs {
		list = append(list, d)
	}
	s.mu.RUnlock()

	if len(list) == 0 {
		return nil, nil
	}
	if req.Filter != nil {
		filtered := list[:0]
		for _, d := range list {
			if matchDoc(d, req.Filter) {
				filtered = append(filtered, d)
			}
		}
		list = filtered
		if len(list) == 0 {
			return nil, nil
		}
	}

	type scored struct {
		doc   ragy.Document
		score float32
	}
	var scoredList []scored

	switch {
	case len(req.DenseVector) > 0:
		for _, d := range list {
			emb, _ := d.Metadata[EmbeddingKey].([]float32)
			if len(emb) != len(req.DenseVector) {
				continue
			}
			score := mathutil.CosineSimilarity(req.DenseVector, emb)
			scoredList = append(scoredList, scored{doc: d, score: score})
		}
	case len(req.TensorVector) > 0:
		for _, d := range list {
			docTensor, _ := d.Metadata[TensorKey].([][]float32)
			score := tensorSimilarity(req.TensorVector, docTensor)
			scoredList = append(scoredList, scored{doc: d, score: score})
		}
	default:
		return nil, nil
	}

	sort.Slice(scoredList, func(i, j int) bool {
		return scoredList[i].score > scoredList[j].score
	})

	limit := req.Limit
	if limit <= 0 {
		limit = 10
	}
	offset := max(req.Offset, 0)
	start := offset
	end := offset + limit
	if start >= len(scoredList) {
		return nil, nil
	}
	if end > len(scoredList) {
		end = len(scoredList)
	}
	out := make([]ragy.Document, 0, end-start)
	for i := start; i < end; i++ {
		d := scoredList[i].doc
		s := scoredList[i].score
		d.Score = s
		// Cosine similarity is in [-1,1]; map to Confidence [0,1]
		d.Confidence = float64((s + 1) / 2)
		if d.Confidence < 0 {
			d.Confidence = 0
		}
		if d.Confidence > 1 {
			d.Confidence = 1
		}
		out = append(out, d)
	}
	return out, nil
}

// Stream implements ragy.VectorStore.
func (s *InMemoryVectorStore) Stream(ctx context.Context, req ragy.SearchRequest) iter.Seq2[ragy.Document, error] {
	docs, err := s.Search(ctx, req)
	return ragy.YieldDocuments(ctx, docs, err)
}

// matchDoc returns true if the document metadata matches the filter expression.
// Supports only Eq and And; other expression types return false so tests fail explicitly
// instead of silently including all documents.
func matchDoc(doc ragy.Document, expr filter.Expr) bool {
	switch e := expr.(type) {
	case filter.Eq:
		v, ok := doc.Metadata[e.Field]
		if !ok {
			return false
		}
		return reflect.DeepEqual(v, e.Value)
	case filter.And:
		for _, sub := range e.Exprs {
			if !matchDoc(doc, sub) {
				return false
			}
		}
		return true
	default:
		return false
	}
}

// tensorSimilarity computes a simple late-interaction style score: for each query token,
// max cosine similarity with any doc token; then average over query tokens.
func tensorSimilarity(queryTokens, docTokens [][]float32) float32 {
	if len(queryTokens) == 0 || len(docTokens) == 0 {
		return 0
	}
	var sum float32
	for _, q := range queryTokens {
		var maxCos float32
		for _, d := range docTokens {
			if len(q) != len(d) {
				continue
			}
			c := mathutil.CosineSimilarity(q, d)
			if c > maxCos {
				maxCos = c
			}
		}
		sum += maxCos
	}
	return sum / float32(len(queryTokens))
}

// Upsert implements ragy.VectorStore.
func (s *InMemoryVectorStore) Upsert(_ context.Context, docs []ragy.Document) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, d := range docs {
		s.docs[d.ID] = d
	}
	return nil
}

// DeleteByFilter implements ragy.VectorStore.
// For test use only: ignores expr structure and clears all documents when f is not nil.
// Does nothing if f is nil. For real filter support adapters must traverse filter.Expr.
func (s *InMemoryVectorStore) DeleteByFilter(_ context.Context, f filter.Expr) error {
	if f == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.docs = make(map[string]ragy.Document)
	return nil
}

// Ensure InMemoryVectorStore implements ragy.VectorStore.
var _ ragy.VectorStore = (*InMemoryVectorStore)(nil)

// InMemoryGraphStore implements ragy.GraphStore with an in-memory graph and BFS.
type InMemoryGraphStore struct {
	mu    sync.RWMutex
	nodes map[string]ragy.Node
	edges []ragy.Edge
	// edgesBySource: source ID -> list of edge indices (into edges)
	edgesBySource map[string][]int
}

// NewInMemoryGraphStore returns a new in-memory graph store.
func NewInMemoryGraphStore() *InMemoryGraphStore {
	return &InMemoryGraphStore{
		nodes:         make(map[string]ragy.Node),
		edges:         nil,
		edgesBySource: make(map[string][]int),
	}
}

// SearchGraph implements ragy.GraphStore. Performs BFS from the given entities up to depth.
func (s *InMemoryGraphStore) SearchGraph(_ context.Context, entities []string, depth int, _ ragy.SearchRequest) ([]ragy.Node, []ragy.Edge, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	seenNodes := make(map[string]struct{})
	seenEdges := make(map[string]struct{})
	var nodes []ragy.Node
	var edges []ragy.Edge

	type item struct {
		id    string
		depth int
	}
	queue := make([]item, 0, len(entities)*2)
	for _, e := range entities {
		queue = append(queue, item{e, 0})
	}

	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		if cur.depth > depth {
			continue
		}
		if n, ok := s.nodes[cur.id]; ok {
			if _, ok := seenNodes[n.ID]; !ok {
				seenNodes[n.ID] = struct{}{}
				nodes = append(nodes, n)
			}
		}
		for _, idx := range s.edgesBySource[cur.id] {
			if idx >= len(s.edges) {
				continue
			}
			e := s.edges[idx]
			edgeKey := e.SourceID + ":" + e.TargetID + ":" + e.Relation
			if _, ok := seenEdges[edgeKey]; !ok {
				seenEdges[edgeKey] = struct{}{}
				edges = append(edges, e)
			}
			if cur.depth < depth {
				queue = append(queue, item{e.TargetID, cur.depth + 1})
			}
		}
	}
	return nodes, edges, nil
}

// UpsertGraph implements ragy.GraphStore.
func (s *InMemoryGraphStore) UpsertGraph(_ context.Context, nodes []ragy.Node, edges []ragy.Edge) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, n := range nodes {
		s.nodes[n.ID] = n
	}
	for _, e := range edges {
		s.edges = append(s.edges, e)
		idx := len(s.edges) - 1
		s.edgesBySource[e.SourceID] = append(s.edgesBySource[e.SourceID], idx)
	}
	return nil
}

var _ ragy.GraphStore = (*InMemoryGraphStore)(nil)
